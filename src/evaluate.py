from typing import Optional
import copy
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm

from utils import (
    get_first_token_of_answers,
    get_conditions_for_capitals_recognition_experiment
)


def _evaluate_single_item(
    model, 
    prefix: str, 
    answer_texts: list[str],
    answer_labels: list[str] = ["correct", "intuitive"],
    meta_data: Optional[dict] = None
):
    """
    Helper function that implements evaluation of a single stimulus.
    """
    # Check that answer texts correspond to labels.
    assert len(answer_texts) == len(answer_labels)

    # Set up the meta data, and record the input text that 
    # the model conditions on (the "prefix").
    if meta_data is None:
        meta_data = {}
    meta_data["model_input"] = prefix

    # Initialize results.
    layerwise_results = []

    # (1) Get rank of the first token of each answer option, at each layer.
    first_tokens = get_first_token_of_answers(
        model.model.tokenizer,
        prefix,
        answer_texts,
        sep=" "
    )
    all_ranks = model.rank_of_token_all_layers(prefix, first_tokens)
        
    # (2) Compute scores for each answer option, conditioned on the same prefix.
    # Each entry of `all_scores` is a dict with the following structure:
    #   * keys: "entropy", "sum", "mean", "first", "logits", "logits_deltas"
    #   * values: numpy array
    # For "sum", "mean", "first", and "entropy", the value has shape (n_layers,)
    # and for "logits" and "logits_deltas" the shape is (n_layers, N) where
    # `N` is the length (# tokens) of the longest answer string
    all_scores = [
        model.conditional_score_all_layers(prefix, answer, sep=" ")
        for answer in answer_texts
    ]

    # (3) Process layerwise ranks and scores into final clean format.
    n_layers = len(model.model.model.layers)
    for layer_idx in range(n_layers):
        # Initialize results dictionary with relevant metadata.
        res = copy.deepcopy(meta_data)
        res["layer_idx"] = layer_idx

        # Record entropy scores.
        # These should be the same across both answer options, within
        # some small tolerance depending on floating point precision.
        res["entropy_first_token"] = all_scores[0]["entropy"][layer_idx]

        # Record the ranks and unnormalized logits (one float per token per item).
        for answer_idx, answer_label in enumerate(answer_labels):
            # Record rank.
            res[f"rank_{answer_label}_first_token"] = all_ranks[answer_idx][layer_idx]
            # Record logits.
            logits = all_scores[answer_idx]["logits"][layer_idx].tolist()
            res[f"logits_{answer_label}"] = logits
            # Record logit deltas (for layer_idx = 0...L-2).
            if layer_idx + 1 < n_layers:
                logits_deltas = all_scores[answer_idx]["logits_deltas"][layer_idx].tolist()
                res[f"logits_deltas_{answer_label}"] = logits_deltas

        # Record logprob scores in dictionary (one float per item).
        for reduction_type in ["mean", "sum", "first"]:
            metric = f"{reduction_type}_logprob"
            for answer_idx, answer_label in enumerate(answer_labels):
                # Record the logprob scores, under the current reduction type.
                logprob_scores = all_scores[answer_idx][reduction_type][layer_idx]
                res[f"{metric}_{answer_label}"] = logprob_scores
            # Compute the argmax score to obtain the model's "chosen" answer.
            top_option = answer_labels[
                np.argmax([s[reduction_type][layer_idx] for s in all_scores])
            ]
            res[f"{metric}_response"] = top_option
            res[f"{metric}_response_isCorrect"] = (top_option == "correct")

        layerwise_results.append(res)

    return layerwise_results

def evaluate(
    model,
    stimuli: pd.DataFrame, 
    task: str = "capitals-recall",
    prompts: Optional[pd.DataFrame] = None,
    # use_chat: bool = False,
    run_controls: bool = False,
    **prompt_kws
) -> pd.DataFrame:
    """
    Wrapper function that implements LM evaluation of a full set of stimuli.
    """
    # Initialize results.
    all_scores = []

    # Answer options corresponding to columns in `stimuli` that will be used
    # for evaluation.
    main_answer_labels = ["correct", "incorrect"] if task == "syllogism" else ["correct", "intuitive"]

    # Answer options for control analyses.
    control_answer_labels = ["control-irrelevant", "control-relevant"]

    # Specify meta variables related to stimuli that we want to record in the 
    # final results. By default, this is all columns in the dataframe.
    if task == "colors":
        # Remove all the fact data, to avoid redundancy.
        stimulus_meta_vars = [
            c for c in stimuli.columns
            if not (c.startswith("fact_") and c != "fact_color_critical")
        ]
    elif task == "animals":
        stimulus_meta_vars = [c for c in stimuli.columns if not c.endswith("_de")]
    else:
        stimulus_meta_vars = stimuli.columns
    
    # Specify meta variables related to prompts sentences.
    if prompts is None:
        prompt_meta_vars = {}
    else:
        prompt_meta_vars = prompts.columns

    # Iterate over all stimuli.
    for _, stim_row in tqdm(stimuli.iterrows(), total=len(stimuli.index)):
        # Aggregate meta data for each item for evaluation.
        stim_meta_data = {v: stim_row[v] for v in stimulus_meta_vars}
        if prompts is None:
            item_meta_data = [stim_meta_data]
        else:
            # Add meta data for each prompt to the meta data for the stimulus.
            item_meta_data = [
                stim_meta_data | {v: prompt_row[v] for v in prompt_meta_vars}
                for _, prompt_row in prompts.iterrows()
            ]
        # For the `colors` and `capitals-recognition` tasks, we additionally 
        # need to do multiple conditions for each stimulus*prompt combination.
        if task == "colors":
            # Get list of conditions for evaluation.         
            conditions = get_conditions_for_color_experiment(stim_row)
            item_meta_data = [
                meta_data | condition
                for meta_data in item_meta_data for condition in conditions
            ]
        elif task == "capitals-recognition":
            conditions = get_conditions_for_capitals_recognition_experiment(stim_row)
            item_meta_data = [
                meta_data | condition
                for meta_data in item_meta_data for condition in conditions
            ]
            
        # Get texts of answer options.
        main_answer_texts = [str(stim_row[label]) for label in main_answer_labels]
        irrelevant_text = "."
        if task.startswith("capitals"):
            relevant_text = stim_row["entity"]
        elif task == "animals":
            relevant_text = stim_row["exemplar"]
        elif task == "syllogism":
            relevant_text = ""
        else:
            raise ValueError(f"Undefined relevant distractor text for task '{task}'")
        control_answer_texts = [irrelevant_text, relevant_text]

        # Combine main answers and control answers.
        answer_labels = main_answer_labels # + control_answer_labels
        answer_texts = main_answer_texts  #+ control_answer_texts

        # Finally, loop over all items for evaluation.
        for item in item_meta_data:
            # Get prompt that precedes the actual problem statement.
            if prompts is None or item["trigger"] == None:
                prompt = "" 
            else:
                prompt = item["trigger"]

            # Simply combine the prompt/instructions with the answer prefix.
            if task == "colors":
                query = item["query"] + " " + item["prefix"]
            elif task == "capitals-recognition":
                query = item["query"] + " " + item["prefix"][0].lower() + item["prefix"][1:]
            else:
                query = item["prefix"]
            prefix = (prompt + " " + query).strip()

            print("* PREFIX FOR EVALUATION:", prefix)
            print("* ANSWER OPTIONS:", list(zip(answer_labels, answer_texts)))

            # Run evaluation!
            layerwise_scores = _evaluate_single_item(
                model,
                prefix,
                answer_texts,
                answer_labels=answer_labels,
                meta_data=item
            )

            # Optionally run controls.
            if run_controls:
                if task.startswith("capitals"):
                    control_prefix = "The capital"
                elif task == "animals":
                    control_prefix = prefix.split()[0]
                elif task == "syllogism":
                    control_prefix = "Argument:"
                layerwise_scores_controls = _evaluate_single_item(
                    model,
                    control_prefix,
                    answer_texts,
                    answer_labels=answer_labels,
                    meta_data=item
                )
                
                for layer_idx, layer_scores in enumerate(layerwise_scores):
                    # Get name of variables that are related to the results themselves,
                    # not meta data contained in the `item` dict.
                    result_vars = [
                        k for k in layer_scores.keys() if k not in item.keys()
                        and k != "layer_idx"
                    ]

                    # Add control results.
                    for r in result_vars:
                        layer_scores[f"control_{r}"] = layerwise_scores_controls[layer_idx][r]
                    layer_scores["control_model_input"] = control_prefix
                    
                    # Update scores.
                    layerwise_scores[layer_idx] = layer_scores

            all_scores += layerwise_scores

    scores_df = pd.DataFrame(all_scores)
    scores_df["task"] = task
    return scores_df