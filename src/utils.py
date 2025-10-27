# File: utils.py
# Description: helper functions used during LM evaluation

import os
import numpy as np


HF_TOKEN = os.getenv("HF_TOKEN")
TASKS = [
    "capitals-recall", 
    "capitals-recognition",
    "animals",
    "gender",
    "syllogism",
    "colors"
]
TL_MODELS = [
    "gpt2", "gpt2-xl", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"
]

def flatten(xss: list) -> list:
    """Helper function for flattening a list."""
    return [x for xs in xss for x in xs]

def get_reduction_fn(reduction: str):
    """Returns an anonymous function that applies a reduction to a Tensor."""
    if reduction == "mean":
        reduction_fn = lambda x: x.mean(0).item()
    elif reduction == "sum":
        reduction_fn = lambda x: x.sum(0).item()
    elif reduction == "mean_and_sum":
        reduction_fn = lambda x: (x.mean(0).item(), x.sum(0).item())
    else:
        raise ValueError("`reduction` should be 'mean', 'sum', or 'mean_and_sum")
    return reduction_fn

def get_file_safe_model_name(model: str) -> str:
    """
    Returns a file-safe version of a Huggingface model identifier by
    only keeping the model name after a forward slash (/).
    Example: meta-llama/Llama-2-7b-hf --> Llama-2-7b-hf
    """
    safe_model_name = model.split("/")[-1] if "/" in model else model
    return safe_model_name

def get_rank(x, indices, one_indexed=True):
    """
    Adapted from https://stephantul.github.io/python/pytorch/2020/09/18/fast_topk/
    """
    vals = x[range(len(x)), indices]
    rank = (x > vals[:, None]).long().sum(1)
    if one_indexed:
        rank += 1
    return rank

def get_model_family(model_name):
    model_name = model_name.lower()
    families = ["llama", "olmo", "gemma", "gpt", "falcon", "mamba"]
    for family in families:
        if family in model_name:
            return family
        else:
            continue
    raise ValueError(f"Unrecognized model family for {model_name}")

def get_vals_of_tokens(vals, tokens):
    """
    Helper function for taking a Tensor of vocabulary-related values `vals`
    of shape (n_tokens x n_layers x vocab_size), extracting the values 
    corresponding to each token ID in `tokens`, and reshaping to be of shape
    (n_layers x n_tokens)
    """
    n_layers = vals.size(1)
    all_layer_vals = np.array([
        [
            vals[i][layer_id][token_id].detach().cpu()
            for i, token_id in enumerate(tokens)
        ]
        for layer_id in range(n_layers)
    ])
    return all_layer_vals

def get_first_token_of_answers(
    tokenizer,
    prefix,
    answer_texts,
    sep: str = " "
):
    # Construct full texts by combined prefix with answer texts.
    full_texts = [prefix + sep + answer for answer in answer_texts]
    full_tokens = tokenizer(full_texts, add_special_tokens=False)["input_ids"]
    # "Prefix" tokens by combining prefix and continuation separately.
    prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    # Answer tokens correspond to everything after the prefix.
    answer_tokens = [t[len(prefix_tokens):] for t in full_tokens]
    # Get first token of each answer.
    first_tokens = [tokenized[0] for tokenized in answer_tokens]
    return first_tokens

def get_conditions_for_capitals_recognition_experiment(stim_row) -> list[dict]:
    """
    Return list of dictionaries defining conditions for evaluation
    in the capitals recognition (forced choice) experiment.
    """
    conditions = []
    for correct_first in [True, False]:
        if correct_first:
            options = [stim_row["correct"], stim_row["incorrect"]]
        else:
            options = [stim_row["incorrect"], stim_row["correct"]]
        query = f"""The capital of {stim_row.entity} is either {options[0]} or {options[1]}. In fact,"""
        conditions.append(dict(
            correct_first=correct_first,
            query=query
        ))
    return conditions

def get_conditions_for_color_experiment(stim_row) -> list[dict]:
    """
    Return list of dictionaries defining conditions for evaluation
    in the colors experiment.
    
    This function creates multiple experimental conditions by varying:
    1. Number of intervening facts (0-5 facts between critical info and question)
    2. Type of intervening facts (normal vs. strange)
    
    The hypothesis:
    - More intervening facts -> model more likely to forget critical info
    - Normal facts -> bias toward intuitive (prior knowledge) answer
    - Strange facts -> less bias toward intuitive answer
    """
    conditions = []
    
    # Available fact types in the order they appear in the CSV
    fact_types = ["appearance", "type", "subtype", "place", "size"]
    
    # Baseline condition: no intervening facts
    query = stim_row["prefix"] + " " + stim_row["fact_color_critical"] + " " + stim_row["question"]
    conditions.append(dict(
        query=query,
        num_intervening_facts=0,
        fact_type_condition="none",
        prefix=stim_row["prefix"]
    ))
    
    # Conditions with 1 intervening fact (test each fact type separately)
    for fact_type in fact_types:
        # Normal fact
        fact_col_normal = f"fact_{fact_type}_normal"
        if fact_col_normal in stim_row and stim_row[fact_col_normal]:
            query = (stim_row["prefix"] + " " + 
                    stim_row["fact_color_critical"] + " " + 
                    stim_row[fact_col_normal] + " " + 
                    stim_row["question"])
            conditions.append(dict(
                query=query,
                num_intervening_facts=1,
                fact_type_condition=f"{fact_type}_normal",
                prefix=stim_row["prefix"]
            ))
        
        # Strange fact
        fact_col_strange = f"fact_{fact_type}_strange"
        if fact_col_strange in stim_row and stim_row[fact_col_strange]:
            query = (stim_row["prefix"] + " " + 
                    stim_row["fact_color_critical"] + " " + 
                    stim_row[fact_col_strange] + " " + 
                    stim_row["question"])
            conditions.append(dict(
                query=query,
                num_intervening_facts=1,
                fact_type_condition=f"{fact_type}_strange",
                prefix=stim_row["prefix"]
            ))
    
    # Conditions with all 5 normal facts
    normal_facts = [stim_row[f"fact_{fact_type}_normal"] 
                   for fact_type in fact_types 
                   if f"fact_{fact_type}_normal" in stim_row and stim_row[f"fact_{fact_type}_normal"]]
    if normal_facts:
        query = (stim_row["prefix"] + " " + 
                stim_row["fact_color_critical"] + " " + 
                " ".join(normal_facts) + " " + 
                stim_row["question"])
        conditions.append(dict(
            query=query,
            num_intervening_facts=len(normal_facts),
            fact_type_condition="all_normal",
            prefix=stim_row["prefix"]
        ))
    
    # Conditions with all 5 strange facts
    strange_facts = [stim_row[f"fact_{fact_type}_strange"] 
                    for fact_type in fact_types 
                    if f"fact_{fact_type}_strange" in stim_row and stim_row[f"fact_{fact_type}_strange"]]
    if strange_facts:
        query = (stim_row["prefix"] + " " + 
                stim_row["fact_color_critical"] + " " + 
                " ".join(strange_facts) + " " + 
                stim_row["question"])
        conditions.append(dict(
            query=query,
            num_intervening_facts=len(strange_facts),
            fact_type_condition="all_strange",
            prefix=stim_row["prefix"]
        ))
    
    # Mixed condition: alternate normal and strange facts
    # Pattern: normal, strange, normal, strange, normal
    if len(normal_facts) >= 3 and len(strange_facts) >= 2:
        mixed_facts = [
            normal_facts[0],   # appearance_normal
            strange_facts[0],  # type_strange
            normal_facts[2],   # subtype_normal
            strange_facts[3],  # place_strange
            normal_facts[4]    # size_normal
        ]
        query = (stim_row["prefix"] + " " + 
                stim_row["fact_color_critical"] + " " + 
                " ".join(mixed_facts) + " " + 
                stim_row["question"])
        conditions.append(dict(
            query=query,
            num_intervening_facts=5,
            fact_type_condition="mixed",
            prefix=stim_row["prefix"]
        ))
    
    return conditions
