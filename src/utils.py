# File: utils.py
# Description: helper functions used during LM evaluation

import os
import torch

from model import LlamaWrapper


HF_TOKEN = os.getenv("HF_TOKEN")

TASKS = [
    "capitals-recall", 
    "capitals-recognition",
    "animals",
    "syllogism"
]

def initialize_lm(
    model_name, 
    reduce_precision=False,
    cache_dir=None
):
    """Initializes language model for experiments."""
    kws = dict(
        device_map="auto",
        cache_dir=cache_dir
    )
    if reduce_precision:
        kws["torch_dtype"] = torch.float16
    model = LlamaWrapper(model_name, **kws)
    return model

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
    safe_model_name = model.split("/")[1] if "/" in model else model
    return safe_model_name

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
            condition_name = "correct_first"
            options = [stim_row["correct"], stim_row["intuitive"]]
        else:
            condition_name = "intuitive_first"
            options = [stim_row["intuitive"], stim_row["correct"]]
        query = f"""The capital of {stim_row.entity} is either {options[0]} or {options[1]}. In fact,"""
        conditions.append(dict(
            condition=condition_name,
            correct_first=correct_first,
            query=query
        ))
    return conditions
