# File: run_experiment.py
# Description: main wrapper script that should be called to evaluate LMs

import argparse
import os
import pandas as pd

import evaluate
from utils import TASKS, initialize_lm, get_file_safe_model_name


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for running LM experiments."""
    parser = argparse.ArgumentParser()
    # File-related parameters
    parser.add_argument("--stimuli_dir", type=str, default="data/stimuli", 
                        help="Path to folder containing stimuli")
    parser.add_argument("-o", "--output_dir", type=str, default="model_output", 
                        help="Path to directory where output files will be written")
    parser.add_argument("--cache_dir", type=str, 
                        help="Path to Huggingface cache")
    parser.add_argument("--prompt_file", default=None, type=str, 
                        help="Path to CSV file containing prompt contrasts")
    # Model-related parameters
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Name of Huggingface model identifier")
    parser.add_argument("--reduce_precision", default=False, action="store_true")
    # Experiment-related parameters
    parser.add_argument("--task", type=str, default=None, nargs="+", choices=TASKS)
    parser.add_argument("--run_controls", default=False, action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    # Initialize model.
    model = initialize_lm(
        args.model,
        reduce_precision=args.reduce_precision,
        cache_dir=args.cache_dir
    )

    # Get file-safe model name.
    safe_model_name = get_file_safe_model_name(args.model)

    # Read prompts if specified.
    if args.prompt_file is not None:
        prompts = pd.read_csv(args.prompt_file)
    else:
        prompts = None

    # Create output directory.
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate model on each task.
    if args.task is None:
        tasks = TASKS
    else:
        tasks = args.task
    for task in tasks:
        print(f"***** Task = {task.upper()} *****")

        # Get name of output file where results will be written.
        file = f"{task}_{safe_model_name}.csv"
        outfile = os.path.join(args.output_dir, file)

        # Read stimuli.
        if task.startswith("capitals"):
            stim_file_name = "capitals"
        else:
            stim_file_name = task
        stimuli = pd.read_csv(
            os.path.join(args.stimuli_dir, f"{stim_file_name}.csv")
        )

        # Run the evaluation.
        result = evaluate.evaluate(
            model, 
            stimuli,
            task=task,
            prompts=prompts,
            run_controls=args.run_controls
        )
        # Save results to file.
        result.to_csv(outfile, index=False)
        print(f"Wrote results to {outfile}")


if __name__ == "__main__":
    main()