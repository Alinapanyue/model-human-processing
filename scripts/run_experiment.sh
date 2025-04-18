#!/bin/bash

MODEL=$1 # Huggingface ID, e.g. "meta-llama/Llama-2-7b-hf" 
TASK=$2 # e.g., "capitals-recall"

echo "$MODEL / $TASK"

python src/run_experiment.py \
    --model $MODEL \
    --task $TASK \
    --stimuli_dir data/stimuli \
    --output_dir data/model_output \
    --run_controls
    # optionally use the flag below to reduce model precision
    # --reduce_precision