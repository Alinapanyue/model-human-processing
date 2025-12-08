#!/bin/bash

# Helper script for running colors experiments with specific model and experiment type
#
# Usage: bash scripts/run_colors_experiment.sh <MODEL> <EXPERIMENT>
#
# Examples:
#   bash scripts/run_colors_experiment.sh gpt2 experiment_1
#   bash scripts/run_colors_experiment.sh meta-llama/Llama-2-7b-hf experiment_2
#   bash scripts/run_colors_experiment.sh gpt2-xl experiment_3

MODEL=$1
EXPERIMENT=$2

if [ -z "$MODEL" ] || [ -z "$EXPERIMENT" ]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage: bash scripts/run_colors_experiment.sh <MODEL> <EXPERIMENT>"
    echo ""
    echo "EXPERIMENT options:"
    echo "  experiment_1    - Systematic manipulation of fact number (1-5) and type"
    echo "  experiment_3    - Individual fact type effects (single-fact conditions)"
    echo "  original        - Original mixed design (default)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run_colors_experiment.sh gpt2 experiment_1"
    echo "  bash scripts/run_colors_experiment.sh meta-llama/Llama-2-7b-hf experiment_1"
    exit 1
fi

echo "Running Colors Experiment"
echo "Model: $MODEL"
echo "Experiment: $EXPERIMENT"
echo ""

# Determine if we need reduced precision
if [[ "$MODEL" == *"xl"* ]] || [[ "$MODEL" == *"large"* ]] || [[ "$MODEL" == *"7b"* ]] || [[ "$MODEL" == *"7B"* ]] || [[ "$MODEL" == *"8b"* ]] || [[ "$MODEL" == *"8B"* ]] || [[ "$MODEL" == *"13b"* ]] || [[ "$MODEL" == *"13B"* ]] || [[ "$MODEL" == *"1B"* ]] || [[ "$MODEL" == *"3B"* ]]; then
    USE_QUANTIZATION="--reduce_precision"
    echo "Note: Large model detected, using 4-bit quantization"
else
    USE_QUANTIZATION=""
fi

python src/run_experiment.py \
    --model $MODEL \
    --task colors \
    --color_experiment $EXPERIMENT \
    --stimuli_dir data/stimuli \
    --output_dir data/model_output \
    $USE_QUANTIZATION

echo ""
echo "Done!"




