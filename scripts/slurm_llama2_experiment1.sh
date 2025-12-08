#!/bin/bash
#SBATCH --job-name=llama2_exp1
#SBATCH --output=logs/llama2_exp1_%A_%a.out
#SBATCH --error=logs/llama2_exp1_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --array=0-1

# Run Llama-2 models for Experiment 1 (补充运行)
# Only runs Llama-2-7b-hf and Llama-2-13b-hf

echo "=========================================="
echo "Colors Task - Experiment 1 (Llama-2 补充)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Load environment
module load anaconda3/2024.02-1

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
echo "Activating conda environment 'colors_exp'..."
conda activate colors_exp

# Configuration
REPO_DIR="/home/ypan50/scratchjhu35/ypan50/model-human-processing"
TASK="colors"
EXPERIMENT="experiment_1"

# Set cache to scratch space
export HF_HOME="/scratch/jhu35/ypan50/hf_cache"
export TRANSFORMERS_CACHE="/scratch/jhu35/ypan50/hf_cache"
# HF_TOKEN should be set as environment variable, not hardcoded
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Set it with: export HF_TOKEN=your_token"
fi
mkdir -p $HF_HOME

# Only Llama-2 models
MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
)

# Get model for this array task
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# Enable quantization for these large models
USE_QUANTIZATION="--reduce_precision"

echo "Model: $MODEL"
echo "Task: $TASK"
echo "Experiment: $EXPERIMENT"
echo "Cache: $HF_HOME"
echo "Quantization: enabled"

cd $REPO_DIR

# Run experiment
python src/run_experiment.py \
    --model $MODEL \
    --task $TASK \
    --color_experiment $EXPERIMENT \
    --stimuli_dir data/stimuli \
    --output_dir data/model_output \
    $USE_QUANTIZATION

echo "Finished: $(date)"




