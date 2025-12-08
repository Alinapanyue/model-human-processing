#!/bin/bash
#SBATCH --job-name=colors_exp1
#SBATCH --output=logs/exp1_%A_%a.out
#SBATCH --error=logs/exp1_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --array=0-7

# Experiment 1: Systematic Manipulation of Fact Number and Type
# 
# This experiment tests the dose-response relationship between number of 
# intervening facts (1-5) and dual processing metrics, comparing normal vs strange
# facts at each level.
#
# Conditions per entity: 11 total
# - Baseline: 0 facts
# - Normal-1 through Normal-5: 1-5 normal facts
# - Strange-1 through Strange-5: 1-5 strange facts
#
# Usage: sbatch scripts/slurm_experiment1.sh
# This will run all 8 models as a job array

echo "=========================================="
echo "Colors Task - Experiment 1"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Load environment
module load anaconda3/2024.02-1

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Create environment if it doesn't exist
if ! conda env list | grep -q "^colors_exp "; then
    echo "Creating conda environment 'colors_exp'..."
    conda create -n colors_exp python=3.10 -y
    conda activate colors_exp
    echo "Installing packages..."

    # Install PyTorch with CUDA support
    echo "Installing PyTorch and dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pandas numpy tuned-lens
    pip install transformers
    pip install bitsandbytes
    pip install nnsight

    # Verify all imports work
    echo ""
    echo "Verifying package installation..."
    python -c "import torch; print(f'✓ torch: {torch.__version__}')"
    python -c "import transformers; print(f'✓ transformers: {transformers.__version__}')"
    python -c "import nnsight; print('✓ nnsight OK')"
    python -c "import tuned_lens; print('✓ tuned_lens OK')"
    python -c "import pandas, numpy; print('✓ pandas and numpy OK')"
    echo ""
    echo "✓ All packages installed successfully"
else
    echo "Activating existing conda environment 'colors_exp'..."
    conda activate colors_exp
    pip install bitsandbytes --quiet 2>/dev/null || true
fi

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

# Model array for Experiment 1
# Test all models from both GPT-2 and Llama families
MODELS=(
    "gpt2"
    "gpt2-medium"
    "gpt2-large"
    "gpt2-xl"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
)

# Get model for this array task
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# Determine if we need quantization for large models
if [[ "$MODEL" == *"xl"* ]] || [[ "$MODEL" == *"large"* ]] || [[ "$MODEL" == *"7b"* ]] || [[ "$MODEL" == *"7B"* ]] || [[ "$MODEL" == *"8b"* ]] || [[ "$MODEL" == *"8B"* ]] || [[ "$MODEL" == *"13b"* ]] || [[ "$MODEL" == *"13B"* ]]; then
    USE_QUANTIZATION="--reduce_precision"
    echo "Large model detected - enabling 4-bit quantization"
else
    USE_QUANTIZATION=""
    echo "Small model - no quantization needed"
fi

echo "Model: $MODEL"
echo "Task: $TASK"
echo "Experiment: $EXPERIMENT"
echo "Cache: $HF_HOME"
echo "Quantization: ${USE_QUANTIZATION:-none}"

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

