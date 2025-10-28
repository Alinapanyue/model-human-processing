#!/bin/bash
#SBATCH --job-name=colors_exp
#SBATCH --output=logs/colors_%j.out
#SBATCH --error=logs/colors_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s

# Usage: sbatch scripts/slurm_colors.sh <MODEL_NAME>

echo "=========================================="
echo "Colors Task Experiment"
echo "Job ID: $SLURM_JOB_ID"
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
    # Ensure bitsandbytes is installed (needed for quantization)
    pip install bitsandbytes --quiet 2>/dev/null || true
fi

# Configuration
MODEL="${1:-gpt2}"
REPO_DIR="/home/ypan50/scratchjhu35/ypan50/model-human-processing"
TASK="colors"

# Set cache to scratch space (also check shared cache)
export HF_HOME="/scratch/jhu35/ypan50/hf_cache"
export TRANSFORMERS_CACHE="/scratch/jhu35/ypan50/hf_cache"
# HF_TOKEN should be set as environment variable before running this script
mkdir -p $HF_HOME

# Determine if we need quantization for large models
# Use quantization for: gpt2-xl, Llama models, or any model with 'xl', 'large', '7b', '8b', '13b' in name
if [[ "$MODEL" == *"xl"* ]] || [[ "$MODEL" == *"large"* ]] || [[ "$MODEL" == *"7b"* ]] || [[ "$MODEL" == *"7B"* ]] || [[ "$MODEL" == *"8b"* ]] || [[ "$MODEL" == *"8B"* ]] || [[ "$MODEL" == *"13b"* ]] || [[ "$MODEL" == *"13B"* ]]; then
    USE_QUANTIZATION="--reduce_precision"
    echo "Large model detected - enabling 4-bit quantization"
else
    USE_QUANTIZATION=""
    echo "Small model - no quantization needed"
fi

echo "Model: $MODEL"
echo "Task: $TASK"
echo "Cache: $HF_HOME"
echo "Quantization: ${USE_QUANTIZATION:-none}"

cd $REPO_DIR

# Run experiment
python src/run_experiment.py \
    --model $MODEL \
    --task $TASK \
    --stimuli_dir data/stimuli \
    --output_dir data/model_output \
    $USE_QUANTIZATION

echo "Finished: $(date)"
