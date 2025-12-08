#!/bin/bash

# Quick reference for available colors experiments

cat << 'EOF'

═══════════════════════════════════════════════════════════════════════
                    COLORS TASK EXPERIMENTS
═══════════════════════════════════════════════════════════════════════

EXPERIMENT 1: Systematic Manipulation of Fact Number and Type
----------------------------------------------------------------
Design:       11 conditions per entity (baseline + 1-5 facts × 2 styles)
Models:       8 models (4 GPT-2 + 4 Llama)
Hypotheses:   - CoM/TTD increase linearly with number of facts
              - Normal facts > strange facts at all levels
              - Difference increases with more facts

Run all:      sbatch scripts/slurm_experiment1.sh
Run single:   bash scripts/run_colors_experiment.sh gpt2 experiment_1


EXPERIMENT 2: Model Family Comparison at Matched Scales
----------------------------------------------------------------
Design:       Same as Experiment 1 (11 conditions)
Focus:        Compare GPT-2 vs Llama at matched parameter scales
Models:       4 matched pairs:
              - gpt2 (124M) ↔ Llama-3.2-1B (1B)
              - gpt2-medium (355M) ↔ Llama-3.2-3B (3B)
              - gpt2-large (774M) ↔ Llama-2-7b-hf (7B) [anomaly test]
              - gpt2-xl (1.5B) ↔ Llama-2-13b-hf (13B)

Run all:      sbatch scripts/slurm_experiment2.sh
Run pair:     bash scripts/run_colors_experiment.sh gpt2-large experiment_1
              bash scripts/run_colors_experiment.sh meta-llama/Llama-2-7b-hf experiment_1


EXPERIMENT 3: Fact Type Breakdown
----------------------------------------------------------------
Design:       10 conditions per entity (5 fact types × 2 styles)
Fact types:   appearance, type, subtype, place, size
Goal:         Rank fact types by interference strength
              Test if ranking differs for normal vs strange

Run all:      sbatch scripts/slurm_experiment3.sh
Run single:   bash scripts/run_colors_experiment.sh gpt2 experiment_3


═══════════════════════════════════════════════════════════════════════
                          QUICK COMMANDS
═══════════════════════════════════════════════════════════════════════

Monitor jobs:
  squeue -u $USER

View logs:
  tail -f logs/exp1_<job_id>_<array_id>.out

Cancel jobs:
  scancel <job_id>

List output files:
  ls -lh data/model_output/logit_lens/colors_experiment_*


═══════════════════════════════════════════════════════════════════════
                      AVAILABLE MODELS
═══════════════════════════════════════════════════════════════════════

GPT-2 Family:
  - gpt2                  (124M)
  - gpt2-medium           (355M)
  - gpt2-large            (774M)
  - gpt2-xl               (1.5B)

Llama Family:
  - meta-llama/Llama-3.2-1B       (1B)
  - meta-llama/Llama-3.2-3B       (3B)
  - meta-llama/Llama-2-7b-hf      (7B)
  - meta-llama/Llama-2-13b-hf     (13B)


═══════════════════════════════════════════════════════════════════════

For detailed documentation, see: EXPERIMENTS_GUIDE.md

EOF




