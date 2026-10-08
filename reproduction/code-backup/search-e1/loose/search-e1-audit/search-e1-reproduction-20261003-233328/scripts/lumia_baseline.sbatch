#!/bin/bash
#SBATCH --job-name=search-e1-baseline
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/jmzhang/Workspace/Search-E1-20261001
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
set -euo pipefail
export PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/upstream/Search-R1:$PWD"
export OMP_NUM_THREADS=4
export GPUS=4
export RUN="$PWD/runs/validation-baseline-$SLURM_JOB_ID"
mkdir -p "$RUN"
source scripts/start_retriever.sh
MODEL="$PWD/models/Qwen2.5-3B-Instruct" OUTPUT="$RUN/checkpoints" \
GRPO_MICRO_PER_GPU=2 GRPO_SCORE_PER_GPU=4 bash scripts/grpo.sh \
trainer.val_before_train=true trainer.val_only=true 2>&1 | tee "$RUN/validation.log"
grep -q 'val/test_score/' "$RUN/validation.log"
touch "$RUN/validation.done"
