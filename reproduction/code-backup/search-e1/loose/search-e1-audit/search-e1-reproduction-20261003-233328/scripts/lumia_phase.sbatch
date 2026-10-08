#!/bin/bash
#SBATCH --job-name=search-e1-phase
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/home/jmzhang/Workspace/Search-E1-20261001
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
set -euo pipefail
: "${PHASE:?Submission must specify PHASE}"
: "${RUN:?Submission must specify RUN}"
export PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/upstream/Search-R1:$PWD"
export OMP_NUM_THREADS=4
export GPUS=${SLURM_GPUS_ON_NODE:-4}
export GRPO_MICRO_PER_GPU=2
export GRPO_SCORE_PER_GPU=4
mkdir -p "$RUN"
source scripts/start_retriever.sh
bash scripts/run_phase.sh
