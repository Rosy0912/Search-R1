#!/bin/bash
#SBATCH --job-name=search-e1-3b-full
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
: "${PILOT_DIR:?Must provide a validated pilot directory}"
test -f "$PILOT_DIR/pilot.done"
export PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/upstream/Search-R1:$PWD"
export RUN="$PWD/runs/paper-3b-$SLURM_JOB_ID"
export OMP_NUM_THREADS=4
export GPUS=4
mkdir -p "$RUN"
source scripts/start_retriever.sh
bash scripts/cycle.sh
