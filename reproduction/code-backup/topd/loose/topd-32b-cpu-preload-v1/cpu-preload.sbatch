#!/bin/bash
#SBATCH --job-name=topd32b-official-cpu-fp32-identity
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd32b-cpu-preload-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export PYTHONPATH="$AGENT_ROOT/repo-terminal-fixed413"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$AGENT_ROOT/tools/topd-32b-native-score-gate"
test "$#" -eq 1
sha256sum -c files.sha256
exec "$AGENT_ROOT/env/bin/python" preload.py --binding "$1" --candidate "$AGENT_ROOT/tools/topd-32b-pipeline-candidate" --output "$AGENT_ROOT/logs/topd32b-cpu-preload-$SLURM_JOB_ID.json"
