#!/bin/bash
#SBATCH --job-name=topd-base-cpu-score-cost
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/teacher-cpu-cost-%j.log
set -euo pipefail
# CPU allocation only: no --gres and no GPU model/algorithm run.
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONPATH
cd "$AGENT_ROOT/repo"
git rev-parse HEAD
"$AGENT_ROOT/env/bin/python" reproduction/h20/check_environment.py
# First submission defaults to one prompt256 + response500 trajectory.
# After reviewing its cost/RSS, pass --prompt-lengths 256,1024,4096 --repetitions 2.
exec "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/topd-teacher-cpu-cost-probe.py" \
  --output "$AGENT_ROOT/logs/teacher-cpu-cost-$SLURM_JOB_ID.jsonl" "$@"
