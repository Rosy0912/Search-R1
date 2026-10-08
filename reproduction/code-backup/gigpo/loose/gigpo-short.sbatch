#!/bin/bash
#SBATCH --job-name=gigpo-reviewed-short
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/data/shaoshitong/search-r1-rl/gigpo/logs/gigpo-short-%j.log
set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo
export PATH="$ROOT/env/bin:$PATH" PYTHONPATH="$ROOT/repo:$ROOT/tools"
export C_INCLUDE_PATH="$ROOT/python-dev/usr/include/python3.10:$ROOT/python-dev/usr/include:$ROOT/python-dev/usr/include/x86_64-linux-gnu/python3.10"
cd "$ROOT/repo"
srun --ntasks=1 "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/gigpo-short-run.py" "$@"
