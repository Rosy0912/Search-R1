#!/bin/bash
#SBATCH --job-name=topd-independent-quality64-raw-review
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/independent-quality64-native-%j.log
set -euo pipefail
if [ "$#" -ne 3 ]; then exit 64; fi
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$AGENT_ROOT/repo-terminal-fixed413"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [ "${SLURM_GPUS_ON_NODE:-0}" != 0 ]; then exit 78; fi
if [ "${SLURM_JOB_NODELIST:-}" != h100-11 ]; then exit 78; fi
READER="$AGENT_ROOT/tools/independent-quality64-review/review_eval64_raw.py"
printf '%s  %s\n' f0573d94df5d4683641057e158e3aa6e95c4073930511e9210720b40d95b37ea "$READER" | sha256sum -c -
test "$(git -C "$AGENT_ROOT/repo-terminal-fixed413" rev-parse HEAD)" = 413a10c50f32febaafc83db8170e2b005e0e979f
"$AGENT_ROOT/env/bin/python" "$READER" --evaluation "$1" --binding "$2" --output "$3"
