#!/bin/bash
#SBATCH --job-name=g1-v11-known-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/g1-v11-known-cpu-%j.log
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
: "${DEPLOY_ENTRY_MANIFEST_SHA256:?frozen entry manifest}"
OUT="$TASK_ROOT/logs/g1-v11-known-cpu-$SLURM_JOB_ID"
mkdir -m 700 "$OUT"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
"$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/g1-native-cpu-deploy-v3/native_entry.py" --manifest-sha256 "$DEPLOY_ENTRY_MANIFEST_SHA256" --output "$OUT"
