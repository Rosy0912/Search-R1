#!/bin/bash
#SBATCH --job-name=g1-32b-owned100
#SBATCH --partition=gpu
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/g1-32b-owned-%j.log
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=7-00:00:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
: "${G1_PIPELINE_CONFIG:?actual source/evidence bound config}"
: "${G1_PIPELINE_DIR:?frozen isolated supervisor}"
: "${G1_PIPELINE_MODE:?initial-only, profile2 or train100}"
: "${G1_PREFLIGHT_ENTRY:?frozen common exclusive 8 GPU lease preflight}"
# Preflight must supply the actual eight open FD/UUID leases in the config before exec.
# No algorithm/scorer proof inherited from old 3B TOPD.
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$TASK_ROOT/env/bin/python" "$G1_PREFLIGHT_ENTRY" -- "$TASK_ROOT/env/bin/python" "$G1_PIPELINE_DIR/supervise.py" --config "$G1_PIPELINE_CONFIG" --mode "$G1_PIPELINE_MODE"
