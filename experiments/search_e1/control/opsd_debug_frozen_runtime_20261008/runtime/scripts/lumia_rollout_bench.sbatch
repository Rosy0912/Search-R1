#!/bin/bash
#SBATCH --job-name=search-e1-rollout-bench
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --nodelist=node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/jmzhang/Workspace/Search-E1-20261001
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
set -euo pipefail
export PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/upstream/Search-R1:$PWD"
export OMP_NUM_THREADS=4
export RUN="$PWD/runs/rollout-bench-$SLURM_JOB_ID"
mkdir -p "$RUN"
# Reuse the already-running retriever on this node; do not start or stop it.
"$PYTHON" - <<'PY'
import requests
r=requests.post('http://127.0.0.1:18000/retrieve',json={'queries':['capital of France'],'topk':3,'return_scores':True},timeout=30)
r.raise_for_status()
assert len(r.json()['result'][0]) == 3
PY
"$PYTHON" -m search_e1.rollout --model "$PWD/models/Qwen2.5-3B-Instruct" --data data/train.jsonl \
 --limit 8 --output "$RUN/rollouts.jsonl" --k 5 --workers 2 --backend vllm --batch-size 4 \
 --retriever-url http://127.0.0.1:18000/retrieve
"$PYTHON" -m search_e1.mine --rollouts "$RUN/rollouts.jsonl" --output "$RUN/pairs.jsonl"
test -f "$RUN/rollouts.jsonl.metrics.json"
touch "$RUN/rollout-bench.done"
