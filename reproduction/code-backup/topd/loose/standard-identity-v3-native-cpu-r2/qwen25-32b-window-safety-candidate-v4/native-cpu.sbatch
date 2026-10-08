#!/bin/bash
#SBATCH --job-name=topd-window-safety-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-window-safety-cpu-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo-terminal-fixed413"
cd "$AGENT_ROOT/repo-terminal-fixed413"
TASK_DIR="$AGENT_ROOT/tools/qwen25-32b-window-safety-candidate-v4"
sha256sum "$TASK_DIR/"{runtime_safety.py,startup_guard.py,launch.py,test_runtime_safety.py,test_startup_guard.py,source-binding.json,actual3056-window.json}
"$AGENT_ROOT/env/bin/python" -m pytest -q --basetemp="$AGENT_ROOT/logs/window-safety-cpu-tmp-$SLURM_JOB_ID" -o cache_dir="$AGENT_ROOT/logs/window-safety-cpu-cache-$SLURM_JOB_ID" "$TASK_DIR/test_runtime_safety.py" "$TASK_DIR/test_startup_guard.py"
