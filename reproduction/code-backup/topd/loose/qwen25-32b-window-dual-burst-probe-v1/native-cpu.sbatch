#!/bin/bash
#SBATCH --job-name=topd-window-dual-burst-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-window-dual-burst-cpu-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo-terminal-fixed413"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$AGENT_ROOT/repo-terminal-fixed413"
sha256sum "$AGENT_ROOT/tools/qwen25-32b-terminal-fix-refreeze/eval64.py"
sha256sum "$AGENT_ROOT/tools/qwen25-32b-window-safety-candidate-v1/"{runtime_safety.py,startup_guard.py,launch.py,test_runtime_safety.py,test_startup_guard.py,source-binding.json,files.json}
sha256sum "$AGENT_ROOT/tools/qwen25-32b-window-dual-burst-probe-v1/"{probe.py,worker.py,test_probe_contract.py}
"$AGENT_ROOT/env/bin/python" -m pytest -q --basetemp="$AGENT_ROOT/logs/window-burst-cpu-tmp-$SLURM_JOB_ID" -o cache_dir="$AGENT_ROOT/logs/window-burst-cpu-cache-$SLURM_JOB_ID" \
 "$AGENT_ROOT/tools/qwen25-32b-window-safety-candidate-v1/test_runtime_safety.py" \
 "$AGENT_ROOT/tools/qwen25-32b-window-safety-candidate-v1/test_startup_guard.py" \
 "$AGENT_ROOT/tools/qwen25-32b-window-dual-burst-probe-v1/test_probe_contract.py"
