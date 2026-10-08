#!/bin/bash
#SBATCH --job-name=topd-v7-burst-cpu
#SBATCH --partition=cpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-v7-burst-cpu-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo-terminal-fixed413"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$AGENT_ROOT/repo-terminal-fixed413"
sha256sum "$AGENT_ROOT/tools/qwen25-32b-startup-guard-candidate-v7/"{startup_guard.py,launch.py,test_startup_guard.py}
sha256sum "$AGENT_ROOT/tools/qwen25-32b-v6-burst-probe/"{probe.py,worker.py,test_probe_contract.py}
"$AGENT_ROOT/env/bin/python" -m pytest -q \
 "$AGENT_ROOT/tools/qwen25-32b-startup-guard-candidate-v7/test_startup_guard.py" \
 "$AGENT_ROOT/tools/qwen25-32b-v6-burst-probe/test_probe_contract.py"
