#!/bin/bash
#SBATCH --job-name=qwen25-32b-eval64-spawn-v1
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --time=03:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/qwen25-32b-terminal-fixed413-eval64-spawn-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export VLLM_WORKER_MULTIPROC_METHOD=spawn
test "$#" -eq 1
printf '%s\n' '21aa558e8f6ce0ad3e719b56c91e8c32644add9dc28326400e1d3655e00bf0ba  /dataset/shaoshitong/search-r1-rl/topd/tools/qwen25-32b-terminal-fix-refreeze/eval64.sbatch' | sha256sum -c -
exec bash "$AGENT_ROOT/tools/qwen25-32b-terminal-fix-refreeze/eval64.sbatch" "$1"
