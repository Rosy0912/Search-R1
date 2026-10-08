#!/bin/bash
#SBATCH --job-name=topd-startup-guard-cpu-v7
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-startup-guard-cpu-v7-%j.log
set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
export PYTHONPATH="$R/test-tools:$R/repo-terminal-fixed413:$R/tools/qwen25-32b-startup-guard-candidate-v7"
cd "$R/tools/qwen25-32b-startup-guard-candidate-v7"
"$R/env/bin/python" - <<'INNER'
import hashlib,json,pathlib
p=pathlib.Path('.')
for n,h in json.loads((p/'files.json').read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
print('ALL_FROZEN_V7_SOURCE_BYTES_MATCH')
INNER
bash -n eval64-startup-guard.sbatch
"$R/env/bin/python" -m pytest test_startup_guard.py -q
printf 'NATIVE_STARTUP_GUARD_CPU_PASS_NO_GPU_NO_QUALITY\n'
