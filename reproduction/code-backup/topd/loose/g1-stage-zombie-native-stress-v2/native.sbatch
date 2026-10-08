#!/bin/bash
#SBATCH --job-name=g1-stage-zombie-cpu-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
sha256sum "$R/tools/g1-stage-zombie-native-stress-v2/probe.py"
"$R/env/bin/python" "$R/tools/g1-stage-zombie-native-stress-v2/probe.py" --output "$R/logs/g1-stage-zombie-native-stress-$SLURM_JOB_ID"
