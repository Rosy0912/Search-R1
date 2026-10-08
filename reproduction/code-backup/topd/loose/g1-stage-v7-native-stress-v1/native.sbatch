#!/bin/bash
#SBATCH --job-name=g1-stage-v7-proc-stress
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
sha256sum "$R/tools/g1-stage-v7-native-stress-v1/stress.py" "$R/tools/g1-staged-calibration-producer-v7/stage_cleanup.py" "$R/tools/g1-staged-calibration-producer-v7/owner_support.py"
"$R/env/bin/python" "$R/tools/g1-stage-v7-native-stress-v1/stress.py" --producer "$R/tools/g1-staged-calibration-producer-v7" --output "$R/logs/g1-stage-v7-native-stress-$SLURM_JOB_ID"
