#!/bin/bash
#SBATCH --job-name=topd-fa3-build
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jxu957/experiments/topd-20261008/environment/slurm-fa3-%j.log
set -e
r=/data/user/jxu957/experiments/topd-20261008/environment
for i in $(seq 1 120); do
 if flock -n "$r/build-fa3.lock" -c true; then
  export MAX_JOBS=8 NVCC_THREADS=1
  bash "$r/build-fa3.sh"
  exit $?
 fi
 sleep 5
done
exit 75
