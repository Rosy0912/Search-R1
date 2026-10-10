#!/bin/bash
#SBATCH --job-name=topd-cpu-readiness
#SBATCH --partition=debug
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/jxu957/experiments/topd-20261008/environment/slurm-readiness-%j.log
set -eo pipefail
umask 077
B=/data/user/jxu957/experiments/topd-20261008
exec 9>"$B/environment/readiness-validation.lock"
flock -n 9 || exit 75
source "$B/environment/activate-topd.sh"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128
export PYTHONPATH="$B/runtime/tools/topd-instruct300-source-v1:$B/runtime/tools/topd-instruct300-launch-v6/runtime:$B/runtime/tools/topd-vllm-teacher-v3:$B/runtime/retriever-gpu-lib"
python -m pip check
python -u "$B/environment/preparation-tools/validate_hpc3_runtime.py"
