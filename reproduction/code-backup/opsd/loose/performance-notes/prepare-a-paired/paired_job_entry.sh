#!/usr/bin/env bash
#SBATCH --job-name=opsd-proposal-a-paired
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=h100-10
#SBATCH --cpus-per-task=192
#SBATCH --mem=1T
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --time=06:00:00
set -euo pipefail
root=/mnt/data/shaoshitong/search-r1-rl/opsd
: "${SLURM_JOB_ID:?Owned Slurm allocation required}"
[[ "${SLURM_JOB_NODELIST:-}" == h100-10 && "$(hostname -s)" == h100-10 ]]
source "$root/env/bin/activate"
cd "$root/repo"
: "${OPSD_EXPECTED_COMMIT:?Pinned source required}"
[[ "$OPSD_EXPECTED_COMMIT" == e43fbfa73088ed70ed3efef79ba561507c0afc33 && "$(git rev-parse HEAD)" == "$OPSD_EXPECTED_COMMIT" ]]
# Lease acquisition and the first double15% guard precede any GPU service.
exec python -m search_r1.baselines.preflight -- bash "$root/tools/proposal-a-paired/paired_job.sh"
