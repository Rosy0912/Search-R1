#!/bin/bash
#SBATCH --job-name=topd-teacher-gate-fixtures
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/teacher-gate-fixtures-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
cd "$AGENT_ROOT/tools"
sha256sum verify-topd-teacher.py test_verify_topd_teacher.py
"$AGENT_ROOT/env/bin/python" -m pytest test_verify_topd_teacher.py -q
printf 'NATIVE_TEACHER_GATE_FIXTURES_PASS_NO_REAL_TEACHER\n'
