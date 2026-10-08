#!/bin/bash
#SBATCH --job-name=search-e1-smoke
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/jmzhang/Workspace/Search-E1-20261001
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
set -euo pipefail
export PYTHONPATH="$PWD/upstream/Search-R1:$PWD"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/gpu_smoke.py
