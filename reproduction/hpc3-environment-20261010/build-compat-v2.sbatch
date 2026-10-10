#!/bin/bash
#SBATCH --job-name=topd-fa-compat
#SBATCH --partition=debug
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/jxu957/experiments/topd-20261008/environment/slurm-compat-%j.log
bash /data/user/jxu957/experiments/topd-20261008/environment/build-compat-v2.sh
