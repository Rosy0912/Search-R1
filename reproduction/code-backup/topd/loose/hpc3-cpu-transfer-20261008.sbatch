#!/bin/bash
#SBATCH --job-name=topd-transfer-cpu-1452
#SBATCH --partition=acd_u
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/user/jxu957/experiments/topd-20261008/migration/cpu-transfer-%j.log
export CUDA_VISIBLE_DEVICES=""
hostname
while [ ! -f /data/user/jxu957/experiments/topd-20261008/migration/complete.json ]; do sleep 30; done
