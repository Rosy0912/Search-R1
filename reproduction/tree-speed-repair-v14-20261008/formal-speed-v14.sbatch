#!/usr/bin/env bash
#SBATCH --job-name=tree-fresh200-train
#SBATCH --nodes=1
#SBATCH --nodelist=h100-10
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/node-move-logs/%j.out
#SBATCH --error=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/node-move-logs/%j.err
#SBATCH --time=168:00:00
#SBATCH --no-requeue
set -euo pipefail
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-speed-v14-ebef786
export TREE_INSTRUCT_PACKAGE_SHA256=9774e8854889ace8e618335d5f09c8cfbe229bd265c9b6f454d46ebc8cdfbc88
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/CURRENT200_RETAINED18_SPEED_DIRECT_RELEASE-V14-ebef786.json
export TREE_INSTRUCT_REVIEW_SHA256=24299b17c5dc2936e7b29b4cfa359cbf87f0f6fa6ae328f90d0f48b2266abf2f
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-speed-v14-ebef786/launch.sbatch
