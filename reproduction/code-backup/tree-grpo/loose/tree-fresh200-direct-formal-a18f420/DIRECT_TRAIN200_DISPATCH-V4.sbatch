#!/usr/bin/env bash
#SBATCH --job-name=tree-fresh200-train
#SBATCH --nodes=1
#SBATCH --nodelist=h100-9
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --time=168:00:00
#SBATCH --no-requeue
set -euo pipefail
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-2e26b55-v4
export TREE_INSTRUCT_PACKAGE_SHA256=b9e7de77b1d1e168dd02af2f3be474af7d0dd2a510d13e4ee812fe6d6c4b9108
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/DIRECT_TRAIN200_RELEASE-V4.json
export TREE_INSTRUCT_REVIEW_SHA256=09b2109843affb89d7648a09ccf7a1bdeb06370169393f771a05d8ec5df95966
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-2e26b55-v4/launch.sbatch
