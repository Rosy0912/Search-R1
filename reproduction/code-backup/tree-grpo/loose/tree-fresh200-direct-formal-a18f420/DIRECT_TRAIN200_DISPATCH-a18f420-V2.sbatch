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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-a18f420
export TREE_INSTRUCT_PACKAGE_SHA256=33fd8b5351a0a28e0a7a1e5742ce5c49db604d950985f35f234ba36e2a7de504
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/DIRECT_TRAIN200_RELEASE-a18f420-V2.json
export TREE_INSTRUCT_REVIEW_SHA256=f42d2b58e1556f35c06ce22175ac7350652fcfe1c87241391fde5857cabc5448
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-a18f420/launch.sbatch
