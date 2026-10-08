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
export TREE_INSTRUCT_PACKAGE_SHA256=02766458bddc2fbe8abb02b0a37478730d418ffb8958959828ff53cc60b22389
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/DIRECT_TRAIN200_RELEASE-a18f420.json
export TREE_INSTRUCT_REVIEW_SHA256=72412e079d9526c4bbe4c956657e05ba5a0be69b4a3f6cf57180ade62e35a00c
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-a18f420/launch.sbatch
