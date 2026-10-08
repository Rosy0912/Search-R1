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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-v10-6efb604
export TREE_INSTRUCT_PACKAGE_SHA256=13268301b6718d77e1159a60526c67d9e139f6d05310438ac3f5b2b10f642cf9
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/CURRENT200_RETAINED18_DIRECT_RELEASE-V10-6efb604.json
export TREE_INSTRUCT_REVIEW_SHA256=8d19e7ae524702b4291e51140ee1d0d20b642b8972bd93ba6a6c8847ed0f1805
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-v10-6efb604/launch.sbatch
