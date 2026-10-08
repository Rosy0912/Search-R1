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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-v11-6efb604
export TREE_INSTRUCT_PACKAGE_SHA256=1a2a7ea8dea6a599d7d170671fb0ff6f2a6076b2dc5c8b61653e0381642ebb6b
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/CURRENT200_RETAINED18_DIRECT_RELEASE-V11-6efb604.json
export TREE_INSTRUCT_REVIEW_SHA256=b7ea8bef0bd89a0c314b70c21e35d776a909f0a13a8e09ffc7edb713522657f9
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-v11-6efb604/launch.sbatch
