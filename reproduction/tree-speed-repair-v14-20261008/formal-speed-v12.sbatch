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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-speed-v12-48a59da
export TREE_INSTRUCT_PACKAGE_SHA256=6b061fd4d0aa7cc6ad6f48c13a671014e5dbfee6f630a1ca80d94785cccdbe5e
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/CURRENT200_RETAINED18_SPEED_DIRECT_RELEASE-V12-48a59da.json
export TREE_INSTRUCT_REVIEW_SHA256=cdd3fa016c6c024c68fdcc88a57af402621127341af7e3a0b66afa2e77e9ed13
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-speed-v12-48a59da/launch.sbatch
