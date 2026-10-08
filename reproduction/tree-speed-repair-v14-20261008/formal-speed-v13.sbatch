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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-speed-v13-6227108
export TREE_INSTRUCT_PACKAGE_SHA256=cbca5e1de3188eaf071ce02c295d2607475ec003c53352110a4c6fbf72cbfa63
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/CURRENT200_RETAINED18_SPEED_DIRECT_RELEASE-V13-6227108.json
export TREE_INSTRUCT_REVIEW_SHA256=17a85334b491a73541d733ae676065b91dda456851d4558aa92a879e181d84a0
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current200-retained18-speed-v13-6227108/launch.sbatch
