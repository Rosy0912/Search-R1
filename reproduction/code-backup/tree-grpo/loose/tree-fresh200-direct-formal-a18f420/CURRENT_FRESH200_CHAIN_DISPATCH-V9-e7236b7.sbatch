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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current-fresh200-chain-mask-v9-e7236b7
export TREE_INSTRUCT_PACKAGE_SHA256=995819deb2d8ba85f4da92a4ab6ed7abc7eb4887e566cda730297f1e86305760
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/CURRENT_FRESH200_CHAIN_RELEASE-V9-e7236b7.json
export TREE_INSTRUCT_REVIEW_SHA256=7e57ba49fd2b29daa7aaf3df2426db15938418b230e3964efa0dfb1756e79ee3
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current-fresh200-chain-mask-v9-e7236b7/launch.sbatch
