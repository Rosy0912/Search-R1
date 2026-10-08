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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-4a65e9a-v5
export TREE_INSTRUCT_PACKAGE_SHA256=cda27d64b7114417f64cf80f23772b288cdbc7ec9ef2ce9e7c86338e08dbf1c9
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/DIRECT_TRAIN200_RELEASE-V5.json
export TREE_INSTRUCT_REVIEW_SHA256=7f1adc8d9d4024b3eb4cbac02646e26208debfb0e634857c5119ef59641f6fba
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-4a65e9a-v5/launch.sbatch
