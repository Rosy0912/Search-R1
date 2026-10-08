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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-a18f420-v3
export TREE_INSTRUCT_PACKAGE_SHA256=3f21730c9665dc25072d71c8e79b7e91becfcf5e96ad0aa136df26545cb2fad4
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/DIRECT_TRAIN200_RELEASE-a18f420-V3.json
export TREE_INSTRUCT_REVIEW_SHA256=502dd15346cde2493873990c49827584e367d4e1b8e2bf9036dc2c03cd06bcee
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-a18f420-v3/launch.sbatch
