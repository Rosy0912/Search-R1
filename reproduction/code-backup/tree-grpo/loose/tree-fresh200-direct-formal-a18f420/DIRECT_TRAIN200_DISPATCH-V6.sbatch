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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-4a65e9a-v6
export TREE_INSTRUCT_PACKAGE_SHA256=ef2817200f999c74deca43ea3de501f6bf18276ea838b38b35342ebf3368265f
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/DIRECT_TRAIN200_RELEASE-V6.json
export TREE_INSTRUCT_REVIEW_SHA256=ae9207e612bc671d19b2cf8a061e9b73117d46b4f33ca47c5c06721d647200f2
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-direct-train-4a65e9a-v6/launch.sbatch
