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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current-fresh200-chain-mask-v7-9693f0c
export TREE_INSTRUCT_PACKAGE_SHA256=5f7071c0c6886b3a5b88abd216cd3b18bc0b14a1fd6293418717d742ece4ccff
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/CURRENT_FRESH200_CHAIN_RELEASE-V7-9693f0c.json
export TREE_INSTRUCT_REVIEW_SHA256=3933f3edfff9ded1b31f45c9fddea12ea412da95279c3b03c34df85b3fba28bd
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current-fresh200-chain-mask-v7-9693f0c/launch.sbatch
