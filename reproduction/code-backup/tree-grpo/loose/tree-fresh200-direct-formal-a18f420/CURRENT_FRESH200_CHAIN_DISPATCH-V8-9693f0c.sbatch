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
export TREE_INSTRUCT_TOOLS=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current-fresh200-chain-mask-v8-9693f0c
export TREE_INSTRUCT_PACKAGE_SHA256=336a0a60df561fd3fcf50e7fdc03cee344a0b33e9d916193c7b4a4716b1b6a12
export TREE_PRODUCTION_RELEASE_PATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/CURRENT_FRESH200_CHAIN_RELEASE-V8-9693f0c.json
export TREE_INSTRUCT_REVIEW_SHA256=7e6c586d3555156435115b5893df3935613bf8603cc20172540a99a2cee9c677
exec bash /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current-fresh200-chain-mask-v8-9693f0c/launch.sbatch
