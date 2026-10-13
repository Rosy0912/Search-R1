#!/usr/bin/env bash
#SBATCH --job-name=tree-final-eval-resume
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-10
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${TREE_EVAL_TOOLS:?}"
: "${TREE_EVAL_PREFIX:?}"
: "${TREE_EVAL_MANIFEST:?}"
: "${TREE_EVAL_RELEASE:?}"
: "${TREE_EVAL_SCORE_OUTPUT:?}"
export PYTHONOPTIMIZE=0
export PYTHONPATH=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/branpo-fresh200-8b6f99d1b3353145200eb3fa7c16a579a79a987c/source/repo:/dataset/shaoshitong/search-r1-rl/tree-grpo/tools
unset SLURM_CLUSTERS SLURM_CONF SLURM_CONF_SERVER SLURM_FEDERATION SLURM_FEDERATION_CLUSTER
exec /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -m search_r1.baselines.preflight -- /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python "$TREE_EVAL_TOOLS/supervisor.py" --prefix "$TREE_EVAL_PREFIX" --manifest "$TREE_EVAL_MANIFEST" --release "$TREE_EVAL_RELEASE" --score-output "$TREE_EVAL_SCORE_OUTPUT"
