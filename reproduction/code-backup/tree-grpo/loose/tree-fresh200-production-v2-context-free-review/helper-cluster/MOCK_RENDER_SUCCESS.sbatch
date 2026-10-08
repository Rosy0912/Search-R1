#!/usr/bin/env bash
#SBATCH --job-name=tree-fresh200-train
#SBATCH --nodes=1
#SBATCH --nodelist=h100-9
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --no-requeue
set -euo pipefail
export TREE_INSTRUCT_TOOLS=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpzjqyhktg/instruct300/candidates/frozen
export TREE_INSTRUCT_PACKAGE_SHA256=bec5c78444f2b32b2d1da0289a6279d2acd58d747b72c5d4ce5f36a52e495655
export TREE_PRODUCTION_RELEASE_PATH=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpzjqyhktg/release.json
export TREE_INSTRUCT_REVIEW_SHA256=243f6f76a7ebf2789acab822953b69097f3b60de267e6fb6b29d3eecf7b74822
exec bash /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpzjqyhktg/instruct300/candidates/frozen/launch.sbatch
