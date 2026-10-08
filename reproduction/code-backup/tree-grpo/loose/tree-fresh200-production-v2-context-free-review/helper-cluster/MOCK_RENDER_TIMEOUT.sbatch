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
export TREE_INSTRUCT_TOOLS=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpoekdxlp4/instruct300/candidates/frozen
export TREE_INSTRUCT_PACKAGE_SHA256=c08afbb354caaec6c19a93eec27c27213fc3b6fda6e0aebe6d53fde843e6ff5f
export TREE_PRODUCTION_RELEASE_PATH=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpoekdxlp4/release.json
export TREE_INSTRUCT_REVIEW_SHA256=e72f0449ce9bb30b4e712e344b39f14caac6be4d2bacd2d71237162413f30e70
exec bash /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpoekdxlp4/instruct300/candidates/frozen/launch.sbatch
