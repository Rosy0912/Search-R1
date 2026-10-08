#!/usr/bin/env bash
#SBATCH --job-name=tree-fresh200-train
#SBATCH --nodes=1
#SBATCH --nodelist=h100-9
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --time=999:00:00
#SBATCH --no-requeue
set -euo pipefail
export TREE_INSTRUCT_TOOLS=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmptwv5byg_/instruct300/candidates/frozen
export TREE_INSTRUCT_PACKAGE_SHA256=537242245a50e286c3a8612644c91f22c84e884878da1488ad43157f15ed107d
export TREE_PRODUCTION_RELEASE_PATH=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmptwv5byg_/release.json
export TREE_INSTRUCT_REVIEW_SHA256=c1bce6340f57d6a270b7b789f295ec1e0a849fffbdea06b896cd3084265a72ec
exec bash /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmptwv5byg_/instruct300/candidates/frozen/launch.sbatch
