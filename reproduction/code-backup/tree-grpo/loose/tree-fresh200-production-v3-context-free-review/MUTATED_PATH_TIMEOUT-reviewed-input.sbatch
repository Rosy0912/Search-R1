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
export TREE_INSTRUCT_TOOLS=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpevrk9phm/instruct300/candidates/frozen
export TREE_INSTRUCT_PACKAGE_SHA256=1b4a16cdb814f96f06627f03099eadd77b177370bd3bffc493e6229d76ebb02b
export TREE_PRODUCTION_RELEASE_PATH=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpevrk9phm/release.json
export TREE_INSTRUCT_REVIEW_SHA256=0e6736ec97b419b359772ba808b10465202e4fdaaf143b8ff50d8749c1f924ec
exec bash /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpevrk9phm/instruct300/candidates/frozen/launch.sbatch
