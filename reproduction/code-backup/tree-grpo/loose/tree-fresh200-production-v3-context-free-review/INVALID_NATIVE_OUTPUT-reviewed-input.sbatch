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
export TREE_INSTRUCT_TOOLS=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpdypch6rn/instruct300/candidates/frozen
export TREE_INSTRUCT_PACKAGE_SHA256=5aac67bff50aa59786d34383cee1e9adcafb97eeaff89fd335e7e8720e34ecaa
export TREE_PRODUCTION_RELEASE_PATH=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpdypch6rn/release.json
export TREE_INSTRUCT_REVIEW_SHA256=7a28db8ad6d738917a9009ef7b1a25d5390f8d1998a8b2a0ac7b6758a2635e7a
exec bash /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpdypch6rn/instruct300/candidates/frozen/launch.sbatch
