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
export TREE_INSTRUCT_TOOLS=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpfviuwfaq/instruct300/candidates/frozen
export TREE_INSTRUCT_PACKAGE_SHA256=007f41594b775e4f1ca539fd951c1314ebe7e24a65bef196939315b59197e57d
export TREE_PRODUCTION_RELEASE_PATH=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpfviuwfaq/release.json
export TREE_INSTRUCT_REVIEW_SHA256=f52a452def3e3da2a837adefb37d12b2df3a6e9bd152ed9bb8dd5af123374aeb
exec bash /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v2-context-free-review/helper-cluster/tmpfviuwfaq/instruct300/candidates/frozen/launch.sbatch
