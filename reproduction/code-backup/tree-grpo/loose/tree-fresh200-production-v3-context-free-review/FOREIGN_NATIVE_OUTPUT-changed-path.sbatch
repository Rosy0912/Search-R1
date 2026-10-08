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
export TREE_INSTRUCT_TOOLS=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmp3g2pwug6/instruct300/candidates/frozen
export TREE_INSTRUCT_PACKAGE_SHA256=11bc6b971ac39e0695e5c322552ca5c61cd38a5e02a0737c57aee409e537bc03
export TREE_PRODUCTION_RELEASE_PATH=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmp3g2pwug6/release.json
export TREE_INSTRUCT_REVIEW_SHA256=78223a258bb9ee3f8f7705f8d3f4b9079cf4765202a4fe976e17401c63aa7bb6
exec bash /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmp3g2pwug6/instruct300/candidates/frozen/launch.sbatch
