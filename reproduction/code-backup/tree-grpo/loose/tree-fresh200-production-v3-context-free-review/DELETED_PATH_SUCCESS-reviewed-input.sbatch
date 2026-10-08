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
export TREE_INSTRUCT_TOOLS=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpev9n4_bs/instruct300/candidates/frozen
export TREE_INSTRUCT_PACKAGE_SHA256=e4f4760806e89aec05138e2622ee35373d75f2132d592b91c2957d635afd35bd
export TREE_PRODUCTION_RELEASE_PATH=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpev9n4_bs/release.json
export TREE_INSTRUCT_REVIEW_SHA256=8b78044adcba7ef17381cbe8e3daa3abf897ae7e9a72a7a806ca41f5b9270663
exec bash /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fresh200-production-v3-context-free-review/tmpev9n4_bs/instruct300/candidates/frozen/launch.sbatch
