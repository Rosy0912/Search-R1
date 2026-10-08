#!/usr/bin/env bash
# Candidate entry only: this does not submit Slurm or modify an existing run.
set -euo pipefail
: "${BAPO_FRESH200_MODEL:?official model directory required}"
: "${BAPO_FRESH200_DATA:?shared prepared data directory required}"
: "${BAPO_FRESH200_RUN:?new independent output directory required}"
: "${BAPO_ALIGNED_DATA_MANIFEST:?common data manifest required}"
: "${BAPO_ALIGNED_DATA_SHA256:?common data manifest SHA256 required}"
: "${BAPO_ALIGNED_ORDER_MANIFEST:?pinned baseline question order required}"
: "${BAPO_ALIGNED_ORDER_SHA256:?pinned baseline order SHA256 required}"
: "${BAPO_ALIGNED_RETRIEVAL_MANIFEST:?baseline retrieval identity required}"
: "${BAPO_ALIGNED_RETRIEVAL_SHA256:?baseline retrieval identity SHA256 required}"
: "${BAPO_FRESH200_RETRIEVER_URL:?training-node loopback GPU retrieval required}"
if [[ -e "$BAPO_FRESH200_RUN/checkpoints" || -e "$BAPO_FRESH200_RUN/train.log" ]]; then
  echo 'Refusing to overwrite/reuse an existing run; resume requires explicit full-state entry.' >&2
  exit 1
fi
BAPO_RUN_ID="$BAPO_FRESH200_RUN"; export BAPO_RUN_ID
repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_dir"
source_commit="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Refusing a dirty experiment source.' >&2; exit 1
fi
: "${SEARCH_R1_GPU_RESOURCE_GUARD:?launch under the own UUID resource supervisor}"
"${BAPO_PYTHON:-python}" -m search_r1.baselines.aligned_startup --run "$BAPO_FRESH200_RUN" --source "$source_commit"
exec "${BAPO_PYTHON:-python}" -m search_r1.baselines.preflight -- \
  "${BAPO_PYTHON:-python}" -m search_r1.baselines.main \
  --config-name ppo_bapo_baseline_aligned \
  "baselines.algorithm_source_commit=$source_commit" "$@"
