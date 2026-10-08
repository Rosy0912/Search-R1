#!/usr/bin/env bash
set -euo pipefail
echo 'BLOCKED corrected eval preview: no actual500/HF500/stage or independent activation yet' >&2
exit 78
# Future activation replaces the guard in a NEW independently reviewed source bundle.
: "${SLURM_JOB_ID:?Own fresh Slurm allocation required}"
: "${CORRECTED_EVAL_REPO:?Separate e3 checkout required}"
: "${OPSD_CORRECTED_EVAL_STAGE_SHA256:?External reviewed stage SHA required}"
: "${CORRECTED_EVAL_CONFIG:?Externally pinned resolved config required}"
: "${CORRECTED_EVAL_STAGE:?Externally pinned stage path required}"
: "${CORRECTED_EVAL_BUNDLE:?New reviewed activated tool directory required}"
export SEARCH_R1_BASE=/mnt/data/shaoshitong/search-r1-rl/opsd
export SEARCH_R1_ASSETS="$SEARCH_R1_BASE/assets-root" SEARCH_R1_REPO="$CORRECTED_EVAL_REPO"
export OPSD_TEACHER="$SEARCH_R1_ASSETS/assets/qwen2.5-3b" OPSD_EXPECTED_COMMIT=e3f78f13b58898f7ab2f5a99cac2b76055d47a5a
source "$SEARCH_R1_BASE/env/bin/activate"
cd "$SEARCH_R1_REPO"
: "${CORRECTED_ACTIVATED_BUNDLE_MANIFEST:?Reviewed activated source manifest required}"
: "${CORRECTED_ACTIVATED_BUNDLE_SHA256:?External source manifest SHA required}"
: "${CORRECTED_CANONICAL_TEMPLATE:?Frozen canonical template required}"
python "$CORRECTED_EVAL_BUNDLE/source_bundle_check.py" --bundle "$CORRECTED_ACTIVATED_BUNDLE_MANIFEST" --sha256 "$CORRECTED_ACTIVATED_BUNDLE_SHA256" --canonical "$CORRECTED_CANONICAL_TEMPLATE"
[[ "$SEARCH_R1_REPO" != "$SEARCH_R1_BASE/repo" && "$(git rev-parse HEAD)" == "$OPSD_EXPECTED_COMMIT" ]]
git diff --quiet HEAD
exec 9>"$SEARCH_R1_BASE/gpu-job.lock"
flock -n 9
exec python -m search_r1.baselines.preflight -- python "$CORRECTED_EVAL_BUNDLE/eval_job.py" --config "$CORRECTED_EVAL_CONFIG" --stage "$CORRECTED_EVAL_STAGE" --stage-sha256 "$OPSD_CORRECTED_EVAL_STAGE_SHA256"
