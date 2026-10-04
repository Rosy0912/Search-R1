#!/usr/bin/env bash
# Reviewable software only: no sbatch, no remote mutation, no retraining.
set -euo pipefail
root=/mnt/data/shaoshitong/search-r1-rl/opsd
: "${SLURM_JOB_ID:?Fresh owned Slurm allocation required}"
: "${OPSD_DISTRIBUTION_BUNDLE:?New root-reviewed bundle directory required}"
: "${OPSD_DISTRIBUTION_PROBE_SHA256:?Actual externally reviewed final probe SHA required}"
: "${OPSD_DISTRIBUTION_SOURCE_SHA256:?Actual externally reviewed source SHA required}"
: "${OPSD_DISTRIBUTION_PROBE:?Actual reviewed probe path required}"
: "${OPSD_DISTRIBUTION_BINDING:?Actual external binding path required}"
: "${OPSD_DISTRIBUTION_BINDING_SHA256:?External binding SHA required}"
: "${OPSD_DISTRIBUTION_LAUNCHER:?Reviewed launcher path required}"
: "${OPSD_DISTRIBUTION_WRAPPER_MANIFEST:?Reviewed wrapper manifest path required}"
: "${OPSD_DISTRIBUTION_WRAPPER_MANIFEST_SHA256:?External wrapper manifest SHA required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_JOB_NODELIST" == h100-10 ]]
source "$root/env/bin/activate"
cd "$root/repo"
[[ "$(git rev-parse HEAD)" == 8e60976d364ab7d7ae51714415813c1359dafe9a ]]
git diff --quiet HEAD
# Keep the requested production global lock descriptor9 across both exec stages.
exec 9>"$root/gpu-job.lock"
flock -n 9
run="$root/runs/opsd-distribution-$SLURM_JOB_ID"
# A pre-existing directory is never modified, including failure markers.
mkdir -- "$run"
export OPSD_DISTRIBUTION_RUN="$run"
export OPSD_DISTRIBUTION_EXECUTING_BOOTSTRAP="$(realpath "$0")"
trap 'rc=$?; printf "%s\n" "$rc" > "$run/bootstrap-exit.txt"' EXIT
python "$OPSD_DISTRIBUTION_LAUNCHER" --verify-wrapper-only
# Native production preflight acquires one actual UUID lease and execs launcher.
# No model/index/tool work occurs in bootstrap or before allocation binding.
python -m search_r1.baselines.preflight -- python -u "$OPSD_DISTRIBUTION_LAUNCHER" > "$run/launcher.log" 2>&1
