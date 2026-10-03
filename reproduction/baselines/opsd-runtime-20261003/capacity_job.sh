#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Owned Slurm allocation required}"
export SEARCH_R1_BASE=/mnt/data/shaoshitong/search-r1-rl/opsd
export SEARCH_R1_ASSETS="$SEARCH_R1_BASE/assets-root" SEARCH_R1_REPO="$SEARCH_R1_BASE/repo"
root=$SEARCH_R1_BASE
run="$root/runs/opsd-capacity-$SLURM_JOB_ID"
[[ ! -e "$run" && ! -L "$run" ]]
mkdir -p "$root/runs"
mkdir "$run" "$run/evidence"
export OPSD_EVIDENCE="$run/evidence" OPSD_TEACHER="$SEARCH_R1_ASSETS/assets/qwen2.5-3b"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME="$root/cache/hf" NCCL_NVLS_ENABLE=0
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
source "$root/env/bin/activate"
cd "$SEARCH_R1_REPO"
trap 'rc=$?; printf "%s\n" "$rc" > "$run/exit.txt"' EXIT
: "${OPSD_EXPECTED_COMMIT:?Pinned source required}"
[[ "$(git rev-parse HEAD)" == "$OPSD_EXPECTED_COMMIT" ]]
[[ "${SLURM_JOB_NODELIST:-}" == h100-10 && "$(hostname -s)" == h100-10 ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 96 ]]
[[ -f "$root/logs/data.ready" && -f "$root/logs/environment.ready" ]]
# A dynamic descriptor cannot overwrite an inherited per-GPU lease FD.
exec {task_lock_fd}>"$root/gpu-job.lock"
flock -n "$task_lock_fd"
git rev-parse HEAD > "$run/source.txt"
scontrol show job "$SLURM_JOB_ID" > "$OPSD_EVIDENCE/slurm-job.txt"
sha256sum "$0" "$root/tools/runtime/capacity_job_entry.sh" "$root/tools/runtime/capacity_supervisor.py" "$root/tools/runtime/microbatch_probe.py" "$root/tools/runtime/http_retrieval_check.py" "$root/tools/runtime/gpu_retrieval_bulk.py" "$root/tools/runtime/resource_sample.py" "$SEARCH_R1_REPO/reproduction/h20/retriever_local.py" "$SEARCH_R1_REPO/search_r1/baselines/distillation.py" "$SEARCH_R1_REPO/search_r1/baselines/actor.py" > "$OPSD_EVIDENCE/scripts.sha256"
python -u "$root/tools/runtime/resource_sample.py" "$OPSD_EVIDENCE/resources.ndjson" $$ &
python - <<'PY'
import hashlib,json,os,re,socket,subprocess
from pathlib import Path
from search_r1.baselines.preflight import validate_inherited_gpu_leases
validate_inherited_gpu_leases(8)
assert re.search(r'(?:^|/)job_'+re.escape(os.environ['SLURM_JOB_ID'])+r'(?:/|$)',Path('/proc/self/cgroup').read_text(),re.M)
with socket.socket() as sock: sock.bind(('127.0.0.1',18084))
source=Path(os.environ['SEARCH_R1_BASE'])/'runs/opsd-acceptance-2916/evidence'
raw=(source/'retrieval-comparison.json').read_bytes()
expected='7d76ac364d3bff82fe5b30f383b24c4dc50c849641da629a5a76fd521bbc66b0'
assert hashlib.sha256(raw).hexdigest()==expected
approval=json.loads((source/'retrieval.accepted').read_text())
assert approval['slurm_job_id']=='2916' and approval['comparison_sha256']==expected
out=Path(os.environ['OPSD_EVIDENCE'])
(out/'retrieval-comparison.json').write_bytes(raw)
(out/'retrieval-comparison-reused.json').write_text(json.dumps(dict(source_job_id='2916',current_job_id=os.environ['SLURM_JOB_ID'],comparison_sha256=expected,approval=approval,scope='Previously independently approved immutable-model/index in-process comparison; current live HTTP conformance and bulk24 rerun in this allocation'),indent=2))
print('OWN_LEASES_PORT_AND_APPROVED_COMPARISON_VERIFIED',flush=True)
PY
PYTHONPATH="$root/retriever-gpu-lib:$SEARCH_R1_REPO" python -u reproduction/h20/retriever_local.py --index_path "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" --corpus_path "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" --retriever_model "$SEARCH_R1_ASSETS/assets/e5-base-v2" --device cuda --faiss_gpu --host 127.0.0.1 --port 18084 --threads 8 > "$run/retriever.log" 2>&1 &
retriever_pid=$!
printf '%s\n' "$retriever_pid" > "$OPSD_EVIDENCE/retriever.pid"
python -u "$root/tools/runtime/http_retrieval_check.py"
kill -0 "$retriever_pid"
python -u "$root/tools/runtime/gpu_retrieval_bulk.py"
# The supervisor validates exact eight UUIDs/one owned GPU service PID, then
# repeats the unchanged double15% guard before every isolated capacity candidate.
python -u "$root/tools/runtime/capacity_supervisor.py" 2>&1 | tee "$run/capacity.log"
printf '%s\n' CAPACITY_DIAGNOSTIC_FINISHED_NO_TRAINING > "$run/capacity-finished.txt"
# Slurm cleans up only children in this owned allocation; no foreign process stop.
