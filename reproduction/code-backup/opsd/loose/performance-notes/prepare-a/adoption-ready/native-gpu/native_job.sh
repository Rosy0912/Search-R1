#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Owned Slurm allocation required}"
export SEARCH_R1_BASE=/mnt/data/shaoshitong/search-r1-rl/opsd
export SEARCH_R1_ASSETS="$SEARCH_R1_BASE/assets-root" SEARCH_R1_REPO="$SEARCH_R1_BASE/repo"
root=$SEARCH_R1_BASE
run="$root/runs/opsd-a-native-$SLURM_JOB_ID"
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
[[ "$OPSD_EXPECTED_COMMIT" != e43fbfa73088ed70ed3efef79ba561507c0afc33 && "$(git rev-parse HEAD)" == "$OPSD_EXPECTED_COMMIT" ]]
[[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" && -z "${OPSD_ACTOR_ALLOCATOR_CONF:-}" ]]
[[ "${SLURM_JOB_NODELIST:-}" == h100-10 && "$(hostname -s)" == h100-10 ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 96 ]]
[[ -f "$root/logs/data.ready" && -f "$root/logs/environment.ready" ]]
# A dynamic descriptor cannot overwrite an inherited per-GPU lease FD.
exec {task_lock_fd}>"$root/gpu-job.lock"
flock -n "$task_lock_fd"
git rev-parse HEAD > "$run/source.txt"
: "${OPSD_NATIVE_MANIFEST_SHA256:?Approved adopted-source native helper manifest required}"
python - <<'PIN'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['SEARCH_R1_BASE'])/'tools/opsd-a-native'
m=root/'native-manifest.json'
assert hashlib.sha256(m.read_bytes()).hexdigest()==os.environ['OPSD_NATIVE_MANIFEST_SHA256']
manifest=json.loads(m.read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in manifest['sha256'].items())
assert hashlib.sha256((Path(os.environ['SEARCH_R1_REPO'])/'search_r1/baselines/distillation.py').read_bytes()).hexdigest()=='2131315b6770779f0c7579bf2a6139e742cf75de03b7ee9f44662f8c32a870cf'
PIN
scontrol show job "$SLURM_JOB_ID" > "$OPSD_EVIDENCE/slurm-job.txt"
sha256sum "$0" "$root/tools/opsd-a-native/native_job_entry.sh" "$root/tools/opsd-a-native/native_supervisor.py" "$root/tools/opsd-a-native/microbatch_probe.py" "$root/tools/opsd-a-native/native-manifest.json" "$root/tools/runtime/http_retrieval_check.py" "$root/tools/runtime/gpu_retrieval_bulk.py" "$root/tools/runtime/resource_sample.py" "$SEARCH_R1_REPO/reproduction/h20/retriever_local.py" "$SEARCH_R1_REPO/search_r1/baselines/distillation.py" "$SEARCH_R1_REPO/search_r1/baselines/actor.py" > "$OPSD_EVIDENCE/scripts.sha256"
python -u "$root/tools/runtime/resource_sample.py" "$OPSD_EVIDENCE/resources.ndjson" $$ &
python - <<'PY'
import hashlib,json,os,re,socket,subprocess
from pathlib import Path
from search_r1.baselines.preflight import validate_inherited_gpu_leases
validate_inherited_gpu_leases(8)
assert re.search(r'(?:^|/)job_'+re.escape(os.environ['SLURM_JOB_ID'])+r'(?:/|$)',Path('/proc/self/cgroup').read_text(),re.M)
with socket.socket() as sock: sock.bind(('127.0.0.1',18084))
source=Path(os.environ['SEARCH_R1_BASE'])/'runs/opsd-acceptance-2960/evidence'
raw=(source/'retrieval-comparison.json').read_bytes()
expected='7b3821cc657154a23bb4314fad700a8e7e6330e12051d09ebe58d5b72b41bdb4'
assert hashlib.sha256(raw).hexdigest()==expected
approval=json.loads((source/'retrieval.accepted').read_text())
assert approval['slurm_job_id']=='2960' and approval['comparison_sha256']==expected
out=Path(os.environ['OPSD_EVIDENCE'])
(out/'retrieval-comparison.json').write_bytes(raw)
(out/'retrieval-comparison-reused.json').write_text(json.dumps(dict(source_job_id='2960',current_job_id=os.environ['SLURM_JOB_ID'],comparison_sha256=expected,approval=approval,scope='Previous independently approved job2960 fixed8 current-query preflight comparison reused only; current live HTTP conformance and bulk24 rerun in this allocation; not new production acceptance'),indent=2))
print('OWN_LEASES_PORT_AND_APPROVED_COMPARISON_VERIFIED',flush=True)
PY
PYTHONPATH="$root/retriever-gpu-lib:$SEARCH_R1_REPO" python -u reproduction/h20/retriever_local.py --index_path "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" --corpus_path "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" --retriever_model "$SEARCH_R1_ASSETS/assets/e5-base-v2" --device cuda --faiss_gpu --host 127.0.0.1 --port 18084 --threads 8 > "$run/retriever.log" 2>&1 &
retriever_pid=$!
printf '%s\n' "$retriever_pid" > "$OPSD_EVIDENCE/retriever.pid"
python -u "$root/tools/runtime/http_retrieval_check.py"
kill -0 "$retriever_pid"
python -u "$root/tools/runtime/gpu_retrieval_bulk.py"
# The supervisor validates exact eight UUIDs/one owned GPU service PID, then
# repeats the unchanged double15% guard before the isolated adopted-source capacity group.
python -u "$root/tools/opsd-a-native/native_supervisor.py" 2>&1 | tee "$run/native-capacity.log"
printf '%s\n' ADOPTED_A_NATIVE_CAPACITY_FINISHED_NO_PRODUCTION_TRAINING > "$run/native-capacity-finished.txt"
# Slurm cleans up only children in this owned allocation; no foreign process stop.
