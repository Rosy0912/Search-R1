#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Must run inside own srun allocation}"
export SEARCH_R1_BASE=/mnt/data/shaoshitong/search-r1-rl/opsd
export SEARCH_R1_ASSETS="$SEARCH_R1_BASE/assets-root" SEARCH_R1_REPO="$SEARCH_R1_BASE/repo"
root=$SEARCH_R1_BASE
run="$root/runs/opsd-formal-$SLURM_JOB_ID"
runtime="/tmp/o4-$SLURM_JOB_ID"
[[ ! -e "$run" && ! -L "$run" && ! -e "$runtime" && ! -L "$runtime" ]]
mkdir -p "$root/runs"
mkdir "$run" "$runtime"
mkdir "$run/evidence" "$runtime/tmp" "$runtime/ray"
ln -s "$runtime" "$run/runtime"
export OPSD_EVIDENCE="$run/evidence" SEARCH_R1_BASELINE_OUTPUT="$run/output" SEARCH_R1_BASELINE_RUN_ROOT="$runtime"
export TMPDIR="$runtime/tmp" RAY_TMPDIR="$runtime/ray" HF_HOME="$root/cache/hf" XDG_CACHE_HOME="$runtime/cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0 VLLM_ATTENTION_BACKEND=XFORMERS OMP_NUM_THREADS=8 RAY_DEDUP_LOGS=0 RAY_USAGE_STATS_ENABLED=0 VLLM_NO_USAGE_STATS=1
export OPSD_TEACHER="$SEARCH_R1_ASSETS/assets/qwen2.5-3b" OPSD_GLOBAL_MICROBATCH=16
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
source "$root/env/bin/activate"
cd "$SEARCH_R1_REPO"
[[ -f "$root/logs/data.ready" && -f "$root/logs/environment.ready" ]]
exec {task_lock_fd}>"$root/gpu-job.lock"
flock -n "$task_lock_fd"
[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 96 ]]
python - <<'PYCODE'
import os,socket
# Ray appends its timestamp/session/socket names to this directory.
assert len(os.fsencode(os.environ['RAY_TMPDIR']+'/session_2026-10-03_05-22-02_615594_9999999999/sockets/plasma_store')) <= 107
for port in (18084,16384):
    with socket.socket() as s: s.bind(('127.0.0.1',port))
print('OWN_PORTS_FREE',flush=True)
PYCODE
python "$root/tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-fd4b1b34cf20/verify_formal_prerequisites.py" --cleanup
# Slurm owns cleanup of all child processes. This trap records failure only.
trap 'rc=$?; printf "%s\n" "$rc" > "$run/exit.txt"' EXIT
scontrol show job "$SLURM_JOB_ID" > "$OPSD_EVIDENCE/slurm-job.txt"
git rev-parse HEAD > "$OPSD_EVIDENCE/commit.txt"
sha256sum "$0" "$root/tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-fd4b1b34cf20/formal_entry.sh" "$root/tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-fd4b1b34cf20/formal_supervisor.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/semantic_device_gate.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/gpu_teacher_probe-factor-dtype.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/gpu_loss_checkpoint_equivalence.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/capacity_supervisor.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/microbatch_probe.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/gpu_retrieval_bulk.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/http_retrieval_check.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/retrieval_compare.py" "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/resource_sample.py" > "$OPSD_EVIDENCE/scripts.sha256"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$OPSD_EVIDENCE/pre-gpus.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv > "$OPSD_EVIDENCE/pre-processes.csv"
python -u "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/resource_sample.py" "$OPSD_EVIDENCE/resources.ndjson" $$ &
python - <<'PY'
import os
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
inv,proc=gpu_snapshot(); records=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inv,proc)
assert len(records)==8
print('EIGHT_GPU_THRESHOLD_PREFLIGHT_OK',records,flush=True)
PY
# Check the actual isolated Ray runtime before loading expensive model/index assets.
ray start --head --port=16384 --num-gpus=8 --num-cpus=80 --object-store-memory=8589934592 --include-dashboard=false --min-worker-port=30000 --max-worker-port=30999 --temp-dir="$runtime/ray"
export RAY_ADDRESS=127.0.0.1:16384
python - <<'PYRAY'
import os,ray
from pathlib import Path
ray.init(address=os.environ['RAY_ADDRESS'])
resources=ray.cluster_resources()
assert resources.get('GPU')==8 and resources.get('CPU')==80, resources
@ray.remote(num_cpus=1)
def own_job_task():
    import os
    from pathlib import Path
    return {'pid':os.getpid(),'job':os.environ['SLURM_JOB_ID'],'cgroup':Path('/proc/self/cgroup').read_text()}
result=ray.get(own_job_task.remote(),timeout=60)
assert result['job']==os.environ['SLURM_JOB_ID'] and ('job_'+result['job']+'/') in result['cgroup'],result
print('RAY_ISOLATED_CPU_TASK_OK',resources,result,flush=True)
ray.shutdown()
PYRAY
# Wheel was installed before job-lock acquisition; trainer environment remains CPU FAISS.
torchrun --standalone --nproc_per_node=8 reproduction/h20/cuda_preflight.py
python -u "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/model_preflight.py"
python - <<'PYCODE'
import os
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
inv,proc=gpu_snapshot();check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inv,proc)
print('POST_MODEL_PREFLIGHT_GPU_IDLE_OK',flush=True)
PYCODE
PYTHONPATH="$root/retriever-gpu-lib:$SEARCH_R1_REPO" python -u "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/retrieval_compare.py"
# Independent review reads exact real comparison evidence before training.
printf '%s\n' "$OPSD_EVIDENCE/retrieval-comparison.json" > "$run/retrieval-review-pending.txt"
for attempt in $(seq 1 240); do
    [[ -f "$OPSD_EVIDENCE/retrieval.accepted" ]] && break
    sleep 30
done
[[ -f "$OPSD_EVIDENCE/retrieval.accepted" ]]
python - <<'PYCODE'
import hashlib,json,os
from pathlib import Path
out=Path(os.environ['OPSD_EVIDENCE'])
record=json.loads((out/'retrieval.accepted').read_text())
assert record['slurm_job_id']==os.environ['SLURM_JOB_ID']
assert record['comparison_sha256']==hashlib.sha256((out/'retrieval-comparison.json').read_bytes()).hexdigest()
print('CURRENT_RETRIEVAL_EVIDENCE_REVIEWED',flush=True)
PYCODE
PYTHONPATH="$root/retriever-gpu-lib:$SEARCH_R1_REPO" python -u reproduction/h20/retriever_local.py --index_path "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" --corpus_path "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" --retriever_model "$SEARCH_R1_ASSETS/assets/e5-base-v2" --device cuda --faiss_gpu --host 127.0.0.1 --port 18084 --threads 8 > "$run/retriever.log" 2>&1 &
retriever_pid=$!
printf '%s\n' "$retriever_pid" > "$OPSD_EVIDENCE/retriever.pid"
python -u "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/http_retrieval_check.py"
kill -0 "$retriever_pid"
python - <<'PYCODE'
import os,subprocess
from pathlib import Path
pid=int((Path(os.environ['OPSD_EVIDENCE'])/'retriever.pid').read_text())
assert ('job_'+os.environ['SLURM_JOB_ID']+'/') in Path(f'/proc/{pid}/cgroup').read_text()
listeners=subprocess.check_output(['ss','-ltnp','sport = :18084'],text=True)
assert f'pid={pid},' in listeners,listeners
print('GPU_RETRIEVER_OWN_PID_JOB_PORT_CONFIRMED',flush=True)
PYCODE
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$OPSD_EVIDENCE/retriever-resident-gpus.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv > "$OPSD_EVIDENCE/retriever-resident-processes.csv"
python - <<'PYGPU'
import os,subprocess
from pathlib import Path
pid=int((Path(os.environ['OPSD_EVIDENCE'])/'retriever.pid').read_text())
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
assert len(rows)==8 and all(int(r.split(',')[1])==pid for r in rows),rows
actual={r.split(',')[0].strip() for r in rows}
from search_r1.baselines.preflight import gpu_snapshot,parse_gpu_inventory
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
inv=parse_gpu_inventory(raw)
selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
expected={v['uuid'] for i,v in inv.items() if i in selected or v['uuid'] in selected}
assert len(actual)==8 and actual==expected,(actual,expected)
print('OWN_RETRIEVER_RESIDENT_ON_ALL_EIGHT_GPUS',flush=True)
PYGPU
python -u "$root/tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/gpu_retrieval_bulk.py"
python -u "$root/tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-fd4b1b34cf20/formal_supervisor.py"
printf '%s\n' FORMAL_500_OUTERS_AND_FINAL_EVALUATION_EXITED > "$run/training-finished.txt"
