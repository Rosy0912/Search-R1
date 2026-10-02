#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Must run inside own srun allocation}"
export SEARCH_R1_BASE=/mnt/data/shaoshitong/search-r1-rl/opsd
export SEARCH_R1_ASSETS="$SEARCH_R1_BASE/assets-root" SEARCH_R1_REPO="$SEARCH_R1_BASE/repo"
root=$SEARCH_R1_BASE
run="$root/runs/opsd-acceptance-$SLURM_JOB_ID"
[[ ! -e "$run" ]]
mkdir -p "$run/evidence" "$run/runtime/tmp" "$run/runtime/ray"
export OPSD_EVIDENCE="$run/evidence" SEARCH_R1_BASELINE_OUTPUT="$run/output" SEARCH_R1_BASELINE_RUN_ROOT="$run/runtime"
export TMPDIR="$run/runtime/tmp" RAY_TMPDIR="$run/runtime/ray" HF_HOME="$root/cache/hf" XDG_CACHE_HOME="$run/runtime/cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0 VLLM_ATTENTION_BACKEND=XFORMERS OMP_NUM_THREADS=8 RAY_DEDUP_LOGS=0
export OPSD_TEACHER="$SEARCH_R1_ASSETS/assets/qwen2.5-3b" OPSD_GLOBAL_MICROBATCH=8
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
source "$root/env/bin/activate"
cd "$SEARCH_R1_REPO"
[[ -f "$root/logs/data.ready" && -f "$root/logs/environment.ready" ]]
exec 9>"$root/gpu-job.lock"
flock -n 9
[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 96 ]]
python - <<'PYCODE'
import socket
for port in (18084,16384):
    with socket.socket() as s: s.bind(('127.0.0.1',port))
print('OWN_PORTS_FREE',flush=True)
PYCODE
# Slurm owns cleanup of all child processes. This trap records failure only.
trap 'rc=$?; printf "%s\n" "$rc" > "$run/exit.txt"' EXIT
scontrol show job "$SLURM_JOB_ID" > "$OPSD_EVIDENCE/slurm-job.txt"
git rev-parse HEAD > "$OPSD_EVIDENCE/commit.txt"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$OPSD_EVIDENCE/pre-gpus.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv > "$OPSD_EVIDENCE/pre-processes.csv"
python -u "$root/tools/runtime/resource_sample.py" "$OPSD_EVIDENCE/resources.ndjson" $$ &
python - <<'PY'
import os
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
inv,proc=gpu_snapshot(); records=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inv,proc)
assert len(records)==8
print('EIGHT_GPU_THRESHOLD_PREFLIGHT_OK',records,flush=True)
PY
# Wheel was installed before job-lock acquisition; trainer environment remains CPU FAISS.
torchrun --standalone --nproc_per_node=8 reproduction/h20/cuda_preflight.py
python -u "$root/tools/runtime/model_preflight.py"
python - <<'PYCODE'
import os
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
inv,proc=gpu_snapshot();check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inv,proc)
print('POST_MODEL_PREFLIGHT_GPU_IDLE_OK',flush=True)
PYCODE
PYTHONPATH="$root/retriever-gpu-lib:$SEARCH_R1_REPO" python -u "$root/tools/runtime/retrieval_compare.py"
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
python -u reproduction/h20/retriever_local.py --index_path "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" --corpus_path "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" --retriever_model "$SEARCH_R1_ASSETS/assets/e5-base-v2" --device cpu --host 127.0.0.1 --port 18084 --threads 8 > "$run/retriever.log" 2>&1 &
retriever_pid=$!
printf '%s\n' "$retriever_pid" > "$OPSD_EVIDENCE/retriever.pid"
python -u "$root/tools/runtime/http_retrieval_check.py"
kill -0 "$retriever_pid"
python - <<'PYCODE'
import os,subprocess
from pathlib import Path
pid=int((Path(os.environ['OPSD_EVIDENCE'])/'retriever.pid').read_text())
assert ('job_'+os.environ['SLURM_JOB_ID']+'/') in Path(f'/proc/{pid}/cgroup').read_text()
listeners=subprocess.check_output(['ss','-ltnp','sport = :18084'],text=True)
assert f'pid={pid},' in listeners,listeners
print('CPU_RETRIEVER_OWN_PID_JOB_PORT_CONFIRMED',flush=True)
PYCODE
ray start --head --port=16384 --num-gpus=8 --num-cpus=80 --object-store-memory=8589934592 --include-dashboard=false --temp-dir="$run/runtime/ray"
export RAY_ADDRESS=127.0.0.1:16384
args=(actor_rollout_ref.actor.ppo_micro_batch_size=8 retriever.url=http://127.0.0.1:18084/retrieve trainer.total_training_steps=6 trainer.smoke_only=true trainer.val_before_train=true trainer.test_freq=5 trainer.experiment_name=opsd-acceptance-5 actor_rollout_ref.rollout.log_prob_micro_batch_size=8 actor_rollout_ref.ref.log_prob_micro_batch_size=8)
bash reproduction/baselines/run.sh opsd "${args[@]}" --cfg job --resolve > "$OPSD_EVIDENCE/resolved.yaml"
"$root/tools/uv-env/bin/uv" pip freeze --python "$root/env/bin/python" > "$OPSD_EVIDENCE/environment-freeze.txt"
bash reproduction/baselines/run.sh opsd "${args[@]}" 2>&1 | tee "$run/training.log"
# HF validation is done in a fresh process after trainer workers exit/Slurm cleanup,
# or in a separate scheduled allocation; successful training alone is not acceptance.
printf '%s\n' TRAINING_FIVE_ITERATIONS_EXITED > "$run/training-finished.txt"
