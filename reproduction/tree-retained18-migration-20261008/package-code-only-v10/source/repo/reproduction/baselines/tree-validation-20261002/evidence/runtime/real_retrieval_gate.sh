#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
source "$ROOT/env/bin/activate"
cd "$ROOT/repo"
test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NODELIST:-}" = h100-9
test -f "$ROOT/logs/derived-assets.ready"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NCCL_NVLS_ENABLE=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [ "${SEARCH_R1_GPU_PREFLIGHT:-0}" != 1 ]; then
 exec python -m search_r1.baselines.preflight -- bash "$ROOT/tools/real_retrieval_gate.sh"
fi
python - <<'CHECK'
import json,os
from types import SimpleNamespace
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus,validate_gpu_lease
validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=8)))
i,p=gpu_snapshot();r=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),i,p)
assert len(r)==8 and {x['uuid'] for x in r}==set(json.load(open('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/expected-gpu-uuids.json')))
CHECK
RUN="$ROOT/runs/retrieval-$SLURM_JOB_ID"
mkdir "$RUN"
exec > >(tee "$RUN/gate.log") 2>&1
start=$(date +%s)
finish() {
 code=$?
 end=$(date +%s)
 printf '{"job_id":%s,"start_epoch":%s,"end_epoch":%s,"allocated_gpus":8,"exit_code":%s,"scope":"gate body wall time, excludes scheduler prolog/epilog"}\n' "$SLURM_JOB_ID" "$start" "$end" "$code" > "$RUN/cost.json"
}
trap finish EXIT
args=(--repo "$ROOT/repo" --work-root "$RUN" --index "$ROOT/assets-root/assets/wiki/e5_Flat.index" --corpus "$ROOT/assets-root/assets/wiki/wiki-18.jsonl" --model "$ROOT/assets-root/assets/e5-base-v2" --asset-manifest "$ROOT/assets-root/assets.json" --hash-large-assets --threads 8 --batch-size 512 --repeats 2)
# Production CPU FAISS and encoder: no GPU overlay in this child.
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/repo" python "$ROOT/tools/real_retrieval_probe.py" "${args[@]}" --mode cpu --embeddings-out "$RUN/cpu-embeddings.npz" --output "$RUN/cpu.json"
# Isolated overlay: applies only to GPU reference children, never trainer env.
for mode in gpu-encoder-cpu-index gpu; do
 PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" python "$ROOT/tools/real_retrieval_probe.py" "${args[@]}" --mode "$mode" --output "$RUN/$mode.json"
 python "$ROOT/tools/real_retrieval_probe.py" --mode compare --left "$RUN/cpu.json" --right "$RUN/$mode.json" --output "$RUN/cpu-vs-$mode.json"
done
PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" python "$ROOT/tools/real_retrieval_probe.py" "${args[@]}" --mode gpu --embeddings-in "$RUN/cpu-embeddings.npz" --output "$RUN/gpu-index-cpu-embeddings.json"
python "$ROOT/tools/real_retrieval_probe.py" --mode compare --left "$RUN/cpu.json" --right "$RUN/gpu-index-cpu-embeddings.json" --output "$RUN/index-only-difference.json"
# No training launch: rank/content/score differences require evidence review.
echo RETRIEVAL_MEASUREMENTS_READY_FOR_REVIEW
printf '%s\n' "$RUN" > "$ROOT/logs/retrieval-measurements.ready"
