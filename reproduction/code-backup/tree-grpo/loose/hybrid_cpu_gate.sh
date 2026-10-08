#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
RUN="$ROOT/runs/hybrid-cpu-gate-$SLURM_JOB_ID"
mkdir "$RUN"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/repo"
export HF_DATASETS_CACHE="$ROOT/retriever-cache/datasets" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
START=$(date +%s)
PID=''
finish() {
 rc=$?
 trap - EXIT
 if [ -n "$PID" ]; then
  kill -TERM "$PID" 2>/dev/null || true
  for n in $(seq 1 15); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$PID" 2>/dev/null; then kill -KILL "$PID" 2>/dev/null || true; fi
  wait "$PID" 2>/dev/null || true
 fi
 END=$(date +%s)
 printf '{"job_id":%s,"start_epoch":%s,"end_epoch":%s,"allocated_gpus":0,"cpus":8,"exit_code":%s,"scope":"CPU gate body excludes scheduler prolog/epilog; auxiliary GPU cost separately recorded"}\n' "$SLURM_JOB_ID" "$START" "$END" "$rc" > "$RUN/cost.json"
 exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$ROOT/env/bin/python" "$ROOT/tools/hybrid_retrieval_server.py" --instance-json "$ROOT/logs/index-candidate-instance.json" --run "$RUN/service" > "$RUN/service.log" 2>&1 &
PID=$!
for n in $(seq 1 180); do
 kill -0 "$PID"
 if curl -fsS --max-time 5 http://127.0.0.1:18082/provenance > "$RUN/provenance.json"; then break; fi
 sleep 2
done
test -s "$RUN/provenance.json"
"$ROOT/env/bin/python" "$ROOT/tools/probe_aux_http.py" --url http://127.0.0.1:18082 --reference "$ROOT/runs/retrieval-2854/cpu.json" --output "$RUN/http-difference.json"
printf '%s\n' "$RUN" > "$ROOT/logs/hybrid-cpu-gate-completed.run"
