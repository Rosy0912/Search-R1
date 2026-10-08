set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
JOB=${1:?jobid}
case "$JOB" in ''|*[!0-9]*) exit 2;; esac
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
RUN="$R/runs/bapo-accept5-$JOB"
test -f "$RUN/TRAIN_RETURNED_SUCCESSFULLY"
"$R/env/bin/python" "$R/tools/bapo-five-iter-evidence-gpu64-tied.py" --run-root "$RUN" --log "$RUN/train.log" --base-model "$R/assets-root/assets/qwen2.5-3b" --compare-weights --hf-load --hash-checkpoint --output "$R/logs/accept5-$JOB-gpu64-tied-extraction.json" > "$R/logs/accept5-$JOB-gpu64-tied-extractor.log" 2>&1
"$R/env/bin/python" - "$R/logs/accept5-$JOB-gpu64-tied-extraction.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['stage_observed_steps']==[1,2,3,4,5]
assert r['expected_outer_updates']==5
assert set(r['outer_step_checks'])==set(map(str,range(1,6)))
assert all(z['status']=='NUMERIC_SUMMARY_PRESENT' and not z['issues'] for z in r['outer_step_checks'].values())
print(json.dumps({k:r[k]for k in ['run_root','expected_outer_updates','outer_step_checks','checkpoint']},indent=2))
PY
