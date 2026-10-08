#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import os,json,pathlib,csv,io
from search_r1.baselines.preflight import validate_inherited_gpu_leases
validate_inherited_gpu_leases(8)
records=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
raw=pathlib.Path(os.environ['E1_STATE_JOB_RUN'],'pre-gpus.csv').read_text()
assert {r['uuid'] for r in records}=={row[1].strip() for row in csv.reader(io.StringIO(raw))}
pathlib.Path(os.environ['E1_STATE_JOB_RUN'],'actual-uuid-leases.json').write_text(os.environ['SEARCH_R1_GPU_LEASES'])
print('OWN_EIGHT_UUID_LEASES_JSON='+os.environ['SEARCH_R1_GPU_LEASES'],flush=True)
PY
set +e
torchrun --standalone --nproc_per_node=8 "$SEARCH_R1_REPO/reproduction/baselines/e1-300/native_opsd_state_fixture.py"
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$E1_STATE_JOB_RUN/torchrun-exit.txt"
[[ $task_rc == 0 ]] || exit "$task_rc"
python - <<'PY'
import json,pathlib,os
root=pathlib.Path(os.environ['E1_STATE_FIXTURE_OUTPUT']);records=[]
for rank in range(8):
 p=root/f'rank-{rank}.json';r=json.loads(p.read_text())
 assert r['status']=='NATIVE_EIGHT_RANK_OPSD_FULL_STATE_FIXTURE_PASS_NOT_RL_TRAINING'
 assert r['rank']==rank and r['world_size']==8 and r['device']==rank and r['source']==os.environ['E1_EXPECTED_SOURCE']
 assert len(r['records'])==2 and all(v['next_update_full_state_exact_equal'] and v['same_model_adam_scheduler_objects'] for v in r['records'])
 assert all(v['total_actual_optimizer_calls']==28 and v['initial_optimizer_calls']==20 for v in r['records'])
 records.append(r)
out=dict(status='ACTUAL_NATIVE_EIGHT_RANK_STATE_ORACLE_PASS',rl_outer_updates=0,official3b_capacity_verified=False,source=os.environ['E1_EXPECTED_SOURCE'],job_id=os.environ['SLURM_JOB_ID'],ranks=records)
pathlib.Path(os.environ['E1_STATE_JOB_RUN'],'actual-state-summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print('ACTUAL_NATIVE_EIGHT_RANK_STATE_ORACLE_PASS_NOT_RL_OUTER',flush=True)
PY
