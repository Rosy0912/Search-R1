set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
RUN=$R/runs/cosearch-local-gpu-five-20261003-k
"$R/env/bin/python" "$R/tools/capture_worker_k.py" > "$RUN/last-supervision-capture.txt"
"$R/env/bin/python" "$R/tools/audit_gpu_k.py"
"$R/env/bin/python" - <<'PY'
from pathlib import Path
import json,subprocess,time
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/runs/cosearch-local-gpu-five-20261003-k')
s=(r/'worker-training.log').read_text(errors='replace')
def entries(prefix):
 result=[]
 for line in s.splitlines():
  if prefix in line:result.append(json.JSONDecoder().raw_decode(line.split(prefix,1)[1])[0])
 return result
outer=entries('COSEARCH_OUTER_METRICS_JSON=')
progress=entries('COSEARCH_BATCH_PROGRESS_JSON=')
rank=entries('COSEARCH_RANK_PROGRESS_JSON=')
record=dict(run=r.name,job=2961,checked_at=time.time(),outer_steps=[x['step'] for x in outer],
 latest_progress=progress[-1:] or None,latest_rank=rank[-1:] or None,
 driver_exit=json.loads((r/'driver-exit.json').read_text()) if (r/'driver-exit.json').exists() else None,
 slurm=subprocess.check_output(['squeue','-h','-j','2961','-o','%i %T %N'],text=True).strip(),
 capture=json.loads((r/'worker-log-provenance.json').read_text()))
if outer:record['latest_outer_metrics']=outer[-1]['metrics']
(r/'supervision-latest.json').write_text(json.dumps(record,indent=2)+'\n')
print('COSEARCH_SUPERVISION_JSON='+json.dumps(record))
if 'Traceback (most recent call last)' in s:print('TRACEBACK_TAIL='+s[-5000:])
PY
bash "$R/tools/start_five_reload_k.sh"
