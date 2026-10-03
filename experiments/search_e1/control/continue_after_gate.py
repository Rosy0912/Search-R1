"""Continue the full pipeline only after this exact running gate was accepted."""
import os,json,time,subprocess
from pathlib import Path
base=Path(os.environ['E1_BASE']);profile=os.environ['E1_PROFILE'];run_id=os.environ['E1_GATE_RUN_ID'];session=os.environ['E1_GATE_SESSION'];stage=os.environ.get('E1_GATE_STAGE','smoke-025')
for _ in range(1440):
 if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(30)
else:raise RuntimeError('Gate wait timed out; no formal training started')
record=json.loads((base/'receipts'/f'{profile}-{stage}.json').read_text())
if record['run_id']!=run_id or record['status']!='COMPLETED' or record['exit_code']!=0:raise RuntimeError('Exact gate did not complete successfully')
tuning=json.loads((base/'receipts'/f'{profile}-memory-tuning.json').read_text())
if tuning['gates'][-1]['run_id']!=run_id or not (base/'receipts'/f'{profile}-gpu-gate.accepted').exists():raise RuntimeError('Exact two-update memory gate was not accepted')
checkpoint=json.loads((base/'receipts'/f'{profile}-{stage}-checkpoint.json').read_text())
if not checkpoint['accepted'] or checkpoint['nonfinite'] or not checkpoint['changed']:raise RuntimeError('Missing finite changed-weight evidence')
print('EXACT_TWO_UPDATE_GATE_ACCEPTED',run_id,flush=True)
env=dict(os.environ,E1_RESUME_GATES='1',E1_REUSE_RUNTIME='1',RAY_enable_metrics_collection='0',TZ='UTC')
subprocess.run(['bash',str(base/'source/experiments/search_e1/control/pipeline.sh')],env=env,check=True)
