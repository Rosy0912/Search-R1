"""Revalidate a completed gate after an execution-only verifier correction."""
import os,json,re,subprocess
from pathlib import Path
from verify_checkpoint import verify
base=Path(os.environ['E1_BASE']);profile=os.environ['E1_PROFILE'];stage=os.environ['E1_GATE_STAGE'];run_id=os.environ['E1_GATE_RUN_ID'];receipts=base/'receipts'
subprocess.run([str(base/'env/bin/python'),str(base/'source/experiments/search_e1/control/verify_runtime.py')],check=True)
r=json.loads((receipts/f'{profile}-{stage}.json').read_text())
assert r['run_id']==run_id and r['status']=='COMPLETED' and r['exit_code']==0
for key in ['E1_PHYSICAL_MICRO','E1_PHYSICAL_SCORE','E1_JIT_OPTIMIZER']:assert r['execution_caps'].get(key)==os.environ.get(key)
out=base/'runs'/profile/stage;log=(out/'train.log').read_text()
assert 'SMOKE_STOP_AFTER_REAL_UPDATES 2' in log
assert not re.search(r'actor/(?:pg_loss|grad_norm|kl_loss)\s*[:=]\s*(?:nan|inf)',log,re.I)
for n in [1,2]:assert (out/'actor'/f'global_step_{n}'/'config.json').is_file()
report=verify(base/'assets/model',out/'actor/global_step_2');assert report['accepted']
for row in r['admission']['devices']:assert r['peaks_mib'][row[0]]<=int(row[2])-6144,'Insufficient6GiB headroom'
assert not (receipts/f'{profile}-memory-tuning.json').exists(),'Refuse overwriting existing accepted history'
(receipts/f'{profile}-{stage}-checkpoint.json').write_text(json.dumps(report,indent=2))
(receipts/f'{profile}-memory-tuning.json').write_text(json.dumps(dict(gates=[r],reserve_mib=6144,selected=r['utilization'],measurement='nvidia-smi sampled every2seconds',revalidation='Completed unchanged runtime; verified tied-head safe-tensor alias'),indent=2))
(receipts/f'{profile}-selected-util.txt').write_text(str(r['utilization'])+'\n');(receipts/f'{profile}-gpu-gate.accepted').touch()
print('COMPLETED_GATE_REVALIDATED_AND_ACCEPTED',run_id,json.dumps(report),flush=True)
