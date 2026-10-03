"""Fresh two-update gates; sampled training peaks bound every KV cache increase."""
import os,json,subprocess,re
from pathlib import Path
base=Path(os.environ['E1_BASE']);profile=os.environ['E1_PROFILE'];control=base/'source/experiments/search_e1/control';python=str(base/'env/bin/python')
receipts=base/'receipts';history=[];selected=None
# Keep at least 6 GiB beyond observed peaks for longer trajectories and temporary tensors.
reserve_mib=6144
for candidate in [.4,.5,.6,.7,.8,.9]:
 if history:
  last=history[-1]
  if any(last['peaks_mib'].get(row[0],int(row[3])) + (candidate-last['utilization'])*int(row[2]) > int(row[2])-reserve_mib for row in last['admission']['devices']):
   break
 stage='smoke-'+str(round(candidate*100)).zfill(3)
 env=dict(os.environ,E1_STAGE=stage,E1_VLLM_UTIL=str(candidate))
 subprocess.run([python,str(control/'run_checked.py')],env=env,check=True)
 out=base/'runs'/profile/stage;log=(out/'train.log').read_text()
 if 'SMOKE_STOP_AFTER_REAL_UPDATES 2' not in log:raise RuntimeError('Two real updates not completed')
 if re.search(r'actor/(?:pg_loss|grad_norm|kl_loss)\s*[:=]\s*(?:nan|inf)',log,re.I):raise RuntimeError('Nonfinite gate metric')
 for n in [1,2]:
  if not (out/'actor'/f'global_step_{n}'/'config.json').exists():raise RuntimeError('Missing real checkpoint')
 subprocess.run([python,str(control/'verify_checkpoint.py'),'--initial',str(base/'assets/model'),'--checkpoint',str(out/'actor/global_step_2'),'--report',str(receipts/f'{profile}-{stage}-checkpoint.json')],check=True)
 record=json.loads((receipts/f'{profile}-{stage}.json').read_text());history.append(record);selected=candidate
 (receipts/f'{profile}-memory-tuning.json').write_text(json.dumps(dict(gates=history,reserve_mib=reserve_mib,selected=selected,measurement='nvidia-smi sampled every 2 seconds; not an instantaneous allocator maximum'),indent=2))
if selected is None:raise RuntimeError('No real-update gate accepted')
(receipts/f'{profile}-selected-util.txt').write_text(str(selected)+'\n')
(receipts/f'{profile}-gpu-gate.accepted').touch()
