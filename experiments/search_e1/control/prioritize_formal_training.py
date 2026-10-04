"""Finish the exact live gate, then prioritize formal training over more tuning."""
import os,json,time,signal,subprocess,re
from pathlib import Path
from verify_checkpoint import verify
base=Path(os.environ['E1_BASE']);profile=os.environ['E1_PROFILE'];stage=os.environ['E1_GATE_STAGE'];run_id=os.environ['E1_GATE_RUN_ID'];pid=int(os.environ['E1_TUNER_PID']);session=os.environ['E1_OLD_PIPELINE_SESSION']
control=base/'source/experiments/search_e1/control';python=str(base/'env/bin/python');receipts=base/'receipts'
def check_owner():
 proc=Path('/proc')/str(pid)
 environ=proc.joinpath('environ').read_bytes().split(b'\0')
 assert ('E1_BASE='+str(base)).encode() in environ and ('E1_PROFILE='+profile).encode() in environ
 assert str(control/'tune_grpo_memory.py').encode() in proc.joinpath('cmdline').read_bytes()
check_owner();os.kill(pid,signal.SIGSTOP)
print('PAUSED_OWN_TUNER_ONLY; LIVE_UPDATE_GATE_CONTINUES',pid,run_id,flush=True)
try:
 for _ in range(1440):
  record=json.loads((receipts/f'{profile}-{stage}.json').read_text());assert record['run_id']==run_id
  if record['status'] not in ['RUNNING','STARTING']:break
  time.sleep(30)
 else:raise RuntimeError('Live gate timed out')
 assert record['status']=='COMPLETED' and record['exit_code']==0
 for key in ['E1_PHYSICAL_MICRO','E1_PHYSICAL_SCORE','E1_JIT_OPTIMIZER','E1_LOGIT_CHUNK']:assert record['execution_caps'].get(key)==os.environ.get(key)
 subprocess.run([python,str(control/'verify_runtime.py')],check=True)
 out=base/'runs'/profile/stage;log=(out/'train.log').read_text()
 assert 'SMOKE_STOP_AFTER_REAL_UPDATES 2' in log
 assert not re.search(r'actor/(?:pg_loss|grad_norm|kl_loss)\s*[:=]\s*(?:nan|inf)',log,re.I)
 for step in [1,2]:assert (out/'actor'/f'global_step_{step}'/'config.json').is_file()
 report=verify(base/'assets/model',out/'actor/global_step_2');assert report['accepted']
 assert all(record['peaks_mib'][row[0]]<=int(row[2])-6144 for row in record['admission']['devices']),'Real gate lacks6GiB reserve'
 history_path=receipts/f'{profile}-memory-tuning.json'
 history=json.loads(history_path.read_text()) if history_path.exists() else dict(gates=[],reserve_mib=6144,measurement='nvidia-smi sampled every2seconds')
 assert not history['gates'] or record['utilization']>history['selected']
 history['gates'].append(record);history['selected']=record['utilization'];history['priority']='User requires both formal runs complete at least5 iterations before yielding'
 (receipts/f'{profile}-{stage}-checkpoint.json').write_text(json.dumps(report,indent=2));history_path.write_text(json.dumps(history,indent=2))
 (receipts/f'{profile}-selected-util.txt').write_text(str(record['utilization'])+'\n');(receipts/f'{profile}-gpu-gate.accepted').touch()
 check_owner();os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 for _ in range(60):
  if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
  time.sleep(3)
 else:raise RuntimeError('Previous pipeline still active; refusing second supervisor')
 print('EXACT_GATE_ACCEPTED; STARTING_FORMAL_PIPELINE',run_id,flush=True)
 env=dict(os.environ,E1_REUSE_RUNTIME='1',E1_RESUME_GATES='1',E1_UTIL_CANDIDATES=str(record['utilization']),RAY_enable_metrics_collection='0',TZ='UTC')
 subprocess.run(['bash',str(control/'pipeline.sh')],env=env,check=True)
except BaseException:
 if Path('/proc',str(pid)).exists():
  check_owner();os.kill(pid,signal.SIGCONT)
 raise
