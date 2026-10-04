"""Run one isolated stage and record sampled GPU peaks and exit status."""
import os,sys,subprocess,json,time,signal,uuid
from pathlib import Path
base=Path(os.environ['E1_BASE']);profile=os.environ['E1_PROFILE'];stage=os.environ['E1_STAGE']
control=base/'source/experiments/search_e1/control';receipt=base/'receipts'/f'{profile}-{stage}.json'
if (base/'runs'/profile/stage).exists():raise SystemExit('Stage output already exists; preserve it before starting a fresh run')
ids=os.environ['CUDA_VISIBLE_DEVICES'];env=dict(os.environ,E1_RUN_ID=uuid.uuid4().hex,RAY_enable_metrics_collection='0',TZ='UTC')
# Ray teardown may outlive the completed driver. Wait briefly without stopping
# any process; every retry still applies the complete UUID/process admission.
for attempt in range(37):
 admission_result=subprocess.run([str(base/'env/bin/python'),str(control/'gpu_admission.py')],env=env,text=True,capture_output=True)
 if admission_result.returncode==0:
  admission=admission_result.stdout
  break
 if not any(message in admission_result.stderr for message in ['Selected GPU busy/unknown:', 'Selected GPU has compute processes:']) or attempt==36:
  sys.stderr.write(admission_result.stderr)
  raise SystemExit(admission_result.returncode)
 if attempt==0:print('WAITING_UP_TO_180_SECONDS_FOR_IDLE_GPU_ADMISSION',flush=True)
 time.sleep(5)

record=dict(profile=profile,stage=stage,run_id=env['E1_RUN_ID'],admission=json.loads(admission),started_unix=time.time(),status='STARTING',utilization=float(env.get('E1_VLLM_UTIL','.4')),peaks_mib={},sample_count=0)
record['execution_caps']={key:env.get(key) for key in ['E1_PHYSICAL_MICRO','E1_PHYSICAL_SCORE','E1_JIT_OPTIMIZER','E1_LOGIT_CHUNK','E1_OPTIMIZER_CHUNK','E1_STAGE_CHUNK_STATES','E1_GPUS','E1_VIRTUAL_DP_WORLD']}
p=subprocess.Popen(['bash',str(control/'train_profile.sh')],env=env,start_new_session=True)
record.update(pid=p.pid,status='RUNNING')
last=0
code=1
try:
 while p.poll() is None:
  output=subprocess.check_output(['nvidia-smi','-i',ids,'--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
  for line in output.splitlines():
   idx,total,used,util=[x.strip() for x in line.split(',')]
   if used.isdigit():record['peaks_mib'][idx]=max(record['peaks_mib'].get(idx,0),int(used))
  record['sample_count']+=1
  if time.time()-last>=30:
   record['updated_unix']=time.time();receipt.write_text(json.dumps(record,indent=2));last=time.time()
  time.sleep(2)
 code=p.returncode
 record.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,finished_unix=time.time())
except BaseException as error:
 record.update(status='FAILED_EXCEPTION',error=repr(error),finished_unix=time.time())
 raise
finally:
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=60)
 receipt.write_text(json.dumps(record,indent=2))
raise SystemExit(code)
