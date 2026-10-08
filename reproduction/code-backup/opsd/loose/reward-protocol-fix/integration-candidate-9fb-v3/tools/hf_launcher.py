"""Independent HF launcher implementation candidate; immutable checker is unchanged.
Not activated: CLI exits78. run() may only be integrated into a NEW reviewed HF stage.
"""
import json,os,re,subprocess,sys,threading,time,hashlib
from pathlib import Path
from training_evidence import digest
from hf_activation_capture import capture_hf_activation,capture_lease,emit_activation

def run(checker,evidence,root,expected_checker_sha):
 job=os.environ.get('SLURM_JOB_ID','')
 if not re.fullmatch('[0-9]+',job):raise ValueError('Actual Slurm job required before launch')
 checker=Path(checker).resolve();evidence=Path(evidence).resolve();root=Path(root).resolve()
 if not checker.is_relative_to(root) or not evidence.is_relative_to(root) or digest(checker)!=expected_checker_sha:raise ValueError('Immutable checker/path differs before launch')
 from search_r1.baselines import preflight
 from training_evidence import COMMIT
 repo=root/'repo';preflight_path=repo/'search_r1/baselines/preflight.py'
 if Path(preflight.__file__).resolve()!=preflight_path:raise ValueError('Actual preflight import outside frozen training source')
 if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise ValueError('HF launcher training source mismatch')
 subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD'],check=True)
 original=subprocess.check_output(['git','-C',str(repo),'show',COMMIT+':search_r1/baselines/preflight.py'])
 if digest(preflight_path)!=hashlib.sha256(original).hexdigest():raise ValueError('Actual preflight differs from training commit')
 # Reuse actual reviewed lease/15% preflight, BEFORE the immutable checker allocates.
 preflight.validate_inherited_gpu_leases(expected_count=1)
 leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);selected=[v.strip() for v in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if v.strip()]
 inventory,processes=preflight.gpu_snapshot();gpu=preflight.check_selected_gpus(selected,inventory,processes)
 if len(leases)!=1 or len(gpu)!=1 or leases[0]['uuid']!=gpu[0]['uuid']:raise ValueError('Exactly one inherited GPU lease required')
 global_lease=capture_lease(9,root/'gpu-job.lock');per_gpu=leases[0];fd=int(per_gpu['fd']);lock=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(per_gpu['uuid']+'.lock');per_gpu_lease=capture_lease(fd,lock)
 log=evidence/'reload.log'
 if log.exists():raise ValueError('New immutable reload.log required')
 launch=[sys.executable,'-u',str(checker)]
 child=subprocess.Popen(launch,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,pass_fds=tuple({9,fd}),env=dict(os.environ))
 errors=[]
 def stream():
  try:
   with log.open('xb') as handle:
    while True:
     part=child.stdout.read(8192)
     if not part:break
     handle.write(part);handle.flush();sys.stdout.buffer.write(part);sys.stdout.buffer.flush()
  except Exception as error:errors.append(error)
 reader=threading.Thread(target=stream);reader.start();audit=None;deadline=time.monotonic()+30
 try:
  while child.poll() is None and time.monotonic()<deadline:
   try:audit=capture_hf_activation(os.getpid(),checker,job,gpu[0]['uuid'],root/'gpu-job.lock');break
   except (FileNotFoundError,ValueError):time.sleep(.05)
  if audit is None:raise ValueError('Actual immutable checker process capture unavailable')
  audit.update(launch_argv=launch,selected_gpu_preflight=gpu,selected_gpu_processes=processes,selected_cuda_devices=selected,selected_gpu_lease=per_gpu_lease,global_lease_before_launch=global_lease,preflight_source=dict(path=preflight.__file__,sha256=digest(preflight.__file__)),launcher_source_sha256=digest(__file__))
  emit_activation(audit,evidence)
  code=child.wait();reader.join()
  if errors:raise errors[0]
  if code:raise RuntimeError('Immutable HF checker exited nonzero: '+str(code))
  return audit
 finally:
  if child.poll() is None:child.terminate();child.wait();reader.join()

if __name__=='__main__':
 print('BLOCKED HF LAUNCHER CANDIDATE: requires independently reviewed NEW activated HF bundle',file=sys.stderr)
 raise SystemExit(78)
