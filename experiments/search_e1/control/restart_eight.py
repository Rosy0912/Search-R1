"""Owned-only transition at a completed iteration; never resume model-only Adam."""
import os,json,time,signal,subprocess,re
from pathlib import Path
old=Path('/data/shaoshitong/search-e1-config-study-20261004');new=Path('/data/shaoshitong/search-e1-config-study-8gpu-20261004');run='e2e33c84f289410384586e04b45e7497'
while not (new/'receipts/migration-backend.accepted').exists():time.sleep(5)
print('WAITING_FOR_OWNED_R1_COMPLETED_STEP_7',flush=True)
for _ in range(1440):
 record=json.loads((old/'receipts/r1-round-1-grpo.json').read_text());assert record['run_id']==run
 log=(old/'runs/r1/round-1-grpo/train.log').read_text()
 steps=[int(v) for v in re.findall(r'\bstep:(\d+)\s+-',log)]
 if max(steps,default=0)>=7:break
 assert record['status']=='RUNNING',record
 time.sleep(5)
else:raise RuntimeError('Old run did not reach boundary')
# Recheck the four newly freed cards before stopping a still-useful run.
subprocess.run([str(new/'env/bin/python'),str(new/'source/experiments/search_e1/control/gpu_admission.py')],env=dict(os.environ,E1_BASE=str(new),CUDA_VISIBLE_DEVICES='4,5,6,7'),check=True)
# Verify every member of the private stage group before terminating it.
owned=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  pid=int(proc.name)
  if os.getpgid(pid)!=2986860:continue
  assert proc.stat().st_uid==os.getuid()
  marker=('E1_RUN_ID='+run).encode()
  if marker not in proc.joinpath('environ').read_bytes().split(b'\0'):
   # Ray multiprocessing dashboard strips environment variables. Admit only
   # its exact child with a freshly verified private-dashboard parent.
   command=proc.joinpath('cmdline').read_bytes()
   assert b'ray-dashboard-' in command and b'--multiprocessing-fork' in command and str(old/'env/bin/python').encode() in command,pid
   status=proc.joinpath('status').read_text();parent=int(re.search(r'^PPid:\s+(\d+)',status,re.M).group(1));pp=Path('/proc')/str(parent)
   assert marker in pp.joinpath('environ').read_bytes().split(b'\0')
   assert b'/ray/dashboard/dashboard.py' in pp.joinpath('cmdline').read_bytes()
   assert os.getsid(pid)==2986860 and os.getpgid(parent)==2986860 and pp.stat().st_uid==os.getuid()

  owned.append(pid)
 except (FileNotFoundError,ProcessLookupError):continue
assert 2986860 in owned and 2989937 in owned
proof=dict(user_choice='fresh common initial weights on eight GPUs; preserve old four-GPU run',completed_steps=steps,old_run_id=run,old_receipt=record,terminated_owned_group=2986860,owned_pids=owned,time=time.time())
(new/'receipts/four-gpu-transition.json').write_text(json.dumps(proof,indent=2));(old/'receipts/fresh-eight-gpu-transition.json').write_text(json.dumps(proof,indent=2))
os.killpg(2986860,signal.SIGTERM)
for _ in range(60):
 time.sleep(2)
 remaining=[]
 for pid in owned:
  p=Path('/proc')/str(pid)
  if p.exists():
   try:
    if ('E1_RUN_ID='+run).encode() in p.joinpath('environ').read_bytes().split(b'\0'):remaining.append(pid)
   except OSError:pass
 if not remaining:break
else:
 # Only the identities captured and freshly rechecked above are eligible.
 for pid in remaining:
  p=Path('/proc')/str(pid)
  if ('E1_RUN_ID='+run).encode() in p.joinpath('environ').read_bytes().split(b'\0'):os.kill(pid,signal.SIGKILL)
print('OLD_FOUR_GPU_PRESERVED; STARTING_FRESH_EIGHT_GPU_GATES',flush=True)
env=dict(os.environ,E1_BASE=str(new),E1_PROFILE='r1',E1_GPUS='8',E1_VIRTUAL_DP_WORLD='4',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',E1_REUSE_RUNTIME='1',E1_RESUME_GATES='0',E1_UTIL_CANDIDATES='.25',E1_PHYSICAL_MICRO='1',E1_PHYSICAL_SCORE='1',E1_JIT_OPTIMIZER='1',E1_LOGIT_CHUNK='128',E1_OPTIMIZER_CHUNK='1',E1_STAGE_CHUNK_STATES='1',E1_RETRIEVER_URL='http://10.120.17.101:18001/retrieve',RAY_enable_metrics_collection='0',TZ='UTC')
subprocess.run(['bash',str(new/'source/experiments/search_e1/control/pipeline.sh')],env=env,check=True)
