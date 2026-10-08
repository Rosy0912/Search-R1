"""Owned h100-9 numerical fixture: same-node, proper Slurm, leases and guard."""
import fcntl,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
from tree_shared_owned_runtime_fixed import ROOT,require,sha,cgroup_scope,OwnedProcesses,gpu_sample
p=Path(sys.argv[1]); manifest=json.loads((p/'manifest.json').read_text())
for file,expected in manifest['files'].items():require(sha(p/file)==expected,'Candidate file identity changed: '+file)
for file,expected in manifest['external_helpers'].items():require(sha(ROOT/'tools'/file)==expected,'Guard/runtime identity changed')
require(socket.gethostname().split('.')[0]=='h100-9' and os.environ['SLURM_JOB_NODELIST']=='h100-9','Wrong node')
job=os.environ['SLURM_JOB_ID'];require(job.isdigit(),'No Slurm job')
from search_r1.baselines.preflight import validate_inherited_gpu_leases,gpu_snapshot,check_selected_gpus
validate_inherited_gpu_leases(8)
lock=(ROOT/'shared-gpu-job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
require(len(selected)==8 and set(json.loads((ROOT/'tools/expected-gpu-uuids.json').read_text()))=={x['uuid'] for x in selected},'UUID changed')
scope,relative=cgroup_scope(job);owned=OwnedProcesses(scope,relative)
allocation=subprocess.check_output(['scontrol','show','job',job],text=True)
require('JobState=RUNNING' in allocation and 'NodeList=h100-9' in allocation and 'OverSubscribe=NO' in allocation and re.search(r'TresPerNode=gres:gpu(?::h100)?:8(?:\s|$)',allocation),'Actual running exclusive8GPU allocation required')
require(re.search(r'CPUs/Task=32(?:\s|$)',allocation),'Requested CPU protocol changed')
run=ROOT/'runs'/f'tree-native-numeric-{job}';run.mkdir();(run/'allocation.txt').write_text(allocation)
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
os.environ.update(NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NCCL_NVLS_ENABLE='0',NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',PYTHONDONTWRITEBYTECODE='1',NUMERIC_FIXTURE_OUTPUT=str(run/'gpu-records'),SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'audit'),SEARCH_R1_TREE_RANK_AUDIT_DIR=str(run/'tree-audit'),SEARCH_R1_BASELINE_COMMIT=manifest['code_commit'],SEARCH_R1_BASELINE_RUN_ID=run.name,PYTHONPATH=str(p/'repo')+':'+str(p)+':'+str(ROOT/'tools'))
result=dict(status='FAILED',job_id=job,code_commit=manifest['code_commit'],manifest_sha256=sha(p/'manifest.json'),scope='Native synthetic8rank numerical and storage fixtures only; no3B training',initial_gpu=gpu_sample());result.update(allocated_cpus=int(re.search(r'NumCPUs=(\d+)',allocation).group(1)),requested_cpus=32,allocated_gpus=8);start=time.time()
def stop(sig,frame):raise RuntimeError('Own native fixture interrupted')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard_identity.py'),'--output',str(run/'resources.jsonl')],run/'guard.log')
 deadline=time.monotonic()+30
 while not (run/'resources.ready').exists():
  require(guard.poll() is None and time.monotonic()<deadline,'Resource guard startup failed');time.sleep(1)
 gpu=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=26382',str(p/'numerics-gpu-fixture.py')],run/'gpu.log')
 deadline=time.monotonic()+600
 while gpu.poll() is None:
  require(guard.poll() is None and time.monotonic()<deadline,'Guard died or fixture timeout');time.sleep(1)
 require(gpu.returncode==0,'Native GPU fixture failed')
 require(guard.poll() is None,'Guard died before final fixture check')
 samples=[json.loads(line) for line in (run/'resources.jsonl').read_text().splitlines()]
 require(samples and all(not x.get('event') for x in samples) and time.time()-samples[-1]['epoch']<=15,'Guard samples missing/stale or STOP present')
 result.update(guard_samples=len(samples),guard_healthy_at_fixture_end=True,guard_last_sample=samples[-1])
 records=[json.loads((run/f'gpu-records/rank-{i}.json').read_text()) for i in range(8)]
 require(all(r['rank']==i and r['world']==8 and r['status']=='PASS' and len(r['records'])==20 and len(r['diagnostics_records'])==2 for i,r in enumerate(records)),'Incomplete actual8rank records')
 result.update(fixtures_passed=True,records=records,gpu_exit_code=gpu.returncode)
except BaseException as error:
 result['error']=repr(error);raise
finally:
 signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
 try:
  owned.cleanup()
  result.update(cleanup_residual=getattr(owned,'cleanup_residual',{}),final_gpu=gpu_sample())
  baseline={1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446}
  final={int(line.split(',')[1].strip()) for line in result['final_gpu']['processes']}
  require(baseline<=final,'Original8 background processes not preserved')
  require(final==baseline,'Unexpected remaining GPU processes; investigate')
  if result.get('fixtures_passed'):result['status']='NATIVE_FIXTURES_COMPLETE_PENDING_REVIEW'
 except BaseException as error:
  result.update(status='FAILED',cleanup_error=repr(error));raise
 finally:
  result.update(start_epoch=start,end_epoch=time.time(),wall_seconds=time.time()-start)
  (run/'result.json').write_text(json.dumps(result,indent=2))
print('NATIVE_TREE_FIXTURES_COMPLETE',flush=True)
