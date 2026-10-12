"""Owned allocation for fresh Instruct source validation; no policy300 training."""
import fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from owned_runtime import ROOT,require,sha,cgroup_scope,OwnedProcesses,gpu_sample
binding=Path(sys.argv[1]);value=json.loads(binding.read_text())
require(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==value['code_commit'],'Source commit mismatch')
for item in value['files']:require(sha(item['path'])==item['sha256'],'Native source/helper changed')
require(socket.gethostname().split('.')[0]=='h100-8','Wrong node')
job=os.environ['SLURM_JOB_ID']
require(os.environ['SLURM_JOB_NODELIST']=='h100-8' and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8,'Not full8 allocation')
from search_r1.baselines.preflight import validate_inherited_gpu_leases,gpu_snapshot,check_selected_gpus
validate_inherited_gpu_leases(8)
lock=(ROOT/'gpu-job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
os.environ.update(NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NCCL_NVLS_ENABLE='0')
run=ROOT/'runs'/f'gigpo-e1-native-job{job}';run.mkdir()
os.environ['CPATH']='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10'
os.environ['NUMERIC_FIXTURE_OUTPUT']=str(run/'gpu-records')
for key in list(os.environ):
    if key.startswith('SEARCH_R1_BASELINE_AUDIT'):os.environ.pop(key,None)
scope,relative=cgroup_scope(job);owned=OwnedProcesses(scope,relative)
inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
(ROOT/'tools/expected-gpu-uuids.json').write_text(json.dumps([g['uuid'] for g in selected]))
import torch,transformers,search_r1,inspect
import search_r1.baselines.recovery as actual_recovery
require(inspect.getfile(actual_recovery)==str(ROOT/'repo/search_r1/baselines/recovery.py'),'Native imported wrong source')
result=dict(runtime_identity=dict(executable=sys.executable,prefix=sys.prefix,python=sys.version,torch=torch.__version__,transformers=transformers.__version__,recovery_file=inspect.getfile(actual_recovery)),status='FAILED',job=job,cgroup=str(scope),code_commit=value['code_commit'],binding_sha256=sha(binding),initial_gpu=gpu_sample())
def interrupted(sig,frame):raise RuntimeError('Own native fixture interrupted')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
    guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard.py'),'--output',str(run/'guard-resources.jsonl')],run/'guard.log')
    deadline=time.monotonic()+30
    while not (run/'guard-resources.ready').exists():
        require(guard.poll() is None and time.monotonic()<deadline,'Resource guard failed');time.sleep(1)
    cpu=owned.spawn([sys.executable,'-m','pytest','tests/baselines','tests/test_recovery.py','tests/reproduction/test_final_checkpoint.py','-q','-p','no:cacheprovider'],run/'cpu.log',env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    while cpu.poll() is None:require(guard.poll() is None,'Own guard failed');time.sleep(1)
    require(cpu.returncode==0,'Native CPU regression failed')
    gpu=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_port=26381',str(ROOT/'tools/native_gpu_v2.py')],run/'gpu.log')
    while gpu.poll() is None:require(guard.poll() is None,'Own guard failed');time.sleep(1)
    require(gpu.returncode==0,'Eight-rank HF FSDP GPU fixture failed')
    records=[json.loads((run/f'gpu-records/rank-{i}.json').read_text()) for i in range(8)]
    require(all(r['status']=='PASS' and r['rank']==i and len(r['records'])==5 for i,r in enumerate(records)),'Missing real rank records')
    result.update(fixtures_passed=True,cpu_exit=cpu.returncode,gpu_exit=gpu.returncode,records=records)
except BaseException as error:result['error']=repr(error);raise
finally:
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    try:
        owned.cleanup()
        if result.get('fixtures_passed'):result['status']='PASS'
    finally:
        result.update(cleanup_residual_states=getattr(owned,'cleanup_residual',{}),final_gpu=gpu_sample())
        (run/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
