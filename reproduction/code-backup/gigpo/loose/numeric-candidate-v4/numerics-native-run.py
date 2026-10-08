"""Owned, leased Slurm fixture runner; never launches a model training job."""
import fcntl,json,os,signal,socket,sys,time
from pathlib import Path
from owned_runtime import ROOT,require,cgroup_scope,OwnedProcesses,gpu_sample,sha

directory=Path(sys.argv[1])
require(socket.gethostname().split('.')[0]=='h100-8','Wrong node')
job=os.environ['SLURM_JOB_ID']
from search_r1.baselines.preflight import validate_inherited_gpu_leases,gpu_snapshot,check_selected_gpus
validate_inherited_gpu_leases(8)
lock=(ROOT/'gpu-job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
os.environ.update(NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NCCL_NVLS_ENABLE='0')
run=ROOT/'runs'/f'numeric-fixtures-job{job}';run.mkdir()
os.environ.update(PYTHONPATH=str(directory/'repo')+':'+str(ROOT/'tools/test-lib')+':'+str(ROOT/'tools'),NUMERIC_FIXTURE_OUTPUT=str(run/'gpu-records'),SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'audit'),SEARCH_R1_REQUIRE_REAL_DATAPROTO='1')
scope,relative=cgroup_scope(job);owned=OwnedProcesses(scope,relative)
inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
require(len(selected)==8,'Eight UUIDs required')
(ROOT/'tools/expected-gpu-uuids.json').write_text(json.dumps([g['uuid'] for g in selected]))
result=dict(status='FAILED',job=job,cgroup=str(scope),source_directory=str(directory),initial_gpu=gpu_sample(),file_hashes={})
def interrupted(sig,frame):
    raise RuntimeError(f'Owned numeric fixture interrupted by signal {sig}')
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
for p in directory.rglob('*'):
    if p.is_file():result['file_hashes'][str(p.relative_to(directory))]=sha(p)
try:
    guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard.py'),'--output',str(run/'guard-resources.jsonl')],run/'guard.log')
    deadline=time.monotonic()+30
    while not (run/'guard-resources.ready').exists():
        require(guard.poll() is None and time.monotonic()<deadline,'Guard startup failed');time.sleep(1)
    cpu=owned.spawn([sys.executable,'-m','pytest',str(directory/'repo/tests/reproduction/test_terminal_answer_protocol.py'),str(directory/'repo/tests/reproduction/test_reward_and_validation.py'),str(directory/'repo/tests/baselines'),'-q'],run/'cpu.log')
    while cpu.poll() is None:
        require(guard.poll() is None,'Guard died');time.sleep(1)
    require(cpu.returncode==0,'Native CPU tests failed')
    gpu=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_port=26381',str(directory/'numerics-gpu-fixture.py')],run/'gpu.log')
    while gpu.poll() is None:
        require(guard.poll() is None,'Guard died');time.sleep(1)
    require(gpu.returncode==0,'Native FSDP fixture failed')
    records=[json.loads((run/f'gpu-records/rank-{i}.json').read_text()) for i in range(8)]
    require(all(r['status']=='PASS' and r['rank']==i and len(r['records'])==16 for i,r in enumerate(records)),'Incomplete GPU records')
    result.update(fixtures_passed=True,cpu_exit=cpu.returncode,gpu_exit=gpu.returncode,records=records)
except BaseException as e:
    result.update(error=repr(e));raise
finally:
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    try:
        owned.cleanup()
        if result.get('fixtures_passed'):
            result['status']='PASS'
    except BaseException as e:
        result.update(status='FAILED',cleanup_error=repr(e))
        raise
    finally:
        result.update(cleanup_residual_states=getattr(owned,'cleanup_residual',{}),final_gpu=gpu_sample())
        (run/'result.json').write_text(json.dumps(result,indent=2))
print('NATIVE_NUMERIC_FIXTURE_PASS',flush=True)
