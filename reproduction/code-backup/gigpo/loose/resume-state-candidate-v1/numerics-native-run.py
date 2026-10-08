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
run=ROOT/'runs'/f'resume-state-fixtures-job{job}';run.mkdir()
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
    external=json.loads((directory/'external-deps.json').read_text())
    for name,expected in external['files'].items():
        require(sha(ROOT/'tools/test-lib'/name)==expected,'Isolated test dependency changed')
    import pytest
    require(pytest.__version__==external['pytest'] and pytest.__file__==external['file'],'Wrong test dependency import')
    result['test_dependency']=dict(version=pytest.__version__,file=pytest.__file__,files=len(external['files']),manifest_sha256=sha(directory/'external-deps.json'))
    guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard.py'),'--output',str(run/'guard-resources.jsonl')],run/'guard.log')
    deadline=time.monotonic()+30
    while not (run/'guard-resources.ready').exists():
        require(guard.poll() is None and time.monotonic()<deadline,'Guard startup failed');time.sleep(1)
    cpu_env=dict(os.environ)
    cpu_env.pop('SEARCH_R1_BASELINE_AUDIT_DIR',None)
    cpu_env['CUDA_VISIBLE_DEVICES']=''
    result['cpu_test_scope']='CPU suite controls its own audit fixtures; training audit directory is not injected; GPU fixture retains diagnostics'
    cpu_bootstrap="import sys;sys.path.insert(0,sys.argv.pop(1));import search_r1,pytest;from search_r1.baselines import gradient_norm;print('ACTUAL_CANDIDATE_IMPORT',search_r1.__file__,gradient_norm.__file__,flush=True);assert gradient_norm.__file__.startswith(sys.path[0]);raise SystemExit(pytest.main(sys.argv[1:]))"
    cpu=owned.spawn([sys.executable,'-c',cpu_bootstrap,str(directory/'repo'),str(directory/'repo/tests/test_recovery.py'),str(directory/'repo/tests/reproduction/test_final_checkpoint.py'),'-q','-p','no:cacheprovider'],run/'cpu.log',env=cpu_env)
    while cpu.poll() is None:
        require(guard.poll() is None,'Guard died');time.sleep(1)
    require(cpu.returncode==0,'Native CPU tests failed')
    gpu=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_port=26381',str(directory/'numerics-gpu-fixture.py')],run/'gpu.log')
    while gpu.poll() is None:
        require(guard.poll() is None,'Guard died');time.sleep(1)
    require(gpu.returncode==0,'Native FSDP fixture failed')
    records=[json.loads((run/f'gpu-records/rank-{i}.json').read_text()) for i in range(8)]
    require(all(r['status']=='PASS' and r['rank']==i and len(r['records'])==4 for i,r in enumerate(records)),'Incomplete GPU records')
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
