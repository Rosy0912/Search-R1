"""Two fresh single-UUID diagnostic workers; no training or numeric approval."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
def require(condition,message):
    if not condition:raise RuntimeError(message)
require(socket.gethostname().split('.')[0]=='h100-11','Wrong node')
require(os.environ.get('AGENT_ROOT')==str(ROOT),'Wrong own root')
job=os.environ['SLURM_JOB_ID']
require(int(os.environ['SLURM_CPUS_PER_TASK'])>=32 and int(os.environ['SLURM_MEM_PER_NODE'])>=96*1024,'Need allocated diagnostic CPU/RAM')
require(os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1','Need outer eight GPU leases')
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);require(len(leases)==8,'Need8 leases')
require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()=='438af3529b40663aae7b8293080e09fda57f6a87','Source drift')
spec=importlib.util.spec_from_file_location('topd_owned_diagnostic',ROOT/'tools/topd-short-run.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
require(launcher.sha(ROOT/'tools/topd-short-run.py')=='9a5d02514db879e7723cddf39a809504cab2b15de1ef5c3fa55e181a1a70b62f','Cleanup source drift')
scope,relative=launcher.cgroup_scope(job);owned=launcher.OwnedProcesses(scope,relative)
directory=ROOT/'logs'/f'parity-diagnostic-{job}';directory.mkdir(exist_ok=False)
manifest=json.loads((ROOT/'logs/afunix-scorer-hardware-2969/teacher-ipc/manifest.json').read_text())
baseline=launcher.gpu_sample()
result=dict(status='FAILED_DIAGNOSTIC_NO_TRAINING',job=job,scope='Original2969 failed input; no teacher/RL update/numeric approval',baseline=baseline,workers=[])
def interrupted(signum,frame):raise RuntimeError('Diagnostic interrupted '+str(signum))
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
    for rank in (4,0):
        uuid=next(service['uuid'] for service in manifest['services'] if service['rank']==rank)
        lease=next(item for item in leases if item['uuid']==uuid)
        env=launcher.local_runtime_env(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=uuid,SEARCH_R1_GPU_PREFLIGHT='1',SEARCH_R1_GPU_LEASES=json.dumps([lease]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        output=directory/f'rank{rank}.json'
        child=owned.spawn([sys.executable,str(ROOT/'tools/diagnose_topd_fp32_parity.py'),'--repo',str(ROOT/'repo'),'--probe-dir',str(ROOT/'logs/afunix-scorer-hardware-2969'),'--output',str(output),'--rank',str(rank),'--decompose'],directory/f'rank{rank}.log',env,pass_fds=(int(lease['fd']),))
        code=child.wait(timeout=900)
        result['workers'].append(dict(rank=rank,pid=child.pid,uuid=uuid,returncode=code,output=str(output)))
        require(code==0,'Diagnostic worker failed; inspect own raw log')
    result['status']='DIAGNOSTIC_COMPLETED_NO_TRAINING_NO_NUMERIC_APPROVAL'
except BaseException as error:
    result['error']=repr(error);raise
finally:
    try:owned.cleanup()
    except Exception as error:
        result.update(status='FAILED_DIAGNOSTIC_NO_TRAINING',cleanup_error=repr(error))
    result['cleanup_record']=getattr(owned,'cleanup_record',None)
    remaining=sorted(launcher.cgroup_pids(scope)-owned.baseline)
    result['own_children_remaining']=remaining
    result['final_gpu']=launcher.gpu_sample()
    identity=lambda lines:{tuple(v.strip() for v in line.split(',')[:2]) for line in lines if ',' in line}
    result['baseline_preserved']=identity(baseline['processes'])<=identity(result['final_gpu']['processes'])
    if remaining or not result['baseline_preserved']:result['status']='FAILED_DIAGNOSTIC_NO_TRAINING'
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
require(result['status']=='DIAGNOSTIC_COMPLETED_NO_TRAINING_NO_NUMERIC_APPROVAL','Diagnostic failed')
