"""Lease only HKUST5 GPUs 0-3 and run one isolated cost diagnostic."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['repaired_b','old_opsd'],required=True)
    p.add_argument('--native-source',required=True);p.add_argument('--tools-source',required=True)
    a=p.parse_args()
    assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
    root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
    native=Path(a.native_source);tools=Path(a.tools_source)
    native.relative_to(root);tools.relative_to(root)
    assert (native/'SOURCE_COMMIT').is_file() and (tools/'SOURCE_COMMIT').is_file()
    expected=['GPU-15061c70-49d6-3014-da87-915b370c1e89','GPU-0c0aeed1-c7b9-c9ad-4b1f-7396b6c19f4c',
              'GPU-d2609c94-e30a-a106-e1cc-6d4ae3e044c2','GPU-d6d81990-8737-f64c-aaef-d0ceabb5123a']
    # The user assigned devices 0-3. NVML queries can themselves block in this
    # node's driver; use exact device nodes and exclusive leases for this pair.
    # UUIDs are the previously observed fixed-host mapping, not live telemetry.
    records=[]
    for i,gpu in enumerate(expected):
        device=os.stat('/dev/nvidia'+str(i))
        assert os.major(device.st_rdev)==195 and os.minor(device.st_rdev)==i
        records.append(dict(index=i,uuid=gpu,uuid_scope='previous fixed-host inventory',
            admission='user-assigned device minor and exclusive lease; no NVML query'))
    leases=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}';leases.mkdir(exist_ok=True,mode=0o700)
    handles=[]
    for gpu in sorted(expected):
        fd=os.open(leases/(gpu+'.lock'),os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(fd)
    run=uuid.uuid4().hex;out=root/'cost80'/(a.case+'-'+run)
    out.parent.mkdir(exist_ok=True)
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('TREE_RECOVERY_','RECURSIVE_B_CONTINUATION')) or key in ('RECURSIVE_BRAN_EXPERIMENT_LOCK','RECURSIVE_BRAN_STRICT_GPU_GATE'):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',RAY_ADDRESS='local',
        PYTHONPATH=str(native)+':'+str(tools/'release/recursive_bran_b'),
        SEARCH_R1_BASELINE_RUN_ID=run,SEARCH_R1_BASELINE_COMMIT=(native/'SOURCE_COMMIT').read_text().strip(),
        SEARCH_R1_BASELINE_DATA_IDENTITY='7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43',
        SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1',
        SEARCH_R1_BASELINE_OUTPUT=str(out),SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),
        SEARCH_R1_TREE_RANK_AUDIT_DIR=str(out/'rank-audit'),
        SLURM_JOB_ID='cost80-'+run,SLURM_JOB_NODELIST='xie-lab-4',
        TREE_COMMON_DATA=str(root/'common-data'),TREE_INSTRUCT_MODEL='/home/shaoshitong/search-e1-config-study-20261004/assets/model',
        TMPDIR=str(root/'tmp'),RAY_TMPDIR='/tmp/rb80-'+run[:16],
        HF_HOME=str(root/'cache/hf'),XDG_CACHE_HOME=str(root/'cache'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='offline',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        VLLM_ATTENTION_BACKEND='XFORMERS',PYTHONDONTWRITEBYTECODE='1')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    entry=tools/'release/recursive_bran_b/profile_cost80.py'
    record=dict(case=a.case,run=run,output=str(out),native_source=str(native),
        native_commit=env['SEARCH_R1_BASELINE_COMMIT'],tools_commit=(tools/'SOURCE_COMMIT').read_text().strip(),
        profiler_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),gpu_inventory=records,
        diagnostic_only=True,questions=16,training_rows=80)
    target=root/'receipts'/('cost80-launch-'+run+'.json');target.write_text(json.dumps(record,sort_keys=True))
    print('COST80_LAUNCH='+json.dumps(record),flush=True)
    command=[sys.executable,'-B',str(entry),'--case',a.case,'--config',
             str(tools/'release/recursive_bran_b/recovery_records/afb2d350/resolved.yaml'),'--output',str(out)]
    try:
        rc=subprocess.call(command,env=env,pass_fds=tuple(handles),cwd=native)
    finally:
        for fd in handles:os.close(fd)
    print('COST80_EXIT='+json.dumps(dict(run=run,case=a.case,exit_code=rc)),flush=True)
    raise SystemExit(rc)

if __name__=='__main__':main()
