"""Post-load wrapper checks owned GPU identity and foreign occupancy only.

The original before-load preflight remains unchanged. This opt-in wrapper is
only installed in this direct trainer after own retriever/model allocation.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import time

from production_contract import DIRECT_MODE,ROOT,read_bound,require,direct_authority
from slurm_identity import verify_runtime_cgroup

MODE_ENV='TREE_DIRECT_OWNED_MODELS_LOADED'

def validate_owned_gpu_leases(expected_count=None):
    identity=verify_runtime_cgroup()
    require(os.environ.get(MODE_ENV)=='1','Identity-only lease validation requires explicit own-models-loaded wrapper mode')
    tools=Path(os.environ['TREE_INSTRUCT_TOOLS'])
    package,_,_=read_bound(tools/'manifest.json',os.environ['TREE_INSTRUCT_PACKAGE_SHA256'])
    require(package['execution_mode']==DIRECT_MODE and package['packet_revision']==3,
            'Identity-only wrapper is restricted to this direct V3 package')
    direct_authority(package)
    allocation,_,_=read_bound(os.environ['TREE_DIRECT_ALLOCATION_IDENTITY_PATH'],
                              os.environ['TREE_DIRECT_ALLOCATION_IDENTITY_SHA256'])
    job=os.environ['SLURM_JOB_ID'];run=ROOT/'runs'/('tree-fresh200-train-'+job)
    require(str(allocation['job_id'])==identity['job_id']==job
            and allocation['cluster_name']=='wuhan-h100' and allocation['node_list']=='h100-9'
            and allocation['gpus']==8 and allocation['requested_cpus']==32 and allocation['allocated_cpus']==192
            and allocation['memory_gib']==256 and allocation['no_requeue'] is True
            and allocation['wall_hours']==168 and allocation['run_id']==run.name
            and allocation['manifest_sha256']==os.environ['TREE_INSTRUCT_PACKAGE_SHA256'],
            'Post-load allocation identity differs from own pinned reservation')
    selected=[value.strip() for value in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if value.strip()]
    require(len(selected)==8 and len(set(selected))==8 and expected_count in (None,8),
            'Post-load lease requires exactly eight distinct selected GPUs')
    records=json.loads(os.environ.get('SEARCH_R1_GPU_LEASES','[]'))
    require(os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1' and len(records)==8
            and len({record['uuid'] for record in records})==8,
            'Held launcher OFD/UUID lease coverage incomplete')
    # The guarded snapshot resolves hardware identity only. Own memory and SM
    # occupancy are expected after allocation and are never used as idle gates.
    from search_r1.baselines.preflight import gpu_snapshot
    inventory,_=gpu_snapshot()
    hardware=[]
    for device in selected:
        matches=[value for index,value in inventory.items() if device in (index,value['uuid'])]
        require(len(matches)==1,'Selected post-load GPU cannot be uniquely resolved')
        hardware.append(matches[0]['uuid'])
    require(len(set(hardware))==8 and set(hardware)=={record['uuid'] for record in records},
            'Held UUID leases differ from exact selected physical GPUs')
    for record in records:
        fd=int(record['fd']);path=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(record['uuid']+'.lock')
        held=os.fstat(fd);named=path.lstat()
        require(stat.S_ISREG(named.st_mode) and (held.st_dev,held.st_ino)==(named.st_dev,named.st_ino),
                'Held GPU lease FD differs from exact regular UUID lock file')
        probe=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
        try:
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise ValueError('Independent OFD acquired a supposed held GPU lease')
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise ValueError('Inherited OFD does not own its GPU lease') from error
        finally:os.close(probe)
    with (run/'resource-samples.jsonl').open('rb') as stream:
        stream.seek(0,os.SEEK_END);size=stream.tell();stream.seek(max(0,size-131072));tail=stream.read()
    rows=tail.splitlines()
    if tail and not tail.endswith(b'\n'):rows=rows[:-1]
    require(rows,'Post-load own foreign-occupancy guard has no complete record')
    sample=json.loads(rows[-1])
    require(sample.get('job_id')==job and not sample.get('event')
            and time.time()-sample['epoch']<15 and len(sample.get('gpus',[]))==8
            and {gpu['uuid'] for gpu in sample['gpus']}==set(hardware),
            'Post-load foreign-occupancy guard is stale/unhealthy or covers different GPUs')
    for gpu in sample['gpus']:
        memory=gpu['foreign_memory_fraction'];sm=gpu['foreign_sm_upper']
        require(type(memory) in (int,float) and math.isfinite(memory) and 0<=memory<=.15
                and type(sm) in (int,float) and math.isfinite(sm) and 0<=sm<=15,
                'Foreign GPU memory or SM exceeds mandatory15% guard')
    return dict(job_id=job,uuid=hardware,whole_card_idle_check=False,
                foreign_memory_and_sm_15_percent_checked=True,held_ofd_identity_checked=True)

def install_postload_lease_wrapper():
    if os.environ.get(MODE_ENV)!='1':return
    from search_r1.baselines import preflight
    preflight.validate_inherited_gpu_leases=validate_owned_gpu_leases
