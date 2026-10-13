"""Explicit own systemd training identity and held physical UUID leases.

Not a Slurm allocation. Numeric scheduler identity is never fabricated. This
opt-in is restricted to a manifest/release-bound shared fresh200 candidate.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from production_contract import ROOT, SOURCE, require, read_bound

PROTOCOL='training_memory_headroom_shared_v1'
MODE='TREE_SHARED_TRAINING_MODE'

def enabled(environment=None):
    return (os.environ if environment is None else environment).get(MODE)=='1'

def unit_name():return 'sst-tree-nested-branch-fresh200-'+SOURCE[:8]+'.service'
def job_id():return 'not-slurm;systemd:'+SOURCE

def package(environment=None):
    environment=os.environ if environment is None else environment
    tools=Path(environment['TREE_INSTRUCT_TOOLS'])
    require(tools.resolve().is_relative_to((ROOT/'instruct300/candidates').resolve()),'Shared candidate outside own root')
    value,_,_=read_bound(tools/'manifest.json',environment['TREE_INSTRUCT_PACKAGE_SHA256'])
    spec=value['dispatch']
    require(value['code_commit']==SOURCE and spec['kind']=='shared_systemd'
            and spec['resource_protocol']==PROTOCOL and spec['nodes']==['h100-10']
            and spec['gpus']==8 and spec['exclusive'] is False
            and spec['slurm_reserved_gpus']==0 and spec['physical_gpus']==8,
            'Explicit reviewed shared package required')
    require(spec['initial_required_free_gib']==68 and spec['runtime_required_headroom_gib']==8
            and spec['own_memory_cap_gib']==72 and spec['measured_training_budget_gib']==60,
            'Reviewed unchanged-training memory admission differs')
    require(spec['expected_gpu_uuids']==json.loads((tools/'expected-gpu-uuids-retained18.json').read_text())
            and len(set(spec['expected_gpu_uuids']))==8,'Exact package-bound physical UUID8 required')
    return tools,value

def verify_cgroup(environment=None,hostname=None,cgroup_text=None):
    environment=os.environ if environment is None else environment
    require(enabled(environment),'Shared identity requires explicit opt-in')
    require((socket.gethostname() if hostname is None else hostname).split('.')[0]=='h100-10','Shared training requires selected asset-local18')
    _,value=package(environment)
    require(environment.get('TREE_SHARED_SYSTEMD_UNIT')==unit_name()
            and environment.get('SLURM_JOB_ID')==job_id()
            and not any(environment.get(k) for k in ('SLURM_JOB_NODELIST','SLURM_JOB_GPUS','SLURM_STEP_ID','SLURM_CLUSTER_NAME')),
            'Shared identity must remain opaque, no scheduler reservation impersonation')
    text=Path('/proc/self/cgroup').read_text() if cgroup_text is None else cgroup_text
    found=[]
    for line in text.splitlines():
        _,controllers,relative=line.split(':',2);parts=relative.split('/')
        require(not any(part.startswith('job_') for part in parts),'Shared child cannot inhabit any Slurm job')
        if unit_name() in parts and (controllers=='' or 'name=systemd' in controllers.split(',') or 'freezer' in controllers.split(',')):
            base='/sys/fs/cgroup' if controllers=='' else '/sys/fs/cgroup/systemd' if 'name=systemd' in controllers.split(',') else '/sys/fs/cgroup/freezer'
            scope=Path(base)/relative.lstrip('/')
            if (scope/'cgroup.procs').is_file():found.append((scope,relative))
    require(found and len({relative for _,relative in found})==1,'Actual own shared unit cgroup missing/ambiguous')
    scope,relative=found[0]
    require(os.getpid() in {int(x) for x in (scope/'cgroup.procs').read_text().split()},'Current PID absent from exact own cgroup')
    return dict(job_id=job_id(),unit=unit_name(),scope=str(scope),relative=relative,current_cgroup=text,
                allocation_kind='user_authorized_shared_systemd',resource_protocol=PROTOCOL,
                physical_gpus=8,slurm_reserved_gpus=0)

def verify_runtime(environment=None,hostname=None,run=subprocess.run):
    environment=os.environ if environment is None else environment
    identity=verify_cgroup(environment,hostname)
    result=run(['systemctl','show',unit_name(),'--property=ActiveState,SubState,MainPID,ControlGroup,CPUQuotaPerSecUSec,MemoryMax,Restart'],
               text=True,capture_output=True,timeout=15)
    require(result.returncode==0 and result.stderr=='','Own systemd identity query failed')
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    require(fields.get('ActiveState')=='active' and fields.get('SubState')=='running'
            and fields.get('MainPID')==str(os.getpid())
            and fields.get('ControlGroup')==identity['relative']
            and fields.get('MemoryMax')==str(256*1024**3)
            and fields.get('CPUQuotaPerSecUSec')=='32s' and fields.get('Restart')=='no', 'Own unit is not actual active training scope')
    return dict(identity,cluster_name='wuhan-h100',node_list='h100-10',gpus=8,requested_cpus=32,
                allocated_cpus=0,memory_gib=256,wall_hours=168,no_requeue=True,
                allocation_stdout=result.stdout,slurm_reserved_cpus=0,systemd_cpu_quota_cores=32)

def selected_hardware():
    import pynvml as nv
    nv.nvmlInit();_,value=package();expected=value['dispatch']['expected_gpu_uuids']
    selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    require(len(selected)==8 and len(set(selected))==8,'Eight distinct physical selections required')
    mapping=[]
    for index in range(nv.nvmlDeviceGetCount()):
        handle=nv.nvmlDeviceGetHandleByIndex(index);uid=nv.nvmlDeviceGetUUID(handle)
        if isinstance(uid,bytes):uid=uid.decode()
        if str(index) in selected or uid in selected:mapping.append((str(index),uid,handle))
    require(len(mapping)==8 and {uid for _,uid,_ in mapping}==set(expected),'Numeric visible mapping differs from fixed UUID8')
    return nv,mapping

def validate_leases(expected_count=None):
    identity=verify_cgroup();require(expected_count in (None,8),'Physical world size changed')
    nv,mapping=selected_hardware();records=json.loads(os.environ.get('SEARCH_R1_GPU_LEASES','[]'))
    require(os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1' and len(records)==8
            and {x['uuid'] for x in records}=={uid for _,uid,_ in mapping},'Exact shared held lease coverage missing')
    for record in records:
        fd=int(record['fd']);path=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(record['uuid']+'.lock')
        held=os.fstat(fd);named=path.lstat()
        require(stat.S_ISREG(named.st_mode) and (held.st_dev,held.st_ino)==(named.st_dev,named.st_ino),'Own held OFD differs from UUID lock file')
        probe=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
        try:
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise ValueError('Supposed held UUID lease was independently acquired')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(probe)
    guard=os.environ.get('TREE_DIRECT_ALLOCATION_IDENTITY_PATH')
    if guard:
        allocation,_,_=read_bound(guard,os.environ['TREE_DIRECT_ALLOCATION_IDENTITY_SHA256'])
        require(allocation['job_id']==identity['job_id'] and allocation['unit']==unit_name()
                and allocation['physical_gpus']==8 and allocation['slurm_reserved_gpus']==0
                and allocation['resource_protocol']==PROTOCOL,'Shared recorded allocation identity differs')
        run=ROOT/'runs'/('tree-nested-branch-fresh200-'+job_id())
        last=last_guard(run)
        require(last['job_id']==job_id() and not last.get('event') and time.time()-last['epoch']<10
                and {x['uuid'] for x in last['gpus']}=={uid for _,uid,_ in mapping},'Shared memory guard stale/unhealthy')
    return dict(job_id=job_id(),uuid=[uid for _,uid,_ in mapping],held_ofd_identity_checked=True,
                physical_gpus=8,slurm_reserved_gpus=0,resource_protocol=PROTOCOL,whole_card_idle_check=False)

def last_guard(run):
    with (Path(run)/'resource-samples.jsonl').open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-131072));tail=stream.read()
    lines=tail.splitlines()
    if tail and not tail.endswith(b'\n'):lines=lines[:-1]
    require(lines,'No complete shared training guard sample');return json.loads(lines[-1])

def install_preflight():
    from search_r1.baselines import preflight
    preflight.validate_inherited_gpu_leases=validate_leases

def guarded_preflight_main():
    verify_cgroup();install_preflight()
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-inherited',action='store_true');parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    require(command,'Shared preflight requires exact child command')
    validate_leases(8)
    for record in json.loads(os.environ['SEARCH_R1_GPU_LEASES']):os.set_inheritable(int(record['fd']),True)
    os.execvp(command[0],command)

def observe_host_admission():
    """Actual host memory/capacity; load is evidence only, no idle-node gate."""
    meminfo=Path('/proc/meminfo').read_text()
    available=int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith('MemAvailable:')))*1024
    capacity=len(os.sched_getaffinity(0));load=list(os.getloadavg())
    require(available>=320*1024**3,'Host shared admission needs256GiB own ceiling+64GiB headroom (320GiB MemAvailable)')
    require(capacity>=32,'Actual host CPU affinity capacity below unchanged32 requested cores')
    return dict(observed_epoch=time.time(),mem_available_bytes=available,required_mem_available_gib=320,
                own_memory_ceiling_gib=256,system_headroom_gib=64,cpu_affinity_capacity=capacity,
                requested_cpu_quota_cores=32,load_average_1_5_15=load,
                load_warning=load[0]>capacity-32,load_is_blocking=False,
                scope='Actual read-only host capacity before own load; MemoryMax is a ceiling, not a reservation')

def establish_leases(opened_records=None):
    records=[] if opened_records is None else opened_records
    require(not records,'Lease establishment requires empty own opened-fd ledger')
    verify_cgroup();nv,mapping=selected_hardware();lockdir=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'
    lockdir.mkdir(mode=0o700,exist_ok=True)
    for _,uid,handle in sorted(mapping,key=lambda x:x[1]):
        fd=os.open(lockdir/(uid+'.lock'),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        held=os.fstat(fd);record=dict(uuid=uid,fd=fd,opened_by_current_pid=os.getpid(),st_dev=held.st_dev,st_ino=held.st_ino,lease_acquired=False)
        records.append(record)
        require(stat.S_ISREG(held.st_mode),'GPU UUID lock must be ordinary')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);record['lease_acquired']=True;os.set_inheritable(fd,True)
    for _,_,handle in mapping:require(nv.nvmlDeviceGetMemoryInfo(handle).free>=68*1024**3,'Need actual68GiB free per physical GPU at startup')
    os.environ.update(SEARCH_R1_GPU_LEASES=json.dumps([dict(uuid=x['uuid'],fd=x['fd']) for x in records]),SEARCH_R1_GPU_PREFLIGHT='1',
                      CUDA_VISIBLE_DEVICES=','.join(index for index,_,_ in mapping))
    validate_leases(8);return records

def submitted_lineage(value,job,manifest_sha,review_sha,intent_path):
    require(value['status']=='SUBMITTED' and value['dispatch_kind']=='shared_systemd'
            and value['job_id']==job==job_id() and value['unit']==unit_name()
            and value['physical_gpus']==8 and value['slurm_reserved_gpus']==0
            and value['resource_protocol']==PROTOCOL and value['systemd_start_returncode']==0
            and value['manifest_sha256']==manifest_sha and value['review_sha256']==review_sha
            and value['dispatch_intent_path']==str(intent_path)
            and value['run_id']=='tree-nested-branch-fresh200-'+job,
            'Exact opaque shared once-lineage missing')
    from render_shared_dispatch import render,payload
    import base64,hashlib
    tools=Path(os.environ['TREE_INSTRUCT_TOOLS'])
    expected=payload(tools,manifest_sha,os.environ['TREE_PRODUCTION_RELEASE_PATH'],review_sha)
    require(value['code_commit']==SOURCE and value['node_list']=='h100-10'
            and value['submit_argv']==render(tools,manifest_sha,os.environ['TREE_PRODUCTION_RELEASE_PATH'],review_sha)
            and value['submitted_input_sha256']==hashlib.sha256(expected).hexdigest()
            and value['dispatch_script_sha256']==hashlib.sha256(expected).hexdigest()
            and value['submitted_input_base64']==base64.b64encode(expected).decode(),
            'Exact shared journal argv/payload binding differs')
    return value
