"""Candidate resource and identity adapter for a zero-update Tree probe.

No launch occurs in this module. The separately reviewed supervisor must own
leases, guard and cleanup. Source preparation is not permission to execute GPU.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PROTECTED = {1680782:1502352181,1680783:1502352181,1680784:1502352181,1680785:1502352181,
             1696443:1502398762,1696444:1502398762,1696445:1502398762,1696446:1502398762}
ACTIVE = None


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def local(path):
    path=Path(path);require(path.is_absolute(),'Absolute same-node path required')
    resolved=path.resolve();require(resolved.is_relative_to(ROOT),'Asset escaped h100-9 task root')
    return resolved


def bound(path, expected):
    path=local(path);payload=path.read_bytes();require(hashlib.sha256(payload).hexdigest()==expected,'Changed metadata '+str(path))
    return json.loads(payload)


def process(pid):
    require(type(pid) is int and pid>0,'Invalid process identity')
    fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return fields[0],int(fields[19])


def allocation(text, job):
    require(type(job) is str and job.isdigit(),'Missing Slurm job identity')
    for token in ['JobId='+job,'JobState=RUNNING','NodeList=h100-9','OverSubscribe=NO']:
        require(re.search(r'(?:^|\s)'+re.escape(token)+r'(?:\s|$)',text) is not None,'Missing actual allocation proof '+token)
    require(re.search(r'TresPerNode=gres:gpu(?::h100)?:8(?:\s|$)',text) is not None,'Actual eight GPU allocation required')
    require(re.search(r'CPUs/Task=32(?:\s|$)',text) is not None,'CPU32 request changed')
    match=re.search(r'NumCPUs=(\d+)',text);require(match is not None,'Actual allocated CPU count missing')
    return int(match.group(1))


def guard_record(record, job, uuids, now):
    require(type(record) is dict and record.get('job_id')==job and not record.get('event'),'Guard identity or STOP')
    epoch=record.get('epoch');require(type(epoch) in (int,float) and 0<=now-epoch<=15,'Guard missing, future or stale')
    gpus=record.get('gpus');require(type(gpus) is list and len(gpus)==8,'Incomplete guard GPU inventory')
    require({g.get('uuid') for g in gpus}==set(uuids),'Guard UUID inventory changed')
    for gpu in gpus:
        memory,sm=gpu.get('foreign_memory_fraction'),gpu.get('foreign_sm_upper')
        require(type(memory) in (int,float) and 0<=memory<=.15 and
                type(sm) in (int,float) and 0<=sm<=15,'Foreign activity exceeds dual15% or is unknown')
    return record


def check(phase):
    require(ACTIVE is not None and type(phase) is str and phase,'Adapter is not bound to current launch')
    state=ACTIVE;job=os.environ.get('SLURM_JOB_ID');require(job==state['job_id'],'Job identity changed')
    require(socket.gethostname().split('.')[0]=='h100-9','Wrong host')
    for pid,start in PROTECTED.items():
        mode,current=process(pid);require(mode!='Z' and current==start,'Original background process identity changed')
    for name in ['supervisor','guard']:
        entry=state[name];mode,start=process(entry['pid'])
        require(mode!='Z' and start==entry['start_ticks'],'Owned '+name+' exited or changed identity')
        cgroup=Path('/proc',str(entry['pid']),'cgroup').read_text()
        require(any('job_'+job in line.split(':',2)[-1].split('/') for line in cgroup.splitlines()),'Owned '+name+' left allocation')
    records=local(state['guard_path']).read_text().splitlines();require(records,'No live guard record')
    row=json.loads(records[-1]);guard_record(row,job,state['gpu_uuids'],time.time())
    for lease in state['leases']:
        fd=Path('/proc',str(state['supervisor']['pid']),'fd',str(lease['fd']))
        expected=Path('/tmp')/('search-r1-gpu-leases-'+str(os.getuid()))/(lease['uuid']+'.lock')
        a,b=fd.stat(),expected.stat();require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Supervisor lease FD changed')
        with expected.open('r') as stream:
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN);raise ValueError('GPU lease is no longer held')
    return dict(phase=phase,job_id=job,guard_epoch=row['epoch'],original_background_processes_preserved=True)


def prepare_and_validate_bundle(bundle):
    global ACTIVE
    from search_r1.baselines.probability_probe_contracts import (bind_sources,bind_exact_files,validate_selection,
        validate_fixed_config,strict_json_load,files_aggregate)
    require(ACTIVE is None,'Adapter cannot rebind launch state')
    require(bundle['status']=='GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED','GPU release receipt required')
    launch_path=os.environ.get('TREE_PROBE_LAUNCH_PATH');launch_sha=os.environ.get('TREE_PROBE_LAUNCH_SHA256')
    launch=bound(launch_path,launch_sha)
    require(launch['job_id']==os.environ.get('SLURM_JOB_ID'),'Launch job changed')
    require(launch['bundle_sha256']==os.environ.get('TREE_PROBE_BUNDLE_SHA256'),'Bundle SHA differs from launch')
    require(sha(os.environ['TREE_PROBE_BUNDLE_PATH'])==launch['bundle_sha256'],'Runtime bundle bytes changed')
    require(sha(__file__)==bundle['reviewed_resource_adapter']['sha256'],'Adapter source changed')
    allocation(launch['allocation_text'],launch['job_id'])
    require(len(launch['gpu_uuids'])==8 and len(set(launch['gpu_uuids']))==8,'Launch UUID inventory invalid')
    require(len(launch['leases'])==8 and {e['uuid'] for e in launch['leases']}==set(launch['gpu_uuids']),'Lease UUIDs differ')
    release=bound(bundle['release']['path'],bundle['release']['sha256'])
    require(release['status']=='ZERO_UPDATE_GPU_PROBE_SOURCE_AND_DISPATCH_INDEPENDENTLY_RELEASED','Independent GPU release missing')
    require(release['selection_sha256']==bundle['selection_file']['sha256'] and
            release['runner_commit']==bundle['runner_commit'] and release['actual_optimizer_writes']==0,
            'Release binds another source/selection')
    selected=bound(bundle['selection_file']['path'],bundle['selection_file']['sha256'])
    require(selected==bundle['selection'] and selected['inventory']==bundle['inventory'],'Runtime selection or inventory changed')
    validate_selection(selected,bundle['identity'],bundle['inventory'],allow_frozen=True)
    bind_sources(bundle['identity']['source_files']);bind_sources(bundle['probe_source_files'])
    for cpu in selected['cpu_results'].values():bound(cpu['path'],cpu['sha256'])
    for asset in bundle['inventory'].values():require(sha(local(asset['path']))==asset['sha256'],'Original tensor asset changed')
    for entry in selected['monitor_companions']:require(sha(local(entry['path']))==entry['sha256'],'Monitor companion changed')
    from omegaconf import OmegaConf
    config=OmegaConf.to_container(OmegaConf.load(bundle['config_path']),resolve=True)
    require(sha(local(bundle['config_path']))==bundle['identity']['config_sha256'],'Original config changed')
    validate_fixed_config(config)
    require(bundle['hf_identity']['sha256']==bundle['identity']['parent_hf_sha256'],'HealthyHF identity changed')
    bind_exact_files(bundle['hf_path'],bundle['hf_identity']['files'],bundle['hf_identity']['sha256'])
    parent=bound(bundle['parent_state_manifest']['path'],bundle['identity']['parent_state_manifest_sha256'])
    require(parent['identity']['completed_step']==110,'Healthy parent completed step changed')
    for entry in bundle['installed_sources']:require(sha(Path(entry['path']))==entry['sha256'],'Installed implementation changed')
    require(bundle['probe_mode']=='eval_like_original_compute_log_prob','Probe mode changed')
    ACTIVE=launch
    check('adapter_bound_after_complete_bundle_validation')
    return True
