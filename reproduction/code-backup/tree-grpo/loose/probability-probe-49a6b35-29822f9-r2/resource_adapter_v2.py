"""Candidate resource and identity adapter for a zero-update Tree probe.

No launch occurs in this module. The separately reviewed supervisor must own
leases, guard and cleanup. Source preparation is not permission to execute GPU.
"""
import fcntl
import hashlib
import json
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import re
import socket
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PROTECTED = {1680782:1502352181,1680783:1502352181,1680784:1502352181,1680785:1502352181,
             1696443:1502398762,1696444:1502398762,1696445:1502398762,1696446:1502398762}
ACTIVE = None
INSTALLED_MODULES = {
    'torch.distributed.fsdp.fully_sharded_data_parallel',
    'torch.distributed.fsdp._runtime_utils','torch.distributed.fsdp._flat_param',
    'torch.distributed.device_mesh','transformers.models.qwen2.modeling_qwen2',
    'transformers.modeling_flash_attention_utils','flash_attn.bert_padding',
    'flash_attn.ops.triton.cross_entropy'}
INSTALLED_VERSIONS = {'torch':'2.4.0','transformers':'4.46.3','flash-attn':'2.7.4.post1',
                      'safetensors':'0.8.0','vllm':'0.6.3'}
PROBE_MODULES = tuple('search_r1.baselines.probability_probe_'+name for name in
    ('runner','loader','capture','io','monitor','contracts'))
ORIGINAL_MODULES = ('verl','verl.protocol','verl.utils','verl.workers.actor.dp_actor',
    'verl.workers.actor.base','verl.utils.torch_functional','verl.utils.fsdp_utils',
    'verl.utils.debug','verl.utils.debug.performance','verl.utils.fs','verl.utils.hdfs_io',
    'verl.utils.model','verl.models.registry','verl.utils.py_functional')


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
    return strict_json_bytes(bound_bytes(path,expected))


def bound_bytes(path,expected):
    from search_r1.baselines.probability_probe_io import bound_descriptor
    with bound_descriptor(local(path),expected) as stream:payload=stream.read()
    return payload


def strict_json_bytes(payload):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key');result[key]=value
        return result
    def constant(value):raise ValueError('Nonfinite JSON constant '+value)
    result=json.loads(payload,object_pairs_hook=pairs,parse_constant=constant)
    def finite(value):
        if type(value) is float:require(math.isfinite(value),'Nonfinite JSON number')
        elif type(value) is dict:
            for v in value.values():finite(v)
        elif type(value) is list:
            for v in value:finite(v)
    finite(result);return result


def bundle_payload_sha256(bundle,job):
    value={key:item for key,item in bundle.items() if key!='release'}
    expected=str(ROOT/'runs'/('tree-probability-probe-'+job)/'gpu-records')
    require(value['output']==expected,'Runtime output does not belong to this Slurm job')
    value['output']='@OWN_RUN_OUTPUT@'
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def fdinfo_exclusive_owner(text,stat):
    """Linux fdinfo reports FLOCK for this open file description, not somebody's FD.

    The printed PID may be the original preflight exec/fork owner; ownership is
    the stable parent FD itself. Do not replace this with a global /proc/locks
    inode match, which would admit a foreign holder of the same inode.
    """
    locks=[line.split() for line in text.splitlines() if line.startswith('lock:')]
    require(len(locks)==1,'Parent lease descriptor has no unique kernel FLOCK')
    row=locks[0]
    require(len(row)==9 and row[2:5]==['FLOCK','ADVISORY','WRITE'] and
        row[5].isdigit() and int(row[5])>0 and row[7:]==['0','EOF'],
        'Parent descriptor does not own a whole-file exclusive FLOCK')
    identity=row[6].split(':');require(len(identity)==3,'Malformed kernel lock inode')
    require((int(identity[0],16),int(identity[1],16),int(identity[2]))==
        (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino),'Kernel lock inode differs from parent FD')
    return True


def installed_bindings(entries,versions):
    require(type(entries) is list and len(entries)==8 and
        {e['module'] for e in entries}==INSTALLED_MODULES,'Required actual installed source set changed')
    require(versions==INSTALLED_VERSIONS,'Required installed version identity changed')
    for name,version in INSTALLED_VERSIONS.items():
        require(importlib.metadata.version(name)==version,'Actual installed version differs: '+name)
    for entry in entries:
        path=Path(entry['path'])
        require(path.is_absolute() and path.is_relative_to(ROOT/'env/lib/python3.10/site-packages'),
            'Installed source escaped own environment')
        module=importlib.import_module(entry['module'])
        require(Path(module.__file__).resolve()==path.resolve() and sha(path)==entry['sha256'],
            'Actual imported implementation differs: '+entry['module'])


def imported_sources(bundle):
    sources={e['path']:e['sha256'] for e in bundle['probe_source_files']}
    original={e['path']:e['sha256'] for e in bundle['identity']['source_files']}
    require(len(sources)==len(bundle['probe_source_files']) and len(sources)>=700,
        'Complete frozen probe source mapping required')
    for name in PROBE_MODULES+ORIGINAL_MODULES:
        module=importlib.import_module(name);path=local(module.__file__)
        require(str(path) in sources and sha(path)==sources[str(path)],'Actual imported probe/helper source changed: '+name)
        if name in ORIGINAL_MODULES:
            relative=str(path).split('/repo/',1)[-1]
            matches={value for key,value in original.items() if key.endswith('/repo/'+relative)}
            require(matches=={sources[str(path)]},'Original actor/helper bytes differ: '+name)
    worker=str(local(bundle['original_worker_path']))
    require(original.get(worker)==bundle['original_worker_sha256'],'Original AST worker is not bound to source identity')
    native=local(importlib.import_module('verl.utils.torch_functional').__file__)
    require(sources[str(native)]==bundle['native_logprob_sha256'],'Actual native logprob source differs')


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
        fdinfo=Path('/proc',str(state['supervisor']['pid']),'fdinfo',str(lease['fd'])).read_text()
        fdinfo_exclusive_owner(fdinfo,a)
        require((fd.stat().st_dev,fd.stat().st_ino)==(a.st_dev,a.st_ino),'Parent FD changed during kernel ownership check')
        with expected.open('r') as stream:
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN);raise ValueError('GPU lease is no longer held')
    return dict(phase=phase,job_id=job,guard_epoch=row['epoch'],original_background_processes_preserved=True)


def prepare_and_validate_bundle(bundle):
    global ACTIVE
    from search_r1.baselines.probability_probe_contracts import (bind_sources,bind_exact_files,validate_selection,
        validate_fixed_config)
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
            release['runner_commit']==bundle['runner_commit'] and type(release['actual_optimizer_writes']) is int and
            release['actual_optimizer_writes']==0 and
            release['adapter_sha256']==bundle['reviewed_resource_adapter']['sha256'] and
            release['bundle_payload_sha256']==bundle_payload_sha256(bundle,launch['job_id']) and
            release['package_manifest_sha256']==launch['package_manifest_sha256'],
            'Release binds another source/selection')
    selected=bound(bundle['selection_file']['path'],bundle['selection_file']['sha256'])
    require(selected==bundle['selection'] and selected['inventory']==bundle['inventory'],'Runtime selection or inventory changed')
    validate_selection(selected,bundle['identity'],bundle['inventory'],allow_frozen=True)
    bind_sources(bundle['identity']['source_files']);bind_sources(bundle['probe_source_files'])
    for cpu in selected['cpu_results'].values():bound(cpu['path'],cpu['sha256'])
    for asset in bundle['inventory'].values():require(sha(local(asset['path']))==asset['sha256'],'Original tensor asset changed')
    for entry in selected['monitor_companions']:require(sha(local(entry['path']))==entry['sha256'],'Monitor companion changed')
    from omegaconf import OmegaConf
    payload=bound_bytes(bundle['config_path'],bundle['identity']['config_sha256'])
    require(type(bundle['config_text']) is str and payload==bundle['config_text'].encode('utf-8'),
        'Consumed bundle config is not the original bound descriptor bytes')
    config=OmegaConf.to_container(OmegaConf.create(bundle['config_text']),resolve=True)
    validate_fixed_config(config)
    require(bundle['hf_identity']['sha256']==bundle['identity']['parent_hf_sha256'],'HealthyHF identity changed')
    bind_exact_files(bundle['hf_path'],bundle['hf_identity']['files'],bundle['hf_identity']['sha256'])
    index=bound(Path(bundle['hf_path'])/'model.safetensors.index.json',bundle['hf_identity']['files']['model.safetensors.index.json'])
    require(index==bundle['hf_index'],'Consumed HF index differs from original serialized index bytes')
    parent=bound(bundle['parent_state_manifest']['path'],bundle['identity']['parent_state_manifest_sha256'])
    require(parent['identity']['completed_step']==110,'Healthy parent completed step changed')
    installed_bindings(bundle['installed_sources'],bundle['installed_versions'])
    imported_sources(bundle)
    require(bundle['probe_mode']=='eval_like_original_compute_log_prob','Probe mode changed')
    ACTIVE=launch
    check('adapter_bound_after_complete_bundle_validation')
    return True
