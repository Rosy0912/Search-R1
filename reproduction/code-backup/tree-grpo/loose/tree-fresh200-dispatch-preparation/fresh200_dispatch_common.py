"""Bound same-node fresh calibration metadata; no service/model/GPU import."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import tarfile

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
STAGE = 'fresh_mu_calibration'
PROTOCOL = 'tree_fresh200_five_leaf_v1'
COMMIT = '1e06c6aa6f54eee5de8e0a29dd50efea76111c10'
MODEL = ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
MODEL_ID = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA_ID = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
ENVIRONMENT = {'TREE_FRESH_INITIALIZATION_ONLY':'1','TREE_PPO_CALIBRATION_ONLY':'1'}
PROTECTED = {**{str(pid):'1502352181' for pid in (1680782,1680783,1680784,1680785)},
             **{str(pid):'1502398762' for pid in (1696443,1696444,1696445,1696446)}}
OWN_PREFIXES = ('tree-fresh200','tree-instruct300','tree-recovery300','tree-safety300',
                'tree-safety110','tree-policy-safety','tree-probability-probe','tree-single-tp1')


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'Required exact lowercase SHA256')
    return value


def local_path(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve()),'Path must remain in this task on h100-9')
    return path


def read_bound(path,expected):
    valid_sha(expected);path=local_path(path)
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno()); require(stat.S_ISREG(before.st_mode),'Bound source is not a regular file')
        raw=stream.read();after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Bound file changed during read')
    require(digest(raw)==expected,'Bound file SHA mismatch: '+str(path))
    return raw


def json_bound(entry):return json.loads(read_bound(entry['path'],entry['sha256']))


def file_sha(path):
    path=local_path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());require(stat.S_ISREG(before.st_mode),'File must be regular')
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'File changed during hashing')
    return h.hexdigest()


def relative_file(name):
    path=PurePosixPath(name)
    require(type(name) is str and name and '\\' not in name and not path.is_absolute()
            and '..' not in path.parts and '.' not in path.parts and str(path)==name,'Unsafe/noncanonical package file path')
    require(path.suffix.lower() not in ('.pt','.bin','.safetensors','.parquet','.npy','.faiss'),
            'Code archive must not transport model/data/index/tensor assets')
    return path


def validate_manifest(manifest):
    require(manifest['stage']==STAGE and manifest['training_protocol']==PROTOCOL and manifest['code_commit']==COMMIT,
            'Wrong fresh source/stage/protocol')
    require(manifest['scope']=='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW','Unfrozen package')
    require(not any(key in manifest for key in ('parent_manifest','healthy110_metadata','recovery_manifest')),
            'Fresh package may not contain recovery state references')
    require(manifest['official_model_path']==str(MODEL) and manifest['model_identity']['sha256']==MODEL_ID
            and manifest['common_data_identity']==DATA_ID,'Official same-node model/split identity changed')
    require(manifest['production_thresholds_accepted'] is False and manifest['actual_3b_training_accepted'] is False
            and manifest['long_training_ready'] is False and manifest['original_sampler_actor_gate_accepted'] is False,
            'Calibration cannot imply training/production/probability acceptance')
    require(manifest['trial_thresholds']==dict(target_old_policy_kl=.01,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.),
            'Wrong explicit unaccepted trial thresholds')
    calibration=manifest['probability_calibration']
    require(calibration['environment']==ENVIRONMENT and calibration['source_interface_reviewed'] is True,
            'Fresh zero-write environment/source interface not accepted')
    require(type(manifest['files']) is dict and manifest['files'],'Missing full package file inventory')
    for name,value in manifest['files'].items():relative_file(name);valid_sha(value)
    require(all(name in manifest['files'] for name in ('supervisor.py','launch.sbatch','CONFIGURATION_OVERRIDES.json',
            'calibration_contract.py','source/repo/search_r1/baselines/sampling_consistency.py','source/repo/verl/workers/fsdp_workers.py',
            'source/repo/verl/trainer/ppo/ray_trainer.py','source/repo/verl/trainer/config/ppo_tree_fresh200.yaml')),
            'Fresh runtime/source/config inventory incomplete')
    require(type(manifest['external_files']) is dict and manifest['external_files'],'Missing external helper bindings')
    for name,value in manifest['external_files'].items():local_path(name);valid_sha(value)


def verify_package(target,manifest):
    for name,expected in manifest['files'].items():require(file_sha(target/name)==expected,'Package file changed: '+name)
    for name,expected in manifest['external_files'].items():require(file_sha(name)==expected,'External helper changed: '+name)


def validate_reviews(inputs,manifest,for_dispatch=False):
    interface=json_bound(inputs['source_interface_receipt'])
    require(interface['status']=='FRESH_CALIBRATION_MODE_AND_DIRECT_RANK_SCHEDULER_SOURCE_CPU_IMPLEMENTED',
            'Wrong source-interface owner receipt')
    for name in ('search_r1/baselines/sampling_consistency.py','verl/workers/fsdp_workers.py'):
        require(interface['source_files'][name]==manifest['files']['source/repo/'+name],'Source-interface bytes changed')
    context=json_bound(inputs['context_free_review'])
    require(context['reviewer_context']['fork_turns']=='none' and context['reviewer_context']['inherited_parent_conversation'] is False
            and context['source']['audited_clean_head']==COMMIT and context['source']['initial_clean'] is True
            and context['source_interface_reviewed'] is True
            and context['stages']['preparation']['accepted'] is True
            and context['execution_scope']['same_h100_9_cpu_only_preparation_allowed'] is True,
            'Missing completed independent source review and explicit same-node CPU preparation scope')
    # A completed, blocked review is sufficient only for CPU preparation.
    # Its False acceptance is preserved verbatim and never converted to READY.
    stage=context['stages'][STAGE]
    require(stage['source_review_completed'] is True and type(stage['source_review_accepted']) is bool,
            'Missing completed source-stage review')
    if for_dispatch:
        require(stage['source_review_accepted'] is True,'Independent fresh calibration source review remains blocked')
    audit=json_bound(inputs['context_byte_audit'])
    require(inputs['context_byte_audit']['sha256']==context['artifacts']['BYTE_AUDIT.json'] and audit['git_head']==COMMIT,
            'Independent byte audit not bound to review/source')
    require(audit['source_files'] and audit['wrapper_files'],'Independent source/runtime inventories are empty')
    for name,entry in audit['source_files'].items():
        require(entry['matches_git_head'] is True and manifest['files']['source/repo/'+name]==entry['sha256'],
                'Independent reviewed source differs from package')
    for name,entry in audit['wrapper_files'].items():
        require(entry['matches'] is True and manifest['files'][name]==entry['sha256'],'Independent reviewed runtime differs from package')
    return dict(source_interface=inputs['source_interface_receipt'],context_free=inputs['context_free_review'],
                context_byte_audit=inputs['context_byte_audit'],source_review_accepted=stage['source_review_accepted'])


def safe_extract(archive,archive_sha,target,manifest_sha,max_bytes):
    """Hash/read one FD; accept only distinct ordinary code/metadata files."""
    require(type(max_bytes) is int and 0<max_bytes<=256*1024**2,'Explicit bounded code archive size required')
    archive=local_path(archive);target=local_path(target)
    require(target.parent.resolve()==(ROOT/'instruct300/candidates').resolve() and not target.exists(),'New no-clobber candidate path required')
    with archive.open('rb') as stream:
        before=os.fstat(stream.fileno());h=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        require(h.hexdigest()==valid_sha(archive_sha),'Archive SHA mismatch')
        stream.seek(0)
        with tarfile.open(fileobj=stream,mode='r:*') as bundle:
            members=bundle.getmembers();seen=set();total=0
            for member in members:
                relative_file(member.name)
                require(member.isfile() and not member.issym() and not member.islnk(),'Archive accepts regular files only')
                require(member.name not in seen,'Duplicate archive member');seen.add(member.name);total+=member.size
            require(0<total<=max_bytes and 'manifest.json' in seen,'Uncompressed code size/manifest bound failed')
            manifest_raw=bundle.extractfile('manifest.json').read()
            require(digest(manifest_raw)==valid_sha(manifest_sha),'Archived manifest SHA mismatch')
            manifest=json.loads(manifest_raw);validate_manifest(manifest)
            require(seen==set(manifest['files'])|{'manifest.json'},'Archive inventory differs from frozen manifest')
            # Reserve the destination before any extraction. A partial failure is retained, never retried in place.
            target.mkdir()
            for member in members:
                destination=target/member.name;destination.parent.mkdir(parents=True,exist_ok=True)
                with bundle.extractfile(member) as source,destination.open('xb') as output:
                    while True:
                        block=source.read(1024**2)
                        if not block:break
                        output.write(block)
        after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Archive changed during extraction')
    require(file_sha(target/'manifest.json')==manifest_sha,'Extracted manifest changed')
    verify_package(target,manifest)
    return manifest



def verify_local_assets(manifest):
    """Rescan existing same-node bytes only; never load/import/copy weights."""
    identity=manifest['model_identity'];expected=identity['files']
    actual={str(path.relative_to(MODEL)):file_sha(path) for path in sorted(MODEL.rglob('*')) if path.is_file()}
    require(actual==expected and digest(json.dumps(actual,sort_keys=True).encode())==MODEL_ID,
            'Full local official model/tokenizer inventory changed')
    data_root=ROOT/'instruct300/common-data'
    common=json_bound(dict(path=str(data_root/'manifest.json'),sha256=DATA_ID))
    require(common['protocol']=='search-e1-common-300' and common['seed']==42,'Fixed split protocol changed')
    require(set(common['splits'])=={'train','validation','stage','final'} and len(common['preparation']['raw_sources'])==9,
            'Full common split/raw source inventory missing')
    counts={'train':169028,'validation':512,'stage':64,'final':51713}
    records=list(common['preparation']['raw_sources'])+list(common['splits'].values())
    for record in records:
        relative=PurePosixPath(record['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Asset path escapes same-node common data')
        path=data_root/str(relative)
        require(path.resolve().is_relative_to(data_root.resolve()) and file_sha(path)==record['sha256'],
                'Existing same-node split/raw data changed')
    require(all(common['splits'][name]['rows']==count for name,count in counts.items()),'Fixed split counts changed')

def assert_host():
    require(socket.gethostname().split('.')[0]=='h100-9','Only h100-9 execution is authorized')
    require(not any(key.startswith('TREE_RECOVERY_') for key in os.environ),'Fresh helper forbids recovery environment')


def write_once(path,value):
    path=local_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return digest(raw)


def queue_clear(text):
    require(type(text) is str,'Unknown queue response')
    for line in text.splitlines():
        fields=line.split('|');require(len(fields)==4 and fields[0].strip().isdigit() and fields[1].strip()
            and fields[2].strip() and fields[3].strip(),'Unparseable scheduler queue response')
        require(not fields[1].strip().startswith(OWN_PREFIXES),'Active own Tree task; duplicate dispatch prohibited')


def protected_processes(proc_root=Path('/proc')):
    for pid,start in PROTECTED.items():
        fields=(proc_root/pid/'stat').read_text().rsplit(') ',1)[1].split()
        require(len(fields)>19 and fields[19]==start and fields[0] not in ('Z','X'),'Original protected PID/start changed')
