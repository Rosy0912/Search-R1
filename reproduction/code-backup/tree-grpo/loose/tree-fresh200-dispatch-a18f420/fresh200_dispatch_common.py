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
COMMIT = "a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98"
MODEL = ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
MODEL_ID = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA_ID = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
ENVIRONMENT = {'TREE_FRESH_INITIALIZATION_ONLY':'1','TREE_PPO_CALIBRATION_ONLY':'1'}
PROTECTED = {**{str(pid):'1502352181' for pid in (1680782,1680783,1680784,1680785)},
             **{str(pid):'1502398762' for pid in (1696443,1696444,1696445,1696446)}}
OWN_PREFIXES = ('tree-fresh200','tree-instruct300','tree-recovery300','tree-safety300',
                'tree-safety110','tree-policy-safety','tree-probability-probe','tree-single-tp1')
EXTERNAL_RELATIVE_FILES = {
    'repo/reproduction/h20/retriever_local.py': 'a78f0b493c538dddc6619e0e462995fd9945c25a187c2165e99d4c10622ce459',
    'tools/check_real_checkpoint.py': 'b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97',
    'tools/probe_tree_shared_retrieval.py': '335a3410bdc19fac01d5d29aa7eea3832c031fae5093f1f9f2d65f76374157b5',
    'tools/resource_guard_identity.py': '284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88',
    'tools/tree_shared_owned_runtime_fixed.py': '4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0',
}
CODE_SUFFIXES = {'.py', '.sh', '.sbatch', '.yaml', '.yml', '.md', '.json', '.txt', '.toml', '.cfg', '.ini', '.patch'}
CODE_PLAIN_NAMES = {'LICENSE', 'NOTICE', '.gitignore', 'Makefile'}
CODE_VERSION_FILES = {'source/repo/verl/version/version', 'source/repo/verl/single_controller/version/version'}
SOURCE_ROOT_FILES = {'setup.py','pyproject.toml','requirements.txt','LICENSE','NOTICE','README.md','.gitignore','Makefile'}
WRAPPER_INVENTORY_ID = 'c0f95b872771da3e37525c2ec56e837bcc3b53d5167e09e127b0818bca2a7c85'
WRAPPER_HASHES = {'CONFIGURATION_OVERRIDES.json': '51c66729aa68203f5d0cd67dc3f7c1a5ef828d3484cbe5f570bef663d18a7c07', 'INPUTS.schema.json': '159d36b6f0b825a64c7ea63e4e3c4fec9b627b1536f6e79cd44b8be03069c48a', 'PLAN.md': 'ab6f72237b873f7c2a07e2d59c7902f00eae19c52e79bd9329774b7718ef51e9', 'SOURCE_CPU_RECEIPT.json': '13be65dd742db6e6adfe61dd9189dbb9a8d2aca8296741868eba844f9da737f0', 'build_package_draft.py': '7da1eadc49a63aa52f525faabd3906b2f78fb375f1fa58adfd5ef43c6cbe6ce8', 'calibration_contract.py': '598cc5acae2037935348629e626c5b461694d8a940f5f0668872b58b4dbe8827', 'launch.sbatch': '8a4ae2bbfaa744a4f86dbaa14cbbb6453283ac20d812f0a21cb5ed171843eca3', 'render_dispatch.py': '92921474e7d38b012adc1b9582770b55fa5b95ae350f4a2828deea5491056c11', 'run_tree.sh': '72e8378ca11d44f31ce38983eff2e9b196901d6c15846ecce44ecdf13112330b', 'sitecustomize.py': '6b7e3cb0ae998b0248323efe1a8dd1815e81985412fdbf087134752f417265a0', 'supervisor.py': '83fc3e9066c9dfa923913b06e8a86153bd6cb0789ade8cfe29849508cae14040', 'test_runtime_contract_cpu.py': '3b179e72c544e9672473d857a8ec1a54a8092dc02cc0133266e81a8218e235db', 'tokenizer_raw_preflight.py': '6aa83503db0a234046b8926a2988025726e1fe5256597e5b3b955e23baeed7ed', 'validation_cost_entry.py': '51e19b3fb0c64a908beaab99f3559fe64903f6d5b4bcb8a2bafb18b9495ed26a', 'validation_cost_hooks_instruct300.py': 'c4c49745b4892f169b601cbc5f27fbcab7fe9c154ae7bc4f508ba82c4440bbd1'}
WRAPPER_FILES = {'CONFIGURATION_OVERRIDES.json','INPUTS.schema.json','PLAN.md','SOURCE_CPU_RECEIPT.json',
    'build_package_draft.py','calibration_contract.py','launch.sbatch','render_dispatch.py','run_tree.sh',
    'sitecustomize.py','supervisor.py','test_runtime_contract_cpu.py','tokenizer_raw_preflight.py',
    'validation_cost_entry.py','validation_cost_hooks_instruct300.py'}
SOURCE_INTERFACE_FILES = {
    'search_r1/baselines/controller.py',
    'search_r1/baselines/training_state.py',
    'search_r1/baselines/trajectory_log.py',
    'search_r1/baselines/tree_rank_audit.py',
    'search_r1/baselines/tree_guard_continue.py',
    'search_r1/baselines/gradient_norm.py',
    'search_r1/baselines/sampling_consistency.py',
    'verl/workers/fsdp_workers.py',
    'verl/workers/actor/dp_actor.py',
    'search_r1/baselines/policy_safety_runtime.py',
    'search_r1/baselines/partial_worker_state.py',
    'search_r1/baselines/tree_partial_state.py',
    'search_r1/baselines/tree_behavior_mu.py',
    'search_r1/baselines/tree_raw_tokens.py',
    'search_r1/baselines/tree_fresh200.py',
    'search_r1/baselines/rollout.py',
    'verl/trainer/ppo/ray_trainer.py',
    'verl/trainer/config/ppo_tree_fresh200.yaml',
    'verl/third_party/vllm/vllm_v_0_6_3/llm.py',
    'verl/utils/fsdp_utils.py',
}


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
    require(path.suffix.lower() in CODE_SUFFIXES or path.name in CODE_PLAIN_NAMES or name in CODE_VERSION_FILES,
            'Code-only allowlist forbids model/data/index/tensor assets or unsupported file types')
    if name.startswith('source/repo/'):
        source_name=name[len('source/repo/'):]
        require(source_name in SOURCE_ROOT_FILES or source_name.startswith(('search_r1/','verl/')),
                'Runtime source path closure forbids historical audits, reproduction, public and test payloads')
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
    require(manifest['trial_thresholds']==dict(target_old_policy_kl=.2,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.),
            'Wrong explicit unaccepted trial thresholds')
    calibration=manifest['probability_calibration']
    require(calibration['environment']==ENVIRONMENT and calibration['source_interface_reviewed'] is True,
            'Fresh zero-write environment/source interface not accepted')
    require(type(manifest['files']) is dict and manifest['files'],'Missing full package file inventory')
    for name,value in manifest['files'].items():relative_file(name);valid_sha(value)
    require(manifest['wrapper_inventory_sha256']==WRAPPER_INVENTORY_ID
            and type(manifest['wrapper_files']) is dict and manifest['wrapper_files']==WRAPPER_HASHES
            and set(manifest['wrapper_files'])==WRAPPER_FILES
            and all(manifest['files'].get(name)==value for name,value in manifest['wrapper_files'].items()),
            'Reviewed complete15 wrapper inventory is missing or changed')
    require(all(name in manifest['files'] for name in ('supervisor.py','launch.sbatch','CONFIGURATION_OVERRIDES.json',
            'calibration_contract.py','source/repo/search_r1/baselines/sampling_consistency.py','source/repo/verl/workers/fsdp_workers.py',
            'source/repo/verl/trainer/ppo/ray_trainer.py','source/repo/verl/trainer/config/ppo_tree_fresh200.yaml')),
            'Fresh runtime/source/config inventory incomplete')
    require(manifest['external_files'] == {str(ROOT/name): value for name, value in EXTERNAL_RELATIVE_FILES.items()},
            'Missing or changed actual external helper import/execution closure')
    for name,value in manifest['external_files'].items():local_path(name);valid_sha(value)


def verify_package(target,manifest):
    for name,expected in manifest['files'].items():require(file_sha(target/name)==expected,'Package file changed: '+name)
    for name,expected in manifest['external_files'].items():require(file_sha(name)==expected,'External helper changed: '+name)


def validate_reviews(inputs,manifest,for_dispatch=False):
    interface=json_bound(inputs['source_interface_receipt'])
    require(interface['status']=='FRESH200_REPAIRED_SOURCE_INTERFACE_CPU_OWNER_VERIFIED',
            'Wrong source-interface owner receipt')
    require(interface.get('code_commit')==COMMIT and interface.get('source_clean') is True,
            'Source-interface owner must bind the final clean source commit')
    require(all(interface.get(key) is False for key in ('actual_gpu_executed','actual_remote_executed','training_release')),
            'Source-interface owner receipt must state CPU-only/no remote/no training release')
    require(type(interface.get('source_files')) is dict
            and SOURCE_INTERFACE_FILES.issubset(interface['source_files']),
            'Repaired source-interface receipt must cover all20 affected source/config/runtime files')
    for name,value in interface['source_files'].items():
        relative_file(name);valid_sha(value)
        require(value==manifest['files'].get('source/repo/'+name),'Source-interface bytes changed: '+name)
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
    require(type(audit['source_files']) is dict and SOURCE_INTERFACE_FILES.issubset(audit['source_files'])
            and audit['wrapper_files'],'Independent byte audit must cover all20 repaired source/config/runtime files')
    require(set(audit['wrapper_files'])==WRAPPER_FILES,'Independent byte audit must cover all15 runtime files')
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
