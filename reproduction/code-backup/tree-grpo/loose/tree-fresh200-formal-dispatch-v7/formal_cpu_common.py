"""Formal-only metadata/config CPU preparation; reused pure V11 byte helpers."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import tarfile

ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
STAGE='fresh_mu_train200'
PROTOCOL='tree_fresh200_five_leaf_v1'
COMMIT='53082e1667a6ab131a036ca2ef66e004989ed1a9'
MODEL=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
MODEL_ID='744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA_ID='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CODE_SUFFIXES={'.py', '.sh', '.sbatch', '.yaml', '.yml', '.md', '.json', '.txt', '.toml', '.cfg', '.ini', '.patch'}
CODE_PLAIN_NAMES={'LICENSE', 'NOTICE', '.gitignore', 'Makefile'}
CODE_VERSION_FILES={'source/repo/verl/version/version', 'source/repo/verl/single_controller/version/version'}
SOURCE_ROOT_FILES={'setup.py', 'pyproject.toml', 'requirements.txt', 'LICENSE', 'NOTICE', 'README.md', '.gitignore', 'Makefile'}
EXTERNAL_RELATIVE_FILES={'repo/reproduction/h20/retriever_local.py': 'a78f0b493c538dddc6619e0e462995fd9945c25a187c2165e99d4c10622ce459', 'tools/check_real_checkpoint.py': 'b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97', 'tools/probe_tree_shared_retrieval.py': '335a3410bdc19fac01d5d29aa7eea3832c031fae5093f1f9f2d65f76374157b5', 'tools/resource_guard_identity.py': '284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88', 'tools/tree_shared_owned_runtime_fixed.py': '4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0'}
WRAPPER_INVENTORY_ID='42d225fb38510e9726107617e72e06b9deead8539acfa858998ce7830d5ca55e'
WRAPPER_HASHES={'BASE_READ_SNAPSHOT.json': '304164342437cff8f51d9645e01ad9c87e79c48d0f122f6dc941a43a5a1b245a', 'CONFIGURATION_SPEC.json': 'ad6599be178c75b199b072e128d641977334c03c5e1f39d59043c68bcde5b6cb', 'INPUTS.schema.json': 'c8ed56b2a7d9ff6e9cc65c6a8ff965eabcdbb470f2e7d57c85d1493cae8676a6', 'PLAN.md': 'f12be8a90fc67f870b9a2a784a7d060d3cbc813833d0a68d938f36d52dc72349', 'SOURCE_CPU_RECEIPT.json': '175c15bbb1c1287b6628f2e04f83be9221a99e1616d2c67cb688fa761c78d73c', 'SOURCE_CPU_RECEIPT_V2.json': '0cceff7d7753af51c84fb2daf8ca423980c07d419fbe8aedd5ddea4ffdbc84f8', 'SOURCE_VALIDATION_PATH.md': '59393d2301e4e9acc6b86f15801ccde1d571420827f561553b528c60cc314fdf', 'V2_REPAIR.md': 'ecb2ca4cf8a2999b0adb8c4abda216cef952cb7ab25910d7dfa3f5d0f03ca9f6', 'attempt_accounting.py': '8426d39486e92773022dd0d40ef5e4cb0c742ee6317550d20244ed4696913c29', 'build_package_draft.py': 'decf5606f67bfed62245976bfdd4fba57092e26f1517456a97683ade9ca5131b', 'calibration_contract.py': 'fd881ce773b59ec3ca429f2dc9c79437e44dd36a97902d820726e2e0178174e3', 'check_behavior_mu_cpu.py': '751716c631a7cdccdefd3a7b7950f95477a60b82cf100de638b5a02666ce1b24', 'check_partial_raw_cpu.py': '88ad89c641f6b6136d299d5ed214b35f7eb21b4efc0d8e44b2e0227a68e455c2', 'check_training_state_cpu.py': '6406d85c0d1470b5196da4f970fa6da20194f6dee74ea22a3221f561694ef236', 'guard_entry.py': '12c370762f2bf4ff04203fb995afb9c6d438b4b02627e1374271033f2401808e', 'launch.sbatch': 'f19ea318b113299014585d8c7ebc73b2d513742f905606ed757db5973a7ea501', 'partial_raw_evidence.py': '438b42314fff4940d8ae7a0070973154ef52590bfb954b345746b0b5f0ff0f06', 'production_contract.py': '8ff97890a326f99a5b823d13681a068bbd9a61fee3eacb5d56692423cd3204f1', 'render_dispatch.py': 'eeb579bdabe0e3cbaf246e0f8842b6aa7c3c151837ad21b19bb3418a29c6276e', 'run_tree.sh': '8c81c362cc702c1bc5ed0f09cf6c9bd8bf971aab27cac941313d58694f76f56f', 'sitecustomize.py': 'cbf47d937a0286fedbf3e00ab51244d444523ed3b1b1314282e0151dc517bcf6', 'slurm_identity.py': '896b2328c35b23a3162187ba247679e8b0a28baf41ba6bdfa5482f15d818844a', 'submit_once.py': '379992530e72e1d00ea9a717ace103ff08d1f5ce3dee6c95b452e7a82cb3c004', 'supervisor.py': '6b5b267b5b5412bd6d9d41b0607832a09e1b1cf72bd2ca779c6be22de7faf938', 'test_partial_raw_cpu.py': '14c46640165e9bf5e225a6f438f477c059ba0119b54998ca378bcba893717db3', 'test_production_v2_cpu.py': '9c5e151c2e042a98704181356517780e05472e318a0193e674ad09d12096e924', 'test_runtime_contract_cpu.py': '3cf52dbea6c541d83fe2152a88f0a2e969e08a72caf3ca7a2d1a88cc796e4b7f', 'test_training_cost_cpu.py': 'd61be96a3ae38e9a7e9e20c7aef6a81b9f1d6a88caa7be44b09c977c39a7cf0c', 'test_validation_adapter_cpu.py': '332c449734bfe866668ffec073e134632a17c7c8481f84227fc2128a3a9ad1b2', 'train_monitor.py': 'a1f35023fb385b60376ed25fb4f1b20487f7890e0dd1f6a3b6e27591c036634f', 'training_cost_hooks.py': 'f97e9bf3a84959bac8946e4536ba81cfd51e0de99267efc5be7e3ce6b3fabe1b', 'validation_cost_entry.py': 'fc1763162fcb207a3c5e43b126d1080b016f85cf61aa9ed4de1ffb37458f30b7', 'validation_cost_hooks_instruct300.py': '6646356027f7a1a4855b909147eeadd408d66ea03aa17b86a24ff36365e7c3b7'}
WRAPPER_FILES=set(WRAPPER_HASHES)
PRIMARY_SOURCE_HASHES={'search_r1/baselines/gradient_norm.py': '6d0b68fb35b26dcf125732c5da586425238b3130c7898fb63d779920413c664b', 'search_r1/baselines/partial_worker_state.py': '091a6b2efcc4d18ee4fe8e72c3ff7d74c5475243f662de552c67324b9f6546ea', 'search_r1/baselines/policy_safety_runtime.py': '13ef4c832335b4abda4caa5c15c0fe345275558f97c60ca9dfa52ead74b14208', 'search_r1/baselines/rollout.py': '012da6f036b56e2ff661b98ca3151b113c10b0769200da1d517f0b418b147bb9', 'search_r1/baselines/sampling_consistency.py': '887603f802ce98fcb8391bd4d9b38ec54265026d34ff9f6f5a325ce65dfe7d59', 'search_r1/baselines/tree_behavior_mu.py': '499cb74e4c41899c8294a47835b6043458090b2ac09f7c61e53ec7c827ef2652', 'search_r1/baselines/tree_fresh200.py': 'b97bd761679b822935696a219100efa4847d2b4466e588c4bd1258f2d216b53c', 'search_r1/baselines/tree_guard_continue.py': '3a46fe46fc2afa078e582d473f711b820839dd37e58b21a68c09b22cad61e574', 'search_r1/baselines/tree_partial_state.py': '8e3e046c4febe7cbe841b92ca633a2e09c85ed3953ed98b9603ab6993b86ace8', 'search_r1/baselines/tree_raw_tokens.py': '15b696607846046f671e30531420c6f48cca321762f77016f4e5bdb7868ea229', 'verl/third_party/vllm/vllm_v_0_6_3/llm.py': '047fa0bd83bb525d6d2469154bdcd65df95fe3be3644dcfe4674d288b6cb7a72', 'verl/trainer/config/ppo_tree_fresh200.yaml': 'b7d68e3bce6247be1424965d6bcfe7a3e4723ca1428fedcd6847c3d384649d75', 'verl/trainer/ppo/ray_trainer.py': '55208794e8dfc14d9f08c5a45b8886b4d58889b5048daec8acd8b177ad9703f2', 'verl/utils/fsdp_utils.py': '6cecf644779faa5abc6933ccd892a5d2da605aa45088ef42f258a3918911e0c6', 'verl/workers/actor/dp_actor.py': '080af9d96e391410acc5568f3a8b8cb9e7aa4d5619919f744292ec74fea90660', 'verl/workers/fsdp_workers.py': 'ee91dd80d9fc59ae21f8e3411f5dd32530b97f6a558967cd266c3979cd3acc1c'}
ADDITIONAL_SOURCE_HASHES={'search_r1/baselines/controller.py': '2a1c5fdf6dd8dc265ebcfb3ef3717f5a9f522e0832c506cbd0e94d820a16eda6', 'search_r1/baselines/training_state.py': '50cf8ab7eb37f198cc2833c20498b12ce6e2d8eacd6b21ed5e7a2357820f62d8', 'search_r1/baselines/trajectory_log.py': '93ba273f28949a0606c6ea468f297723187831dc760b7205d9c87d9730827b6f', 'search_r1/baselines/tree_rank_audit.py': '4ef35ac5e0ab0089a669fd459a342678cd9dd46f00e4cd51c917f191e56c183b'}
SOURCE_INTERFACE_RECEIPT_SHA='e52577f068bc7b19ef90dbff1b98ae3e803aa97cbaf4a43bf037fce82d439242'
INPUT_NAMES=('actual_calibration_ready','actual_calibration_acceptance','threshold_acceptance','source_context_free_review')


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


def verify_package(target,manifest):
    for name,expected in manifest['files'].items():require(file_sha(target/name)==expected,'Package file changed: '+name)
    for name,expected in manifest['external_files'].items():require(file_sha(name)==expected,'External helper changed: '+name)


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


def require_observation_descriptors(values):
    """Refuse missing actual metadata before imports, extraction or composition."""
    require(type(values) is dict and set(values)==set(INPUT_NAMES), 'Four actual observation descriptors required')
    for name in INPUT_NAMES:
        entry=values[name]
        require(type(entry) is dict and set(entry)=={'path','sha256'}, 'Actual non-null path/SHA descriptor required: '+name)
        valid_sha(entry['sha256'])
        require(type(entry['path']) is str, 'Actual observation path must be text: '+name)
        local_path(entry['path'])


def validate_manifest(manifest):
    """Formal source/wrapper/metadata declarations only; no observation acceptance."""
    require(manifest['stage']==STAGE and manifest['training_protocol']==PROTOCOL and manifest['code_commit']==COMMIT,
            'Wrong formal source/stage/protocol')
    require(manifest['scope']=='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW', 'Unfrozen formal package')
    require(not any(key in manifest for key in ('parent_manifest','healthy110_metadata','recovery_manifest')), 'Formal fresh package forbids recovery references')
    require(manifest['official_model_path']==str(MODEL) and manifest['model_identity']['sha256']==MODEL_ID
            and manifest['common_data_identity']==DATA_ID, 'Official formal model/data identity changed')
    require(all(manifest[key] is False for key in ('production_thresholds_accepted','actual_3b_training_accepted',
            'long_training_ready','original_sampler_actor_gate_accepted')), 'CPU preparation cannot imply training or production acceptance')
    require(manifest['bootstrap_thresholds_accepted'] is True
            and manifest['full_run_continuation_requires_same_instance_first5_threshold_assessment'] is True,
            'Only first-five bootstrap may initialize this same live formal instance')
    require(type(manifest['files']) is dict and manifest['files'], 'Missing full formal package inventory')
    for name,value in manifest['files'].items():
        relative_file(name);valid_sha(value)
    require(manifest['wrapper_inventory_sha256']==WRAPPER_INVENTORY_ID
            and manifest['wrapper_files']==WRAPPER_HASHES and set(manifest['wrapper_files'])==WRAPPER_FILES
            and len(WRAPPER_FILES)==33 and all(manifest['files'].get(name)==value for name,value in WRAPPER_HASHES.items()),
            'Reviewed immutable formal33 wrapper inventory changed')
    require(manifest['source_interface_files']==PRIMARY_SOURCE_HASHES
            and manifest['additional_runtime_closure']==ADDITIONAL_SOURCE_HASHES
            and not set(PRIMARY_SOURCE_HASHES).intersection(ADDITIONAL_SOURCE_HASHES)
            and len(PRIMARY_SOURCE_HASHES)==16 and len(ADDITIONAL_SOURCE_HASHES)==4, 'Formal16+4 source identity closure changed')
    for name,value in {**PRIMARY_SOURCE_HASHES,**ADDITIONAL_SOURCE_HASHES}.items():
        require(manifest['files'].get('source/repo/'+name)==value, 'Formal source byte pin changed: '+name)
    require(manifest['source_interface_manifest_sha256']==SOURCE_INTERFACE_RECEIPT_SHA
            and manifest['files'].get('observed-inputs/source-interface-manifest.json')==SOURCE_INTERFACE_RECEIPT_SHA,
            'Formal source-interface compatibility envelope changed')
    require(manifest['external_files']=={str(ROOT/name):value for name,value in EXTERNAL_RELATIVE_FILES.items()},
            'Formal external helper closure changed')
    require_observation_descriptors(manifest['production_inputs'])
    require(type(manifest['production_input_sha256']) is dict
            and set(manifest['production_input_sha256'])==set(INPUT_NAMES), 'Actual four observation SHA inventory required')
    for name,entry in manifest['production_inputs'].items():
        require(manifest['production_input_sha256'][name]==entry['sha256']
                and manifest['files'].get('observed-inputs/'+name+'.json')==entry['sha256'],
                'Formal observation byte pin changed: '+name)


def selected(config,path):
    value=config
    for key in path.split('.'):
        value=value[key]
    return value


def check_resolved(resolved,spec,thresholds,checkpoints):
    """Compare the frozen88 fields, own path and two actually observed fields."""
    expected=dict(spec['expected_resolved_values'])
    require(len(expected)==88 and expected['trainer.default_local_dir']=='@OWN_RUN_CHECKPOINTS@', 'Formal fixed88 spec changed')
    expected['trainer.default_local_dir']=str(checkpoints)
    prefix='actor_rollout_ref.actor.policy_safety.'
    expected[prefix+'negative_tail_limit']=thresholds['negative_tail_limit']
    expected[prefix+'sampling_probability_correction.initial_actor_repeat_max_abs_delta']=thresholds['initial_actor_repeat_max_abs_delta']
    for key,value in expected.items():
        actual=selected(resolved,key)
        require(actual==value and (type(actual) is type(value) if type(value) in (bool,int) else type(actual) is not bool),
                'Resolved formal config field changed: '+key)
    return expected
