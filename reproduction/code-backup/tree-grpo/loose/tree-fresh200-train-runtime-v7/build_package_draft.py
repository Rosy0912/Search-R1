"""Local-only production-package draft builder; never release/submit/upload.

All observation/threshold reviews must be real pinned JSON inputs. Whole-package
release is external and obtained after this immutable package exists.
"""
import argparse,ast,hashlib,io,json,os,re,stat,subprocess,tarfile
from pathlib import Path
from production_contract import (INPUT_NAMES,ROOT,SOURCE,PROFILE,read_bound,
    observed_production_inputs,configuration_overrides)
HERE=Path(__file__).resolve().parent
WRAPPER_FILES=(
    'BASE_READ_SNAPSHOT.json','CONFIGURATION_SPEC.json','INPUTS.schema.json','PLAN.md','attempt_accounting.py',
    'SOURCE_VALIDATION_PATH.md','SOURCE_CPU_RECEIPT.json','build_package_draft.py',
    'calibration_contract.py','check_behavior_mu_cpu.py','check_training_state_cpu.py',
    'launch.sbatch','production_contract.py','render_dispatch.py','run_tree.sh',
    'sitecustomize.py','supervisor.py','test_runtime_contract_cpu.py',
    'test_validation_adapter_cpu.py','test_training_cost_cpu.py','training_cost_hooks.py','train_monitor.py','validation_cost_entry.py',
    'validation_cost_hooks_instruct300.py','slurm_identity.py','guard_entry.py','submit_once.py',
    'partial_raw_evidence.py','check_partial_raw_cpu.py','test_partial_raw_cpu.py','test_production_v2_cpu.py','V2_REPAIR.md','SOURCE_CPU_RECEIPT_V2.json')
RUNTIME_ENTRIES=('run_tree.sh','validation_cost_entry.py','validation_cost_hooks_instruct300.py','sitecustomize.py')
CODE_SUFFIXES=frozenset(('.py','.sh','.sbatch','.yaml','.yml','.md','.json','.txt','.toml','.cfg','.ini','.patch'))
CODE_EXTENSIONLESS=frozenset(('LICENSE','NOTICE','.gitignore','Makefile','verl/version/version','verl/single_controller/version/version'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition: raise ValueError(message)


def read_regular_bound(path, expected):
    """Read pinned ordinary bytes on one FD; no symlink or guessed digest."""
    require(type(expected) is str and re.fullmatch('[0-9a-f]{64}',expected), 'Exact source SHA required')
    path=Path(path)
    observed=path.lstat()
    require(stat.S_ISREG(observed.st_mode), 'Reviewed source must be a regular file, not symlink: '+str(path))
    flags=os.O_RDONLY|os.O_NOFOLLOW
    with os.fdopen(os.open(path,flags),'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), 'Reviewed source is not a regular file: '+str(path))
        require((observed.st_dev,observed.st_ino)==(before.st_dev,before.st_ino),
                'Reviewed source path changed before opening: '+str(path))
        raw=stream.read(); after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                'Reviewed source changed during read: '+str(path))
    require(hashlib.sha256(raw).hexdigest()==expected,'Reviewed source bytes differ: '+str(path))
    return raw


def prepare_wrapper_inventory(inventory, source_inputs, source_dir=HERE):
    """Verify the actual fixed production-file review inventory, including four entries."""
    require(type(inventory.get('wrapper_files')) is dict and
            set(inventory['wrapper_files'])==set(WRAPPER_FILES), 'Reviewed wrapper inventory must contain all fixed production wrapper files')
    frozen=inventory['wrapper_files']; all_files=inventory['files']
    require(all(all_files.get(name)==frozen[name] for name in WRAPPER_FILES),
            'Wrapper review inventory differs from frozen full file list')
    prepared={name:read_regular_bound(Path(source_dir)/name,frozen[name]) for name in WRAPPER_FILES}
    supplied=source_inputs.get('runtime_entries',{})
    require(all(name in supplied for name in RUNTIME_ENTRIES), 'Missing fresh reviewed entry/validation observer inputs')
    for name in RUNTIME_ENTRIES:
        entry=supplied[name]
        require(entry['sha256']==frozen[name], 'Runtime entry SHA differs from reviewed inventory: '+name)
        require(read_regular_bound(entry['path'],entry['sha256'])==prepared[name],
                'Runtime entry bytes differ from reviewed inventory: '+name)
    return prepared


def copy_wrapper_inventory(output, prepared):
    """Copy every reviewed byte unchanged; this does not create a release."""
    require(set(prepared)==set(WRAPPER_FILES),'Cannot publish a partial wrapper inventory')
    for name in WRAPPER_FILES:
        with (Path(output)/name).open('xb') as stream: stream.write(prepared[name])
    return {name:sha(Path(output)/name) for name in WRAPPER_FILES}


def source_file_allowed(name):
    relative=Path(name)
    return name in CODE_EXTENSIONLESS or relative.suffix.lower() in CODE_SUFFIXES


def materialize_source_archive(archive, output):
    """Select ordinary reviewed code/metadata; never transport tracked assets."""
    selected,excluded={},{}
    seen=set()
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as bundle:
        for member in bundle.getmembers():
            relative=Path(member.name)
            require(not relative.is_absolute() and '..' not in relative.parts
                    and not member.issym() and not member.islnk()
                    and (member.isfile() or member.isdir()),
                    'Source archive contains an unsupported link/path/type')
            if member.isdir(): continue
            require(str(relative)==member.name and member.name not in seen,'Source archive file path is duplicate/noncanonical')
            seen.add(member.name)
            raw=bundle.extractfile(member).read()
            digest=hashlib.sha256(raw).hexdigest()
            if not source_file_allowed(member.name):
                excluded[member.name]=digest
                continue
            destination=Path(output)/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream: stream.write(raw)
            selected[member.name]=digest
    return dict(protocol='runtime_review_code_metadata_allowlist_v1',
                allowed_suffixes=sorted(CODE_SUFFIXES),allowed_extensionless=sorted(CODE_EXTENSIONLESS),
                selected_count=len(selected),excluded_count=len(excluded),
                selected_files=selected,excluded_files=excluded,not_a_full_repository_asset_copy=True)


def require_actual_fresh_api(repo):
    """Reject old interface packages; presence checks are not independent review."""
    workers=ast.parse((repo/'verl/workers/fsdp_workers.py').read_text())
    declarations=[node for node in ast.walk(workers) if isinstance(node,ast.FunctionDef)]
    rpc=[node for node in declarations if node.name=='inspect_fresh_initialization']
    require(len(rpc)==1 and [a.arg for a in rpc[0].args.args]==['self','expected_model_path'],
            'Fresh worker RPC must consume the driver-bound official model path')
    require(any(node.name=='inspect_fresh_worker_state' for node in declarations),
            'Actual empty-state inspection helper is missing')
    driver=ast.parse((repo/'verl/trainer/ppo/ray_trainer.py').read_text())
    calls=[node for node in ast.walk(driver) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute) and node.func.attr=='inspect_fresh_initialization']
    require(len(calls)==1 and len(calls[0].args)==1 and not calls[0].keywords
            and ast.unparse(calls[0].args[0])=='self.config.actor_rollout_ref.model.path',
            'Fresh fit must pass its actual configured official model path to the RPC')
    text=(repo/'search_r1/baselines/sampling_consistency.py').read_text()
    require('FRESH_BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES' in text
            and 'fresh_post_monitor_state' in text and 'fresh_calibration_driver_provenance' in text,
            'Actual fresh calibration terminal/state interface missing')


def verify_source_interface(source,repo):
    """Verify sixteen primary and four runtime source byte interfaces."""
    descriptor=source['source_interface_manifest']
    value,digest,raw=read_bound(descriptor['path'],descriptor['sha256'])
    require(digest==source['source_interface_receipt_sha256'],'Owner-interface descriptor/hint digest differs')
    require(value['code_commit']==source['code_commit'] and value['source_clean'] is True
            and value['source_interface_count']==16 and len(value['source_files'])==16
            and value['status']=='FRESH200_REPAIRED_SOURCE_INTERFACE_CPU_OWNER_VERIFIED'
            and value['source_owner_verification_completed'] is True and value['source_cpu_tests_passed'] is True
            and value['actual_gpu_executed'] is False and value['training_release'] is False
            and value['required_new_interface_key']=='search_r1/baselines/tree_guard_continue.py'
            and value['actor_continue_protocol']=='tree_guard_continue_fresh_rollout_v1',
            'Frozen actual sixteen-interface source identity missing')
    closure=dict(value['source_files'],**value['additional_changed_runtime_files'])
    require(len(value['additional_changed_runtime_files'])==4,'Affected runtime closure source inventory missing')
    require('verl/utils/fsdp_utils.py' in value['source_files'] and len(closure)==20
            and not set(value['source_files']).intersection(value['additional_changed_runtime_files']),
            'Adam offload helper and complete twenty-file source closure required')
    for relative,expected in closure.items():
        path=Path(relative);require(not path.is_absolute() and '..' not in path.parts,'Unsafe source interface path')
        read_regular_bound(Path(repo)/path,expected)
    return value,digest,raw


def verified_observation_metadata(inputs):
    metadata=inputs['production_metadata']; pinned=inputs['production_inputs']
    require(set(metadata)==set(pinned)==set(INPUT_NAMES),'Four actual production metadata/runtime descriptors required')
    values={}
    for name in INPUT_NAMES:
        local=metadata[name]; remote=pinned[name]
        require(type(local) is dict and set(local)=={'path','sha256'} and type(remote) is dict and set(remote)=={'path','sha256'},'Null/missing metadata is not actual acceptance')
        require(remote['sha256']==local['sha256'] and Path(remote['path']).is_absolute() and '..' not in Path(remote['path']).parts and Path(remote['path']).is_relative_to(ROOT),'Actual runtime evidence must stay in own h100-9 ROOT and match local JSON bytes')
        values[name]=read_bound(local['path'],local['sha256'])
    package=dict(code_commit=inputs['source']['code_commit'],common_data_identity=inputs['assets']['common_data_identity'],model_identity=inputs['assets']['model_identity'])
    thresholds=observed_production_inputs(package,values)
    return thresholds,values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','inputs-sha','output','wrapper-inventory','wrapper-inventory-sha'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();inputs,_,_=read_bound(a.inputs,a.inputs_sha)
    inventory=json.loads(read_regular_bound(a.wrapper_inventory,a.wrapper_inventory_sha))
    prepared=prepare_wrapper_inventory(inventory,inputs)
    source=inputs['source'];repo=Path(source['path'])
    require(source.get('pending_actual_guard_continue_interface') is False,'Latest opportunity/write/attempt API not yet aligned; old4ff contract cannot execute new user protocol')
    require(type(source['code_commit']) is str and re.fullmatch('[0-9a-f]{40}',source['code_commit']) and source['code_commit']!=SOURCE and source['source_interface_receipt_sha256'],'New actual source/API review input required; historical4ff cannot implement new count protocol')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    require(actual==source['code_commit'] and not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),'Frozen actual source must remain clean')
    require_actual_fresh_api(repo)
    interface,interface_sha,interface_raw=verify_source_interface(source,repo)
    require((repo/'search_r1/baselines/tree_guard_continue.py').is_file(),'Actual additional15th runtime/source interface module missing')
    spec=json.loads(read_regular_bound(inputs['configuration_spec_path'],inventory['wrapper_files']['CONFIGURATION_SPEC.json']))
    require(spec['trial_only'] is False and not spec['environment'] and spec['production_thresholds']=={'target_old_policy_kl':.2,'negative_tail_limit':None,'initial_actor_repeat_max_abs_delta':None},'Production spec must fix user KL0.2 and require observed other thresholds')
    thresholds,observations=verified_observation_metadata(inputs)
    overrides=configuration_overrides(spec,thresholds)
    dispatch=inputs['dispatch'];intent=dispatch['once_intent'];wall=dispatch['train_wall_hours']
    require(type(intent) is str and Path(intent).is_absolute() and '..' not in Path(intent).parts and Path(intent).is_relative_to(ROOT/'instruct300'),'Root must supply reviewed NEW production once intent; old zero intent cannot be inferred')
    from submit_once import INTENT_RE
    require(Path(intent).parent==ROOT/'instruct300' and re.fullmatch(INTENT_RE,Path(intent).name),'New production intent must match reviewed production-only once namespace')
    require(type(wall) is int and wall>0,'Explicit positive finite formal wall hours required; no default')
    require(dispatch['cluster_name']=='wuhan-h100','Explicit own local production cluster binding missing')
    require(dispatch['nodes']==['h100-9'] and dispatch['gpus']==8 and dispatch['cpus_per_task']==32 and dispatch['memory_gib']==256 and dispatch['exclusive'] is True and dispatch['no_requeue'] is True,'Immutable reservation differs')
    output=Path(a.output).resolve();require(not output.exists() and str(output).startswith('/Users/'),'Local no-clobber package only')
    output.mkdir(parents=True);(output/'source/repo').mkdir(parents=True)
    archive=subprocess.check_output(['git','archive','--format=tar',actual],cwd=repo)
    selected=materialize_source_archive(archive,output/'source/repo')
    wrappers=copy_wrapper_inventory(output,prepared)
    (output/'observed-inputs').mkdir()
    with (output/'observed-inputs/source-interface-manifest.json').open('xb') as stream:stream.write(interface_raw)
    for name,(_,_,raw) in observations.items():
        with (output/'observed-inputs'/(name+'.json')).open('xb') as stream:stream.write(raw)
    assets=inputs['assets']
    manifest=dict(status='LOCAL_TRAIN200_PACKAGE_MATERIALIZED_NOT_RELEASED',scope='LOCAL_DRAFT_NOT_RELEASED',stage='fresh_mu_train200',training_protocol=PROFILE,
        code_commit=actual,official_model_path=assets['official_model_path'],model_identity=assets['model_identity'],common_data_identity=assets['common_data_identity'],
        configuration_overrides=overrides,bootstrap_thresholds=thresholds,bootstrap_thresholds_accepted=True,production_thresholds_accepted=False,
        full_run_continuation_requires_same_instance_first5_threshold_assessment=True,
        production_inputs=inputs['production_inputs'],production_input_sha256={name:value[1] for name,value in observations.items()},
        target_kl_authority=observations['threshold_acceptance'][0]['target_kl_authority'],dispatch=dispatch,
        actual_3b_training_accepted=False,long_training_ready=False,original_sampler_actor_gate_accepted=False,
        wrapper_inventory_sha256=a.wrapper_inventory_sha,wrapper_files=wrappers,source_selection=selected,
        historical_synthetic_receipts=inputs['reference_receipts'],external_files=inputs['external_files'],
        proposed_source_interface_receipt_sha256=source['source_interface_receipt_sha256'],
        source_interface_manifest_sha256=interface_sha,source_interface_files=interface['source_files'],additional_runtime_closure=interface['additional_changed_runtime_files'],
        source_interface_manifest_is_byte_identity_not_acceptance=True,
        protocol=inputs['protocol'],actual_execution_unknown=True)
    manifest['files']={str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    with (output/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(status=manifest['status'],path=str(output),manifest_sha256=sha(output/'manifest.json'),release_created=False,slurm_submitted=False)))


if __name__=='__main__':main()
