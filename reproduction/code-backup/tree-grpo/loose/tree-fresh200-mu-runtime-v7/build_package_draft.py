"""Local-only draft builder. Never deploys, releases or submits the package.

Root must freeze actual new source/config/CPU evidence and obtain independent
review before changing LOCAL_DRAFT scope. Old recovery source/receipts refuse.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
WRAPPER_FILES = ('CONFIGURATION_OVERRIDES.json','INPUTS.schema.json','PLAN.md','SOURCE_CPU_RECEIPT.json',
    'build_package_draft.py','calibration_contract.py','launch.sbatch','render_dispatch.py','run_tree.sh',
    'sitecustomize.py','supervisor.py','test_runtime_contract_cpu.py','tokenizer_raw_preflight.py',
    'validation_cost_entry.py','validation_cost_hooks_instruct300.py')
RUNTIME_ENTRIES = ('run_tree.sh','validation_cost_entry.py','validation_cost_hooks_instruct300.py','sitecustomize.py')
CODE_SUFFIXES = frozenset(('.py','.sh','.sbatch','.yaml','.yml','.md','.json','.txt','.toml','.cfg','.ini','.patch'))
CODE_EXTENSIONLESS = frozenset(('LICENSE','NOTICE','.gitignore','Makefile',
                              'verl/version/version','verl/single_controller/version/version'))
SOURCE_ROOT_FILES = frozenset(('setup.py','pyproject.toml','requirements.txt',
                             'LICENSE','NOTICE','README.md','.gitignore','Makefile'))


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
    """Verify the actual fifteen-file review inventory, including four entries."""
    require(type(inventory.get('wrapper_files')) is dict and
            set(inventory['wrapper_files'])==set(WRAPPER_FILES), 'Reviewed wrapper inventory must contain all15 fixed files')
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
    runtime_path = name in SOURCE_ROOT_FILES or name.startswith(('search_r1/', 'verl/'))
    return runtime_path and (name in CODE_EXTENSIONLESS or relative.suffix.lower() in CODE_SUFFIXES)


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
    return dict(protocol='runtime_review_code_metadata_allowlist_v2',
                allowed_suffixes=sorted(CODE_SUFFIXES),allowed_extensionless=sorted(CODE_EXTENSIONLESS),
                allowed_source_prefixes=['search_r1/','verl/'],allowed_source_root_files=sorted(SOURCE_ROOT_FILES),
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--wrapper-inventory',required=True);parser.add_argument('--wrapper-inventory-sha',required=True)
    args=parser.parse_args(); source_inputs=json.loads(Path(args.inputs).read_bytes())
    inventory=json.loads(read_regular_bound(args.wrapper_inventory,args.wrapper_inventory_sha))
    prepared=prepare_wrapper_inventory(inventory,source_inputs)
    source=source_inputs['source']; repo=Path(source['path'] or '')
    require(source['path'] and source['code_commit'] and source['source_interface_receipt_sha256'],
            'New clean source and actual fresh API review inputs are required; null never means accepted')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    require(actual==source['code_commit'] and not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),
            'Candidate source is not clean exact commit')
    require(source['config_name']=='ppo_tree_fresh200' and (repo/'verl/trainer/config/ppo_tree_fresh200.yaml').is_file(),
            'Fresh source/config not implemented')
    require_actual_fresh_api(repo)
    configuration=json.loads(read_regular_bound(source_inputs['configuration_overrides_path'],
        inventory['wrapper_files']['CONFIGURATION_OVERRIDES.json']))
    require(configuration['trial_only'] is True and configuration['production_thresholds']=={
        'target_old_policy_kl':None,'negative_tail_limit':None,'initial_actor_repeat_max_abs_delta':None},
        'Draft cannot infer production acceptance')
    assets=source_inputs['assets']; output=Path(args.output).resolve()
    require(not output.exists(), 'No-clobber new local package path required')
    require(str(output).startswith('/Users/'), 'This draft builder must not write into a remote/runtime directory')
    output.mkdir(parents=True); (output/'source/repo').mkdir(parents=True)
    archive=subprocess.check_output(['git','archive','--format=tar',actual],cwd=repo)
    source_selection=materialize_source_archive(archive,output/'source/repo')
    copied_wrapper_files=copy_wrapper_inventory(output,prepared)
    manifest=dict(status='LOCAL_DRAFT_PACKAGE_MATERIALIZED_NOT_RELEASED',
        scope='LOCAL_DRAFT_NOT_RELEASED',stage='fresh_mu_calibration',
        training_protocol='tree_fresh200_five_leaf_v1',code_commit=actual,
        official_model_path=assets['official_model_path'],model_identity=assets['model_identity'],
        common_data_identity=assets['common_data_identity'],
        configuration_overrides=configuration['configuration_overrides'],
        trial_thresholds=configuration['trial_thresholds'],production_thresholds_accepted=False,
        actual_3b_training_accepted=False,long_training_ready=False,
        original_sampler_actor_gate_accepted=False,
        wrapper_inventory_sha256=args.wrapper_inventory_sha,wrapper_files=copied_wrapper_files,
        source_selection=source_selection,
        probability_calibration=dict(environment=configuration['environment'],source_interface_reviewed=False,
            proposed_source_interface_receipt_sha256=source['source_interface_receipt_sha256'],
            ready_relative_path='output/probability-calibration/ready.json',
            ready_status='FRESH_BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES'),
        historical_synthetic_receipts=source_inputs['reference_receipts'],
        external_files=source_inputs['external_files'])
    manifest['files']={str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=manifest['status'],path=str(output),manifest_sha256=sha(output/'manifest.json'),
                         release_created=False,slurm_submitted=False)))


if __name__=='__main__': main()
