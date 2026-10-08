"""Local-only draft builder. Never deploys, releases or submits the package.

Root must freeze actual new source/config/CPU evidence and obtain independent
review before changing LOCAL_DRAFT scope. Old recovery source/receipts refuse.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition: raise ValueError(message)


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
    args=parser.parse_args(); source_inputs=json.loads(Path(args.inputs).read_bytes())
    source=source_inputs['source']; repo=Path(source['path'] or '')
    require(source['path'] and source['code_commit'] and source['source_interface_receipt_sha256'],
            'New clean source and actual fresh API review inputs are required; null never means accepted')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    require(actual==source['code_commit'] and not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),
            'Candidate source is not clean exact commit')
    require(source['config_name']=='ppo_tree_fresh200' and (repo/'verl/trainer/config/ppo_tree_fresh200.yaml').is_file(),
            'Fresh source/config not implemented')
    require_actual_fresh_api(repo)
    configuration=json.loads(Path(source_inputs['configuration_overrides_path']).read_bytes())
    require(configuration['trial_only'] is True and configuration['production_thresholds']=={
        'target_old_policy_kl':None,'negative_tail_limit':None,'initial_actor_repeat_max_abs_delta':None},
        'Draft cannot infer production acceptance')
    assets=source_inputs['assets']; output=Path(args.output).resolve()
    require(not output.exists(), 'No-clobber new local package path required')
    require(str(output).startswith('/Users/'), 'This draft builder must not write into a remote/runtime directory')
    output.mkdir(parents=True); (output/'source/repo').mkdir(parents=True)
    archive=subprocess.check_output(['git','archive','--format=tar',actual],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as bundle:
        for member in bundle.getmembers():
            relative=Path(member.name)
            require(not relative.is_absolute() and '..' not in relative.parts and not member.issym() and not member.islnk(),
                    'Source archive contains an unsupported link/path')
        bundle.extractall(output/'source/repo')
    for name in ('supervisor.py','calibration_contract.py','launch.sbatch','CONFIGURATION_OVERRIDES.json','INPUTS.schema.json'):
        (output/name).write_bytes((HERE/name).read_bytes())
    # Runtime entry wrappers need new source hashes/fresh paths. Root supplies
    # these reviewed local bytes, never the old healthy110 observer unmodified.
    required=('run_tree.sh','validation_cost_entry.py','validation_cost_hooks_instruct300.py','sitecustomize.py')
    supplied=source_inputs.get('runtime_entries',{})
    require(all(name in supplied for name in required), 'Missing fresh reviewed entry/validation observer inputs')
    for name in required:
        entry=supplied[name]; require(sha(entry['path'])==entry['sha256'], 'Runtime entry input SHA differs: '+name)
        (output/name).write_bytes(Path(entry['path']).read_bytes())
    manifest=dict(status='LOCAL_DRAFT_PACKAGE_MATERIALIZED_NOT_RELEASED',
        scope='LOCAL_DRAFT_NOT_RELEASED',stage='fresh_mu_calibration',
        training_protocol='tree_fresh200_five_leaf_v1',code_commit=actual,
        official_model_path=assets['official_model_path'],model_identity=assets['model_identity'],
        common_data_identity=assets['common_data_identity'],
        configuration_overrides=configuration['configuration_overrides'],
        trial_thresholds=configuration['trial_thresholds'],production_thresholds_accepted=False,
        actual_3b_training_accepted=False,long_training_ready=False,
        original_sampler_actor_gate_accepted=False,
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
