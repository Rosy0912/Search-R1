"""One immutable V8 wrapper repair of real3330 duplicate Ray.options startup fault.

No tests/reviews/core imports/source edits/remote calls or submission.
"""
import argparse,copy,datetime,hashlib,json,shlex,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE/'package-code-only-v7';PACKAGE=HERE/'package-code-only-v8'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');PARENT=ROOT/'instruct300'
SOURCE='9693f0c846fab483a03e1d70ce1109aeddc5e118'
OLD_SHA='5f7071c0c6886b3a5b88abd216cd3b18bc0b14a1fd6293418717d742ece4ccff'
MODE='HUMAN_AUTHORIZED_DIRECT_TRAIN200';SCOPE='CURRENT_FRESH200_CHAIN_CONTINUATION'
CANDIDATE='current-fresh200-chain-mask-v8-9693f0c'
INTENT='SUBMISSION_ONCE-current-fresh200-chain-mask-v8-9693f0c.json'
AUTHORITY='CURRENT_FRESH200_CHAIN_HUMAN_AUTHORITY-V8-9693f0c.json'
RELEASE='CURRENT_FRESH200_CHAIN_RELEASE-V8-9693f0c.json'
DISPATCH='CURRENT_FRESH200_CHAIN_DISPATCH-V8-9693f0c.sbatch'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def write_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def render(package,manifest_sha,release_path,release_sha,wall_hours):
    return '\n'.join(('#!/usr/bin/env bash','#SBATCH --job-name=tree-fresh200-train',
        '#SBATCH --nodes=1','#SBATCH --nodelist=h100-9','#SBATCH --gres=gpu:8',
        '#SBATCH --cpus-per-task=32','#SBATCH --mem=256G','#SBATCH --exclusive',
        f'#SBATCH --time={wall_hours}:00:00','#SBATCH --no-requeue','set -euo pipefail',
        'export TREE_INSTRUCT_TOOLS='+shlex.quote(str(package)),
        'export TREE_INSTRUCT_PACKAGE_SHA256='+shlex.quote(manifest_sha),
        'export TREE_PRODUCTION_RELEASE_PATH='+shlex.quote(str(release_path)),
        'export TREE_INSTRUCT_REVIEW_SHA256='+shlex.quote(release_sha),
        'exec bash '+shlex.quote(str(package/'launch.sbatch')),''))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('predecessor-proof-result','predecessor-proof-result-sha','predecessor-proof-sha'):p.add_argument('--'+name,required=True)
    a=p.parse_args();require(not (PACKAGE/'manifest.json').exists(),'V8 already finalized, never overwrite/retry')
    require(sha(OLD/'manifest.json')==OLD_SHA,'Failed V7 original manifest changed')
    old=json.loads((OLD/'manifest.json').read_text())
    require(old['code_commit']==SOURCE and old['packet_revision']==7 and old['scope']==SCOPE,'Actual3330 parent packet differs')
    require(sha(a.predecessor_proof_result)==a.predecessor_proof_result_sha,'Root native3330 proof metadata differs')
    result=json.loads(Path(a.predecessor_proof_result).read_text());native=result['proof']
    require(result['proof_sha256']==a.predecessor_proof_sha and native['job_id']==3330 and native['code_commit']==SOURCE
        and native['status']=='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP' and native['scheduler_state']=='FAILED'
        and native['scheduler_exit_code']=='1:0' and native['cleanup_residual']=={}
        and native['original_eight_pid_start_times_preserved'] is True,
        'Exact native3330 positive terminal/cleanup required')
    require(native['run_result']['sha256']=='354324b39e0bf5cd9ca7c6fdc758a5e681081b650743b13562a045aeca45e4b2'
        and native['cost']['sha256']=='a74e23582aa65ded0f153b90c4fc3a0fd9588fe2149eab5acb476365bbe842ae'
        and native['native_optimizer_committed'] is None and native['native_scheduler_advanced'] is None
        and native['actual_migration_receipt_present'] is False and native['child_training_current_counts'] is None,
        'Native missing receipts/nulls must not become zero write counts')
    files={str(path.relative_to(PACKAGE)):sha(path) for path in sorted(PACKAGE.rglob('*')) if path.is_file()}
    require(set(files)==set(old['files']),'V8 one-line repair/revision pins cannot change members')
    source_files={name.removeprefix('source/repo/'):pin for name,pin in files.items() if name.startswith('source/repo/')}
    require(source_files==old['source_selection']['selected_files'],'Frozen9693 core bytes must remain unchanged')
    changed={name:pin for name,pin in files.items() if pin!=old['files'][name]}
    require(set(changed)=={'current_chain_runtime.py','production_contract.py','owned_lease_identity.py','launch.sbatch','supervisor.py','submit_once.py','validation_cost_entry.py'},
        'Only exactrevision8 wrappers, actual3330 closure and RemoteFunction fault repair may differ')
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    delta=dict(code_commit=SOURCE,base_code_commit=SOURCE,packet_revision=8,base_manifest_sha256=OLD_SHA,
        core_source_bytes_changed=False,actual_fault="FAILED3330 before migration: 'FuncWrapper' object has no attribute 'options'",
        correction='Keep measured_remote original RemoteFunction; baseline main.options applies full driver audit environment once; worker explicit3 environment patch retained',
        changed_wrapper_files_sha256=changed,new_tests_executed=False,new_reviews_executed=False,
        current_chain3328_rolling2_and_all_training_parameters_unchanged=True,
        resource_guards_and_new_run_prefix_unchanged=True)
    delta_path=HERE/'WRAPPER_FIX_DELTA_V8.json';write_json(delta_path,delta)
    proofs=copy.deepcopy(old['terminal_proof_closure']);proofs['3330']=dict(path=result['proof_path'],sha256=a.predecessor_proof_sha)
    sources=copy.deepcopy(old['historical_code_commit_by_job']);sources['3330']=SOURCE
    fix=dict(code_commit=SOURCE,base_source_commit=SOURCE,core_source_bytes_changed=False,
        wrapper_delta_metadata_sha256=sha(delta_path),changed_wrapper_files_sha256=changed,
        actual_fault=delta['actual_fault'],new_tests_executed=False,new_reviews_executed=False)
    authority=copy.deepcopy(json.loads((HERE/'CURRENT_FRESH200_CHAIN_HUMAN_AUTHORITY-V7-9693f0c.json').read_text()))
    authority['historical_root_fixed_core_files_sha256']=authority.pop('root_fixed_core_files_sha256',{})
    authority.update(packet_revision=8,core_source_bytes_changed=False,real_error_fix=delta['actual_fault'],
        root_fixed_core_files_sha256={},root_fixed_wrapper_files_sha256=changed,
        error_fix_observed_at_utc=observed,new_tests_executed=False,new_reviews_executed=False)
    authority_path=HERE/AUTHORITY;write_json(authority_path,authority)
    manifest=copy.deepcopy(old)
    manifest['historical_core_error_fix']=copy.deepcopy(old['root_actual_error_fix'])
    manifest.update(packet_revision=8,materialized_at_utc=observed,
        human_training_authority=dict(path=str(PARENT/AUTHORITY),sha256=sha(authority_path)),
        supersedes_failed_packet_manifest_sha256=OLD_SHA,predecessor_terminal_cleanup_proof=proofs['3330'],
        terminal_proof_closure=proofs,historical_code_commit_by_job=sources,core_source_bytes_changed=False,
        root_actual_error_fix=fix,root_actual_wrapper_error_fix=fix,new_tests_executed=False,new_cpu_tests_executed=False,new_reviews_executed=False,
        wrapper_inventory_scope='Exact V8 wrapper bytes bind actual startup-error repair and revision8; no new tests/review')
    manifest['dispatch']['once_intent']=str(PARENT/INTENT);manifest['files']=files
    manifest['wrapper_files']={name:pin for name,pin in files.items() if not name.startswith('source/')}
    manifest['source_selection'].update(inherited_unchanged_from_manifest_sha256=OLD_SHA,changed_source_files_sha256={})
    manifest['native_rolling_save']['wrapper_files_sha256']={name:files[name] for name in ('rolling_training_state_hooks.py','validation_cost_entry.py')}
    manifest['previous_failed_direct_runs'].append(dict(job_id=3330,status='FAILED',scheduler_exit_code='1:0',code_commit=SOURCE,
        native_proof=proofs['3330'],run_result_sha256=native['run_result']['sha256'],cost_sha256=native['cost']['sha256'],
        once_journal_sha256=native['once_journal']['sha256'],actual_optimizer_committed=None,scheduler_advanced=None,
        failed_unrecorded_partial_optimizer_writes_unknown=True,actual_migration_receipt_present=False,
        actual_initialization_receipt_present=False,child_training_current_counts=None,checkpoint_inheritance=False,
        scope='Native startup nulls remain null; no migration/training receipt means no new current-chain count inferred'))
    manifest_path=PACKAGE/'manifest.json';write_json(manifest_path,manifest);manifest_sha=sha(manifest_path)
    release=copy.deepcopy(json.loads((HERE/'CURRENT_FRESH200_CHAIN_RELEASE-V7-9693f0c.json').read_text()))
    release.update(packet_revision=8,manifest_sha256=manifest_sha,human_training_authority_sha256=sha(authority_path),
        dispatch_intent_path=manifest['dispatch']['once_intent'],dispatch_renderer_sha256=files['render_dispatch.py'],
        dispatch_submitter_sha256=files['submit_once.py'],predecessor_terminal_cleanup_proof=proofs['3330'],
        terminal_proof_closure=proofs,historical_code_commit_by_job=sources,core_source_bytes_changed=False,
        root_actual_error_fix=fix,root_actual_wrapper_error_fix=fix,actual_error_fix=delta['actual_fault'],
        native_rolling_save=manifest['native_rolling_save'],new_tests_executed=False,new_reviews_executed=False)
    release_path=HERE/RELEASE;write_json(release_path,release)
    remote=PARENT/'candidates'/CANDIDATE;dispatch_path=HERE/DISPATCH
    with dispatch_path.open('x') as stream:stream.write(render(remote,manifest_sha,PARENT/RELEASE,sha(release_path),168))
    argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONOPTIMIZE=0','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1',
        str(ROOT/'env/bin/python'),'-B',str(remote/'submit_once.py'),'--package',str(remote),'--manifest-sha',manifest_sha,
        '--release-path',str(PARENT/RELEASE),'--release-sha',sha(release_path),'--script',str(PARENT/DISPATCH),'--script-sha',sha(dispatch_path),
        '--predecessor-proof',proofs['3330']['path'],'--predecessor-proof-sha',a.predecessor_proof_sha]
    archive=HERE/(CANDIDATE+'-code-only.tar.gz');require(not archive.exists(),'V8 archive must be new')
    with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as bundle:
        for name in sorted([*files,'manifest.json']):bundle.add(PACKAGE/name,arcname='candidates/'+CANDIDATE+'/'+name,recursive=False)
        for path in (authority_path,release_path,dispatch_path):bundle.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='CODE_ONLY_CURRENT_FRESH200_CHAIN_V8_PACKET_MATERIALIZED_NOT_SUBMITTED',packet_revision=8,
        execution_mode=MODE,scope=SCOPE,code_commit=SOURCE,current_chain_resume=manifest['current_chain_resume'],
        local_package=str(PACKAGE),remote_extract_destination=str(PARENT),remote_package=str(remote),archive=str(archive),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,manifest_sha256=manifest_sha,
        authority_path=str(authority_path),authority_sha256=sha(authority_path),release_path=str(release_path),release_sha256=sha(release_path),
        dispatch_path=str(dispatch_path),dispatch_sha256=sha(dispatch_path),once_submission_argv=argv,
        once_submission_shell_command=shlex.join(argv),once_submission_management_timeout_seconds=120,
        once_intent=manifest['dispatch']['once_intent'],terminal_proof_closure=proofs,historical_code_commit_by_job=sources,
        changed_wrapper_files_sha256=changed,wrapper_delta_path=str(delta_path),wrapper_delta_sha256=sha(delta_path),
        core_source_bytes_changed=False,actual_slurm_submitted=False,actual_migration_executed=False,exact_executor_resume=False,
        parent_completed_outer=2,planned_new_completed_outer=198,contains_model_data_index_checkpoint_assets=False,
        new_tests_executed=False,new_reviews_executed=False,production_thresholds_accepted=False,
        original_sampler_actor_gate_accepted=False,training_accepted=False,independent_acceptance=False,
        postload_held_ofd_identity_and_foreign15_percent_guards_preserved=True)
    write_json(HERE/'CURRENT_CHAIN_PACKET_BOUND_ARGV_V8.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
