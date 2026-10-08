"""Finalize one V6 packet for the observed pre-actor wrapper revision fault.

Only stdlib assembly; source and rolling wrappers are unchanged. No tests,
review, source-module imports, GPU/remote calls or submission.
"""
import argparse,copy,datetime,hashlib,json,shlex,tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
V5=HERE/'package-code-only-v5';PACKAGE=HERE/'package-code-only-v6'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');PARENT=ROOT/'instruct300'
SOURCE='4a65e9a3ab87e84e1e9dd283b664cdef3a224c8a'
V5_MANIFEST_SHA='cda27d64b7114417f64cf80f23772b288cdbc7ec9ef2ce9e7c86338e08dbf1c9'
AUTHORITY_NAME='DIRECT_TRAIN200_HUMAN_AUTHORITY-V6.json'
RELEASE_NAME='DIRECT_TRAIN200_RELEASE-V6.json'
DISPATCH_NAME='DIRECT_TRAIN200_DISPATCH-V6.sbatch'
CANDIDATE_NAME='fresh200-direct-train-4a65e9a-v6'
INTENT_NAME='SUBMISSION_ONCE-fresh200-direct-4a65e9a-v6.json'
ROLLING_PINS={'rolling_training_state_hooks.py':'5d42585e9b355ccd2aa85cc1d47160e0ae1a831afa58c5645e24c7cc6363aa85',
    'validation_cost_entry.py':'fb993d920d7cab3ef77d63cb4f3eedef7a3e9260b588aec3eb37045567146572'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def render(package,manifest_sha,release_path,release_sha,wall_hours):
    # Exact existing renderer template, without importing a package module.
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
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('predecessor-proof-result','predecessor-proof-result-sha','predecessor-proof-sha'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    require(not (PACKAGE/'manifest.json').exists(),'V6 already finalized; cannot overwrite final packet')
    require(sha(V5/'manifest.json')==V5_MANIFEST_SHA,'Actual failed V5 manifest changed')
    require(sha(args.predecessor_proof_result)==args.predecessor_proof_result_sha,'Exact root3327 proof metadata changed')
    proof_result=json.loads(Path(args.predecessor_proof_result).read_text());native=proof_result['proof']
    require(proof_result['proof_sha256']==args.predecessor_proof_sha and native['job_id']==3327
        and native['code_commit']==SOURCE and native['status']=='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP'
        and native['scheduler_state']=='FAILED' and native['scheduler_exit_code']=='1:0'
        and native['cleanup_residual']=={} and native['original_eight_pid_start_times_preserved'] is True,
        'Actual3327 own positive terminal/source/cleanup proof differs')
    require(native['run_result']['sha256']=='7d197727d8f4e58d2c9cd90752b66ce5881f61b3e6c96a3819c8e14f5bb2bd08',
        'Actual3327 native result bytes differ')
    old=json.loads((V5/'manifest.json').read_text())
    files={str(path.relative_to(PACKAGE)):sha(path) for path in sorted(PACKAGE.rglob('*')) if path.is_file()}
    require(all('__pycache__' not in Path(name).parts and Path(name).suffix!='.pyc' for name in files),'No import caches allowed')
    require(set(files)==set(old['files']),'V6 wrapper repair cannot add/remove package members')
    wrapper_changed={name:pin for name,pin in files.items() if pin!=old['files'][name]}
    require(set(wrapper_changed)=={'owned_lease_identity.py','submit_once.py'},'Only lease revision coupling and terminal closure wrappers may change')
    require(all(files[name]==pin for name,pin in ROLLING_PINS.items()),'Original rolling patch bytes must stay unchanged')
    source_files={name.removeprefix('source/repo/'):pin for name,pin in files.items() if name.startswith('source/repo/')}
    require(source_files==old['source_selection']['selected_files'],'Frozen core source4a65 must remain byte-identical')
    lease=(PACKAGE/'owned_lease_identity.py').read_text()
    require("package['packet_revision']==6" in lease and 'restricted to this direct V6 package' in lease,
        'Lease wrapper must remain restricted to exact current V6 revision')
    delta=dict(code_commit=SOURCE,base_code_commit=SOURCE,packet_revision=6,
        base_manifest_sha256=V5_MANIFEST_SHA,core_source_bytes_changed=False,
        changed_wrapper_files={name:dict(path=str(PACKAGE/name),sha256=pin,prior_sha256=old['files'][name]) for name,pin in wrapper_changed.items()},
        actual_fault='FAILED3327 before Ray/actor: owned_lease_identity demanded packet_revision4 from V5 packet5',
        correction='Pin owned lease helper to exactly packet_revision6 and V6 message; add actual3327 terminal closure and exact V5 journal',
        ofd_uuid_cgroup_and_foreign15_percent_guards_unchanged=True,
        native_rolling_wrapper_bytes_unchanged=True,new_tests_executed=False,new_reviews_executed=False)
    delta_path=HERE/'WRAPPER_FIX_DELTA_V6.json';write_json(delta_path,delta)
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    proofs=copy.deepcopy(old['terminal_proof_closure'])
    proofs['3327']=dict(path=proof_result['proof_path'],sha256=args.predecessor_proof_sha)
    source_by_job=copy.deepcopy(old['historical_code_commit_by_job']);source_by_job['3327']=SOURCE
    fix=dict(code_commit=SOURCE,base_source_commit=SOURCE,core_source_bytes_changed=False,
        wrapper_delta_metadata_sha256=sha(delta_path),changed_wrapper_files_sha256=wrapper_changed,
        actual_fault=delta['actual_fault'],new_tests_executed=False,new_reviews_executed=False)
    authority=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_HUMAN_AUTHORITY-V5.json').read_text()))
    authority['historical_root_fixed_core_files_sha256']=authority.pop('root_fixed_core_files_sha256',{})
    authority.update(packet_revision=6,code_commit=SOURCE,core_source_bytes_changed=False,
        real_error_fix=delta['actual_fault'],root_fixed_core_files_sha256={},
        root_fixed_wrapper_files_sha256=wrapper_changed,error_fix_observed_at_utc=observed,
        new_tests_executed=False,new_reviews_executed=False)
    authority_path=HERE/AUTHORITY_NAME;write_json(authority_path,authority)
    manifest=copy.deepcopy(old)
    manifest['historical_core_error_fix']=copy.deepcopy(old['root_actual_error_fix'])
    manifest.update(packet_revision=6,code_commit=SOURCE,materialized_at_utc=observed,
        human_training_authority=dict(path=str(PARENT/AUTHORITY_NAME),sha256=sha(authority_path)),
        supersedes_failed_packet_manifest_sha256=V5_MANIFEST_SHA,
        predecessor_terminal_cleanup_proof=proofs['3327'],terminal_proof_closure=proofs,
        historical_code_commit_by_job=source_by_job,core_source_bytes_changed=False,
        root_actual_error_fix=fix,root_actual_wrapper_error_fix=fix,
        new_tests_executed=False,new_cpu_tests_executed=False,new_reviews_executed=False,
        wrapper_inventory_scope='Inherited historical inventory identifier; exact V6 files are directly byte-bound without new tests or review')
    manifest['dispatch']['once_intent']=str(PARENT/INTENT_NAME)
    manifest['files']=files
    manifest['wrapper_files']={name:pin for name,pin in files.items() if not name.startswith('source/')}
    manifest['source_selection']['inherited_unchanged_from_manifest_sha256']=V5_MANIFEST_SHA
    manifest['source_selection']['changed_source_files_sha256']={}
    manifest['previous_failed_direct_runs'].append(dict(job_id=3327,status='FAILED',scheduler_exit_code='1:0',code_commit=SOURCE,
        native_proof=proofs['3327'],run_result_sha256=native['run_result']['sha256'],cost_sha256=native['cost']['sha256'],
        once_journal_sha256=native['once_journal']['sha256'],actual_optimizer_committed=None,scheduler_advanced=None,
        failed_unrecorded_partial_optimizer_writes_unknown=True,checkpoint_inheritance=False,
        pre_actor_wrapper_version_fault=True,
        scope='Native null fields remain null; actual wrapper startup fault before Ray/actor does not create completed credit or inherited state'))
    manifest_path=PACKAGE/'manifest.json';write_json(manifest_path,manifest);manifest_sha=sha(manifest_path)
    release=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_RELEASE-V5.json').read_text()))
    release.update(packet_revision=6,code_commit=SOURCE,manifest_sha256=manifest_sha,
        human_training_authority_sha256=sha(authority_path),dispatch_intent_path=manifest['dispatch']['once_intent'],
        dispatch_renderer_sha256=files['render_dispatch.py'],dispatch_submitter_sha256=files['submit_once.py'],
        predecessor_terminal_cleanup_proof=proofs['3327'],terminal_proof_closure=proofs,
        historical_code_commit_by_job=source_by_job,core_source_bytes_changed=False,
        actual_error_fix=delta['actual_fault'],root_actual_wrapper_error_fix=fix,
        new_tests_executed=False,new_reviews_executed=False)
    release_path=HERE/RELEASE_NAME;write_json(release_path,release)
    remote_package=PARENT/'candidates'/CANDIDATE_NAME;dispatch_path=HERE/DISPATCH_NAME
    with dispatch_path.open('x') as stream:stream.write(render(remote_package,manifest_sha,PARENT/RELEASE_NAME,sha(release_path),168))
    argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONOPTIMIZE=0','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1',
        str(ROOT/'env/bin/python'),'-B',str(remote_package/'submit_once.py'),
        '--package',str(remote_package),'--manifest-sha',manifest_sha,
        '--release-path',str(PARENT/RELEASE_NAME),'--release-sha',sha(release_path),
        '--script',str(PARENT/DISPATCH_NAME),'--script-sha',sha(dispatch_path),
        '--predecessor-proof',proofs['3327']['path'],'--predecessor-proof-sha',args.predecessor_proof_sha]
    archive=HERE/(CANDIDATE_NAME+'-code-only.tar.gz');require(not archive.exists(),'V6 archive must be new')
    with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as bundle:
        for relative in sorted([*files,'manifest.json']):bundle.add(PACKAGE/relative,arcname='candidates/'+CANDIDATE_NAME+'/'+relative,recursive=False)
        for path in (authority_path,release_path,dispatch_path):bundle.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='CODE_ONLY_DIRECT200_V6_PACKET_MATERIALIZED_NOT_SUBMITTED',packet_revision=6,code_commit=SOURCE,
        local_package=str(PACKAGE),remote_extract_destination=str(PARENT),remote_package=str(remote_package),
        archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=manifest_sha,authority_path=str(authority_path),authority_sha256=sha(authority_path),
        release_path=str(release_path),release_sha256=sha(release_path),dispatch_path=str(dispatch_path),dispatch_sha256=sha(dispatch_path),
        once_submission_argv=argv,once_submission_shell_command=shlex.join(argv),once_submission_management_timeout_seconds=120,
        once_intent=manifest['dispatch']['once_intent'],terminal_proof_closure=proofs,historical_code_commit_by_job=source_by_job,
        core_source_bytes_changed=False,changed_wrapper_files_sha256=wrapper_changed,
        wrapper_delta_path=str(delta_path),wrapper_delta_sha256=sha(delta_path),native_rolling_save=manifest['native_rolling_save'],
        new_tests_executed=False,new_reviews_executed=False,actual_slurm_submitted=False,
        contains_model_data_index_checkpoint_assets=False,retained_v5_failed_run_evidence=True,
        fresh_independent_200=True,resume=False,postload_held_ofd_identity_and_foreign15_percent_guards_preserved=True)
    write_json(HERE/'DIRECT_PACKET_BOUND_ARGV_V6.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
