"""Finalize one fresh V5 code-only direct200 packet from exact actual fault pins.

Stdlib assembly only: no source/wrapper imports, tests, review, remote calls,
GPU calls or submission. Prior failed packages and evidence are retained.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import tarfile

HERE=Path(__file__).resolve().parent
V4=HERE/'package-code-only-v4'
PACKAGE=HERE/'package-code-only-v5'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PARENT=ROOT/'instruct300'
SOURCE='4a65e9a3ab87e84e1e9dd283b664cdef3a224c8a'
PRIOR_SOURCE='2e26b5505ce8ddfad8b627bba74337fbf4e23373'
BASE_SOURCE='a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98'
AUTHORITY_NAME='DIRECT_TRAIN200_HUMAN_AUTHORITY-V5.json'
RELEASE_NAME='DIRECT_TRAIN200_RELEASE-V5.json'
DISPATCH_NAME='DIRECT_TRAIN200_DISPATCH-V5.sbatch'
CANDIDATE_NAME='fresh200-direct-train-4a65e9a-v5'
INTENT_NAME='SUBMISSION_ONCE-fresh200-direct-4a65e9a-v5.json'
PROOF_SHA='9d4a4754d5364affd930ede6205fa419c63102fbbcbb7b31f935cbaa0d83696b'
PATCH_PINS={'rolling_training_state_hooks.py':'5d42585e9b355ccd2aa85cc1d47160e0ae1a831afa58c5645e24c7cc6363aa85',
    'validation_cost_entry.py':'fb993d920d7cab3ef77d63cb4f3eedef7a3e9260b588aec3eb37045567146572'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def require(condition,message):
    if not condition:raise ValueError(message)

def render(package,manifest_sha,release_path,release_sha,wall_hours):
    # Exact existing render_dispatch.py template; no candidate module is loaded.
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
    parser.add_argument('--source-delta-sha',required=True)
    parser.add_argument('--predecessor-proof-result',required=True)
    parser.add_argument('--predecessor-proof-result-sha',required=True)
    args=parser.parse_args()
    require(not (PACKAGE/'manifest.json').exists(),'V5 already finalized; never rebuild or overwrite final packet')
    require(sha(V4/'manifest.json')=='b9e7de77b1d1e168dd02af2f3be474af7d0dd2a510d13e4ee812fe6d6c4b9108','Actual failed V4 base differs')
    delta_path=HERE/'CORE_FIX_DELTA_V5.json'
    require(sha(delta_path)==args.source_delta_sha,'Exact V5 source delta metadata changed')
    delta=json.loads(delta_path.read_text())
    require(delta['code_commit']==SOURCE and delta['base_code_commit']==PRIOR_SOURCE
        and set(delta['files'])=={'search_r1/baselines/tree_raw_tokens.py'},'V5 must contain only actual UTF-8 core delta')
    changed={}
    for name,pin in delta['files'].items():
        require(sha(pin['path'])==pin['sha256']==sha(PACKAGE/'source/repo'/name),'V5 fixed core bytes changed')
        changed[name]=pin['sha256']
    require(sha(PACKAGE/'source/repo/search_r1/baselines/policy_safety_runtime.py')=='d0f4cf7166056b2292d8b9f8faea71bc078df4cfa54406eea415a52739f37e84','Retained JSON publication repair changed')
    require(all(sha(PACKAGE/name)==pin for name,pin in PATCH_PINS.items()),'Exact owner native rolling-save wrappers changed')
    require(sha(args.predecessor_proof_result)==args.predecessor_proof_result_sha,'Actual native proof result metadata changed')
    proof_result=json.loads(Path(args.predecessor_proof_result).read_text());native=proof_result['proof']
    require(proof_result['proof_sha256']==PROOF_SHA and native['job_id']==3326 and native['code_commit']==PRIOR_SOURCE
        and native['status']=='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP' and native['cleanup_residual']=={}
        and native['original_eight_pid_start_times_preserved'] is True,'Root actual3326 positive terminal cleanup proof differs')
    require(native['model_optimizer_scheduler_checkpoint_available'] is False,'V5 must not imply nonexistent3326 resume checkpoint')
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    old=json.loads((V4/'manifest.json').read_text())
    proofs=copy.deepcopy(old['terminal_proof_closure'])
    proofs['3326']=dict(path=proof_result['proof_path'],sha256=PROOF_SHA)
    source_by_job={'3319':BASE_SOURCE,'3321':BASE_SOURCE,'3322':BASE_SOURCE,'3326':PRIOR_SOURCE}
    rolling=dict(configuration_kind='prospective_native_completed_boundary_save',wrapper_files_sha256=PATCH_PINS,
        after_every_completed_outer=True,keep_latest_completed_boundaries=2,
        native_api='save_training_state(..., verify=False) plus driver_snapshot',
        permanent_milestones_unchanged=[5,50,100,150,200],
        hf_permanent_saves_unchanged=True,automatic_resume_supported=False,exact_executor_resume_verified=False,
        source_core_save_bytes_unchanged=True,new_tests_executed=False,new_reviews_executed=False)
    authority=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_HUMAN_AUTHORITY-V4.json').read_text()))
    authority.update(packet_revision=5,code_commit=SOURCE,core_source_bytes_changed=True,
        real_error_fix='Actual FAILED3326 second rollout invalid native generated UTF-8 token handling',
        root_fixed_core_files_sha256=changed,error_fix_observed_at_utc=observed,
        native_rolling_save=rolling,fresh_independent_200=True,resume=False,
        expected_initialized_completed_outer=0,prior_credited_outer_not_inherited=1,
        prior_model_optimizer_scheduler_checkpoint_available=False,new_tests_executed=False,new_reviews_executed=False)
    authority_path=HERE/AUTHORITY_NAME;write_json(authority_path,authority)
    manifest=copy.deepcopy(old)
    manifest.update(packet_revision=5,code_commit=SOURCE,materialized_at_utc=observed,
        human_training_authority=dict(path=str(PARENT/AUTHORITY_NAME),sha256=sha(authority_path)),
        supersedes_failed_packet_manifest_sha256=sha(V4/'manifest.json'),
        predecessor_terminal_cleanup_proof=proofs['3326'],terminal_proof_closure=proofs,
        historical_code_commit_by_job=source_by_job,native_rolling_save=rolling,
        fresh_independent_200=True,resume=False,expected_initialized_completed_outer=0,
        historical_completed_outer_not_inherited=1,prior_model_optimizer_scheduler_checkpoint_available=False,
        new_tests_executed=False,new_cpu_tests_executed=False,new_reviews_executed=False,
        wrapper_inventory_scope='Inherited historical inventory identifier; exact V5 files are directly byte-bound without new tests or review',
        root_actual_error_fix=dict(code_commit=SOURCE,base_source_commit=PRIOR_SOURCE,
            source_delta_metadata_sha256=args.source_delta_sha,changed_source_files_sha256=changed,
            retained_json_publication_fix=delta['retained_json_publication_fix'],
            new_tests_executed=False,new_reviews_executed=False))
    previous=manifest['previous_failed_direct_runs']
    for record in previous:
        job=str(record['job_id']);record['code_commit']=source_by_job[job]
        record['native_proof']=proofs[job]
    previous.append(dict(job_id=3326,status='FAILED',scheduler_exit_code='1:0',code_commit=PRIOR_SOURCE,
        native_proof=proofs['3326'],run_result_sha256=native['run_result']['sha256'],cost_sha256=native['cost']['sha256'],
        once_journal_sha256=native['once_journal']['sha256'],actual_optimizer_committed=None,scheduler_advanced=None,
        failed_unrecorded_partial_optimizer_writes_unknown=True,
        direct_completed_outer_observation=native['direct_completed_outer_observation'],
        model_optimizer_scheduler_checkpoint_available=False,checkpoint_inheritance=False,
        scope='Native run-result null fields stay null; real historical attempt1 credited outer is separate, and V5 starts a new official-base 200 without splicing 1 plus 199'))
    manifest['dispatch']['once_intent']=str(PARENT/INTENT_NAME)
    files={str(path.relative_to(PACKAGE)):sha(path) for path in sorted(PACKAGE.rglob('*')) if path.is_file()}
    require(all('__pycache__' not in Path(name).parts and Path(name).suffix!='.pyc' for name in files),'Import caches cannot enter V5 code-only archive')
    manifest['files']=files
    manifest['wrapper_files']={name:value for name,value in files.items() if not name.startswith('source/')}
    source_files={name.removeprefix('source/repo/'):value for name,value in files.items() if name.startswith('source/repo/')}
    old_source=old['source_selection']['selected_files']
    require(set(source_files)==set(old_source) and {name for name in source_files if source_files[name]!=old_source[name]}==set(changed),'V5 source selection differs beyond single actual UTF-8 fix')
    manifest['source_selection']=dict(protocol='actual_2e26_json_fixed_code_base_plus_native_utf8_fix_v5',base_source_commit=PRIOR_SOURCE,
        original_a18_code_base=BASE_SOURCE,code_commit=SOURCE,selected_files=source_files,selected_count=len(source_files),
        changed_source_files_sha256=changed,retained_json_publication_fix=delta['retained_json_publication_fix'],
        not_a_full_repository_asset_copy=True,contains_model_data_index_checkpoint_assets=False)
    manifest_path=PACKAGE/'manifest.json';write_json(manifest_path,manifest);manifest_sha=sha(manifest_path)
    release=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_RELEASE-V4.json').read_text()))
    release.update(packet_revision=5,code_commit=SOURCE,manifest_sha256=manifest_sha,
        human_training_authority_sha256=sha(authority_path),dispatch_intent_path=manifest['dispatch']['once_intent'],
        dispatch_renderer_sha256=files['render_dispatch.py'],dispatch_submitter_sha256=files['submit_once.py'],
        predecessor_terminal_cleanup_proof=proofs['3326'],terminal_proof_closure=proofs,
        historical_code_commit_by_job=source_by_job,native_rolling_save=rolling,
        actual_error_fix='Actual FAILED3326 native generated UTF-8 failure fixed; prior JSON publication fix retained',
        fresh_independent_200=True,resume=False,historical_completed_outer_not_inherited=1,
        new_tests_executed=False,new_reviews_executed=False)
    release_path=HERE/RELEASE_NAME;write_json(release_path,release)
    remote_package=PARENT/'candidates'/CANDIDATE_NAME
    dispatch_path=HERE/DISPATCH_NAME
    with dispatch_path.open('x') as stream:stream.write(render(remote_package,manifest_sha,PARENT/RELEASE_NAME,sha(release_path),168))
    argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONOPTIMIZE=0','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1',
        str(ROOT/'env/bin/python'),'-B',str(remote_package/'submit_once.py'),
        '--package',str(remote_package),'--manifest-sha',manifest_sha,
        '--release-path',str(PARENT/RELEASE_NAME),'--release-sha',sha(release_path),
        '--script',str(PARENT/DISPATCH_NAME),'--script-sha',sha(dispatch_path),
        '--predecessor-proof',proofs['3326']['path'],'--predecessor-proof-sha',PROOF_SHA]
    archive=HERE/(CANDIDATE_NAME+'-code-only.tar.gz')
    require(not archive.exists(),'V5 archive already exists; final assembly is once only')
    with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as bundle:
        for relative in sorted([*files,'manifest.json']):bundle.add(PACKAGE/relative,arcname='candidates/'+CANDIDATE_NAME+'/'+relative,recursive=False)
        for path in (authority_path,release_path,dispatch_path):bundle.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='CODE_ONLY_DIRECT200_V5_PACKET_MATERIALIZED_NOT_SUBMITTED',packet_revision=5,code_commit=SOURCE,
        local_package=str(PACKAGE),remote_extract_destination=str(PARENT),remote_package=str(remote_package),
        archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=manifest_sha,authority_path=str(authority_path),authority_sha256=sha(authority_path),
        release_path=str(release_path),release_sha256=sha(release_path),dispatch_path=str(dispatch_path),dispatch_sha256=sha(dispatch_path),
        once_submission_argv=argv,once_submission_shell_command=shlex.join(argv),
        once_submission_management_timeout_seconds=120,once_intent=manifest['dispatch']['once_intent'],
        terminal_proof_closure=proofs,historical_code_commit_by_job=source_by_job,
        changed_source_files_sha256=changed,native_rolling_save=rolling,
        new_tests_executed=False,new_reviews_executed=False,actual_slurm_submitted=False,
        contains_model_data_index_checkpoint_assets=False,retained_v4_failed_run_evidence=True,
        fresh_independent_200=True,resume=False,postload_held_ofd_identity_and_foreign15_percent_guards_preserved=True)
    write_json(HERE/'DIRECT_PACKET_BOUND_ARGV_V5.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
