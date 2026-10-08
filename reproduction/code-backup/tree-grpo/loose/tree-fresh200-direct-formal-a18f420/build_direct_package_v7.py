"""Assemble one immutable code-only V7 current fresh200 chain continuation packet.

The root supplies already committed source and actual native3328 closure. This
stdlib assembler does not import wrappers/source, run tests/reviews/calibration,
load assets or perform remote/submission work.
"""
import argparse,copy,datetime,hashlib,json,re,shlex,subprocess,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE/'package-code-only-v6';PACKAGE=HERE/'package-code-only-v7'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');PARENT=ROOT/'instruct300'
BASE='4a65e9a3ab87e84e1e9dd283b664cdef3a224c8a'
OLD_SHA='ef2817200f999c74deca43ea3de501f6bf18276ea838b38b35342ebf3368265f'
MODE='HUMAN_AUTHORIZED_DIRECT_TRAIN200';SCOPE='CURRENT_FRESH200_CHAIN_CONTINUATION'
PROTOCOL='current_fresh200_chain_fault_repair_continuation_v1'
PARENT_STATE=str(ROOT/'runs/tree-fresh200-train-3328/checkpoints/training_state/rolling-completed/global_step_2/manifest.json')
PARENT_STATE_SHA='067d71ee3c7fa0e00da471bebbd9e8913ec7e5de0b1c98bdb7598d7a022565a4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def write_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args])
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
def chain_spec():
    counts={str(rank):40 for rank in range(8)}
    env=dict(TREE_CURRENT_CHAIN_RESUME_MANIFEST=PARENT_STATE,TREE_CURRENT_CHAIN_RESUME_MANIFEST_SHA256=PARENT_STATE_SHA,
        TREE_CURRENT_CHAIN_RESUME_AUTHORITY=PROTOCOL)
    return dict(protocol=PROTOCOL,scope=SCOPE,parent_job_id='3328',parent_run_id='tree-fresh200-train-3328',
        parent_code_commit=BASE,parent_config_sha256='6715a4276599724aaaae1511d973f3dff3d6994c3f41074506d17a016e7ef644',
        parent_manifest=dict(path=PARENT_STATE,sha256=PARENT_STATE_SHA),parent_completed_outer=2,parent_training_attempts=2,
        parent_optimizer_calls_per_rank=counts,parent_scheduler_step=2,parent_sampler_cursor=1024,parent_sampler_epoch=0,
        next_outer_update=3,completed_outer_goal=200,planned_new_completed_outer=198,environment=env,
        automatic_resume_supported=False,exact_executor_resume=False,executor_equivalence=False,actual_migration_executed=False,
        initial_evaluation_added=False,step5_approval_hold_added=False,parent_failed_attempt=dict(attempt_id=3,
            completed_outer_credit=0,optimizer_writes=None,published_rank_optimizer_record=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','source-commit','source-delta','source-delta-sha','predecessor-proof-result',
                 'predecessor-proof-result-sha','predecessor-proof-sha'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();source=a.source_commit;repo=Path(a.source_dir).resolve()
    require(not (PACKAGE/'manifest.json').exists(),'V7 already frozen; never overwrite/retry assembly')
    require(re.fullmatch('[0-9a-f]{40}',source) and source!=BASE,'Root new unified source required')
    require(sha(OLD/'manifest.json')==OLD_SHA,'Actual3328 V6 manifest changed')
    require(sha(a.source_delta)==a.source_delta_sha,'Root code-only delta changed')
    delta=json.loads(Path(a.source_delta).read_text())
    require(delta['code_commit']==source and delta['base_code_commit']==BASE and delta['source_dir']==str(repo)
        and delta['new_tests_executed'] is False and delta['new_reviews_executed'] is False,
        'Root freeze identity or execution claims differ')
    require(git(repo,'rev-parse','HEAD').decode().strip()==source and not git(repo,'status','--porcelain').strip(),
        'Unified committed source changed after freeze')
    changed=delta['changed_source_files_sha256'];names=git(repo,'diff','--name-only',BASE,source).decode().splitlines()
    require(set(names)==set(changed),'Root delta omits a unified commit change')
    old=json.loads((OLD/'manifest.json').read_text());selected=set(old['source_selection']['selected_files'])|set(changed)
    require('search_r1/baselines/tree_current_chain_resume.py' in changed,'Actual core current-chain migration module required')
    source_files={}
    for name in sorted(selected):
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and (name in old['source_selection']['selected_files'] or name.startswith(('search_r1/','verl/','reproduction/'))),
            'Unsafe/non-code source selection')
        raw=git(repo,'show',source+':'+name);pin=hashlib.sha256(raw).hexdigest()
        require(pin==(changed[name] if name in changed else old['source_selection']['selected_files'][name]),
            'Frozen source SHA differs from unchanged parent or exact root delta: '+name)
        path=PACKAGE/'source/repo'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);source_files[name]=pin
    contract=PACKAGE/'production_contract.py';text=contract.read_text()
    require("SOURCE = 'V7_SOURCE_COMMIT_PENDING'" in text,'V7 prospective source placeholder already replaced')
    contract.write_text(text.replace("SOURCE = 'V7_SOURCE_COMMIT_PENDING'","SOURCE = '"+source+"'"))
    require(sha(a.predecessor_proof_result)==a.predecessor_proof_result_sha,'Root native3328 proof result changed')
    result=json.loads(Path(a.predecessor_proof_result).read_text());native=result['proof']
    require(result['proof_sha256']==a.predecessor_proof_sha and native['job_id']==3328 and native['code_commit']==BASE
        and native['status']=='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP' and native['scheduler_state']=='FAILED'
        and native['scheduler_exit_code']=='1:0' and native['cleanup_residual']=={}
        and native['original_eight_pid_start_times_preserved'] is True,
        'Exact native3328 terminal/source/cleanup proof required')
    require(native['run_result']['sha256']=='7386e708ee492f618085b9dcede97d22704dd58bc6676abf1a5650fa8a41cbf6'
        and native['cost']['sha256']=='abd0336c14967af8eac5c01327f427b529ba8f021c7ac3143a2d10b3f3dc00b2'
        and native['current_chain_parent_manifest']==dict(path=PARENT_STATE,sha256=PARENT_STATE_SHA),
        'Exact real parent terminal files/rolling2 differ')
    short=source[:7];candidate='current-fresh200-chain-mask-v7-'+short
    intent='SUBMISSION_ONCE-current-fresh200-chain-mask-v7-'+short+'.json'
    authority_name='CURRENT_FRESH200_CHAIN_HUMAN_AUTHORITY-V7-'+short+'.json'
    release_name='CURRENT_FRESH200_CHAIN_RELEASE-V7-'+short+'.json';dispatch_name='CURRENT_FRESH200_CHAIN_DISPATCH-V7-'+short+'.sbatch'
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat();resume=chain_spec()
    proofs=copy.deepcopy(old['terminal_proof_closure']);proofs['3328']=dict(path=result['proof_path'],sha256=a.predecessor_proof_sha)
    history=copy.deepcopy(old['historical_code_commit_by_job']);history['3328']=BASE
    authority=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_HUMAN_AUTHORITY-V6.json').read_text()))
    authority.update(packet_revision=7,scope=SCOPE,code_commit=source,current_chain_resume=resume,
        same_fresh200_chain=True,fresh_independent_200=False,resume=True,exact_executor_resume=False,
        core_source_bytes_changed=True,root_fixed_core_files_sha256=changed,root_fixed_wrapper_files_sha256={},
        real_error_fix=delta['actual_fault'],error_fix_observed_at_utc=observed,
        new_tests_executed=False,new_reviews_executed=False,actual_migration_executed=False,
        parent_costs_included_in_new_run=False)
    authority['historical_native_rolling_save']=authority.pop('native_rolling_save',{})
    authority_path=HERE/authority_name;write_json(authority_path,authority)
    files={str(path.relative_to(PACKAGE)):sha(path) for path in sorted(PACKAGE.rglob('*')) if path.is_file()}
    require(all('__pycache__' not in Path(name).parts and Path(name).suffix not in ('.pyc','.pt','.safetensors','.parquet','.index') for name in files),
        'V7 package must stay code-only without import caches/assets')
    wrappers={name:pin for name,pin in files.items() if not name.startswith('source/')}
    changed_wrappers={name:pin for name,pin in wrappers.items() if old['files'].get(name)!=pin}
    fix=dict(code_commit=source,base_source_commit=BASE,source_delta_metadata_sha256=a.source_delta_sha,
        changed_source_files_sha256=changed,changed_wrapper_files_sha256=changed_wrappers,
        actual_fault=delta['actual_fault'],new_tests_executed=False,new_reviews_executed=False)
    m=copy.deepcopy(old)
    for key in ('historical_completed_outer_not_inherited','prior_model_optimizer_scheduler_checkpoint_available','root_actual_wrapper_error_fix'):
        m.pop(key,None)
    m.update(packet_revision=7,scope=SCOPE,execution_mode=MODE,code_commit=source,materialized_at_utc=observed,
        run_prefix='tree-current200-continue-',current_chain_resume=resume,human_training_authority=dict(path=str(PARENT/authority_name),sha256=sha(authority_path)),
        supersedes_failed_packet_manifest_sha256=OLD_SHA,predecessor_terminal_cleanup_proof=proofs['3328'],
        terminal_proof_closure=proofs,historical_code_commit_by_job=history,core_source_bytes_changed=True,
        root_actual_error_fix=fix,fresh_independent_200=False,resume=True,exact_executor_resume=False,
        expected_initialized_completed_outer=2,planned_new_completed_outer=198,actual_migration_executed=False,
        new_tests_executed=False,new_cpu_tests_executed=False,new_reviews_executed=False,
        wrapper_inventory_scope='Exact V7 code bytes bound; existing historical inventory and receipts remain reference-only; no new tests or review')
    m['dispatch']['once_intent']=str(PARENT/intent);m['files']=files;m['wrapper_files']=wrappers
    m['source_selection'].update(code_commit=source,selected_files=source_files,selected_count=len(source_files),
        inherited_unchanged_from_manifest_sha256=OLD_SHA,changed_source_files_sha256=changed)
    m['protocol'].update(fresh_official_3b_no_inheritance=False,cold_official_3b_initialization_before_actual_migration=True,
        reference_official_3b=True,current_chain_parent_actor_optimizer_scheduler_inheritance=True,
        parent_completed_outer=2,planned_new_completed_outer=198,completed_outer_goal=200,exact_executor_resume=False)
    m['native_rolling_save'].update(wrapper_files_sha256={name:files[name] for name in ('rolling_training_state_hooks.py','validation_cost_entry.py')},
        source_core_save_bytes_unchanged=False,exact_executor_resume_verified=False)
    m['previous_failed_direct_runs'].append(dict(job_id=3328,status='FAILED',scheduler_exit_code='1:0',code_commit=BASE,
        native_proof=proofs['3328'],run_result_sha256=native['run_result']['sha256'],cost_sha256=native['cost']['sha256'],
        once_journal_sha256=native['once_journal']['sha256'],actual_optimizer_committed=native['native_optimizer_committed'],
        scheduler_advanced=native['native_scheduler_advanced'],failed_unrecorded_partial_optimizer_writes_unknown=True,
        direct_completed_outer_observation=native['direct_completed_outer_observation'],current_chain_parent_manifest=native['current_chain_parent_manifest'],
        checkpoint_inheritance=True,parent_failed_attempt=native['parent_failed_attempt'],scope='Same current fresh200 rolling2 is explicitly migrated; native nulls and failed third generation remain historical'))
    manifest_path=PACKAGE/'manifest.json';write_json(manifest_path,m);manifest_sha=sha(manifest_path)
    release=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_RELEASE-V6.json').read_text()))
    release.update(packet_revision=7,scope=SCOPE,code_commit=source,current_chain_resume=resume,manifest_sha256=manifest_sha,
        human_training_authority_sha256=sha(authority_path),dispatch_intent_path=m['dispatch']['once_intent'],
        dispatch_renderer_sha256=files['render_dispatch.py'],dispatch_submitter_sha256=files['submit_once.py'],
        predecessor_terminal_cleanup_proof=proofs['3328'],terminal_proof_closure=proofs,historical_code_commit_by_job=history,
        core_source_bytes_changed=True,root_actual_error_fix=fix,exact_executor_resume=False,
        fresh_independent_200=False,resume=True,expected_initialized_completed_outer=2,planned_new_completed_outer=198,
        new_tests_executed=False,new_reviews_executed=False,actual_migration_executed=False)
    release.pop('root_actual_wrapper_error_fix',None);release.pop('historical_completed_outer_not_inherited',None)
    release['native_rolling_save']=copy.deepcopy(m['native_rolling_save'])
    release_path=HERE/release_name;write_json(release_path,release)
    remote=PARENT/'candidates'/candidate;dispatch_path=HERE/dispatch_name
    with dispatch_path.open('x') as stream:stream.write(render(remote,manifest_sha,PARENT/release_name,sha(release_path),168))
    argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONOPTIMIZE=0','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1',
        str(ROOT/'env/bin/python'),'-B',str(remote/'submit_once.py'),'--package',str(remote),'--manifest-sha',manifest_sha,
        '--release-path',str(PARENT/release_name),'--release-sha',sha(release_path),'--script',str(PARENT/dispatch_name),
        '--script-sha',sha(dispatch_path),'--predecessor-proof',proofs['3328']['path'],'--predecessor-proof-sha',a.predecessor_proof_sha]
    archive=HERE/(candidate+'-code-only.tar.gz');require(not archive.exists(),'V7 archive must be new')
    with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as bundle:
        for name in sorted([*files,'manifest.json']):bundle.add(PACKAGE/name,arcname='candidates/'+candidate+'/'+name,recursive=False)
        for path in (authority_path,release_path,dispatch_path):bundle.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='CODE_ONLY_CURRENT_FRESH200_CHAIN_V7_PACKET_MATERIALIZED_NOT_SUBMITTED',packet_revision=7,
        execution_mode=MODE,scope=SCOPE,code_commit=source,parent_code_commit=BASE,current_chain_resume=resume,
        local_package=str(PACKAGE),remote_extract_destination=str(PARENT),remote_package=str(remote),archive=str(archive),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,manifest_sha256=manifest_sha,
        authority_path=str(authority_path),authority_sha256=sha(authority_path),release_path=str(release_path),release_sha256=sha(release_path),
        dispatch_path=str(dispatch_path),dispatch_sha256=sha(dispatch_path),once_submission_argv=argv,
        once_submission_shell_command=shlex.join(argv),once_submission_management_timeout_seconds=120,
        once_intent=m['dispatch']['once_intent'],terminal_proof_closure=proofs,historical_code_commit_by_job=history,
        source_delta_metadata_path=a.source_delta,source_delta_metadata_sha256=a.source_delta_sha,
        changed_source_files_sha256=changed,changed_wrapper_files_sha256=changed_wrappers,actual_slurm_submitted=False,
        actual_migration_executed=False,exact_executor_resume=False,parent_completed_outer=2,planned_new_completed_outer=198,
        contains_model_data_index_checkpoint_assets=False,new_tests_executed=False,new_reviews_executed=False,
        production_thresholds_accepted=False,original_sampler_actor_gate_accepted=False,training_accepted=False,independent_acceptance=False,
        postload_held_ofd_identity_and_foreign15_percent_guards_preserved=True)
    write_json(HERE/'CURRENT_CHAIN_PACKET_BOUND_ARGV_V7.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
