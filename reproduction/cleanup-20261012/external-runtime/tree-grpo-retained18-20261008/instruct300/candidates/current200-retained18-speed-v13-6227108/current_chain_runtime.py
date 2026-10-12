"""V13 JSON identity and runtime environment for the explicit current fresh200 chain.

No PT/model reload is performed here. The actual core startup migrates pinned
native files and publishes equality receipts; prospective packet fields never
claim that migration has executed.
"""
import os
from pathlib import Path
from production_contract import ROOT,require,read_bound,require_identity,CHAIN_SCOPE
PROTOCOL='current_fresh200_chain_fault_repair_continuation_v1'
PARENT_SOURCE='6efb60466a233f70bf8f50de26df32012f49b4bc'
PARENT_RUN='tree-current200-continue-3335'
PARENT_CONFIG='5e530fc597e26e113c42418ca54faa9bdd0a42ad40bc1998e22c9599be9668fb'
PARENT_MANIFEST=ROOT/'runs'/PARENT_RUN/'checkpoints/training_state/global_step_5/manifest.json'
PARENT_SHA='56832544db6e55cbd346575aca0d06238602141df2704309c1a7f36aa4a27260'
COUNTS={str(rank):100 for rank in range(8)}
ENV={'TREE_CURRENT_CHAIN_RESUME_MANIFEST':str(PARENT_MANIFEST),
     'TREE_CURRENT_CHAIN_RESUME_MANIFEST_SHA256':PARENT_SHA,
     'TREE_CURRENT_CHAIN_RESUME_AUTHORITY':PROTOCOL}
RUN_PREFIX='tree-current200-continue-'


def spec():
    return dict(protocol=PROTOCOL,scope=CHAIN_SCOPE,parent_job_id='3335',parent_run_id=PARENT_RUN,
        parent_code_commit=PARENT_SOURCE,parent_config_sha256=PARENT_CONFIG,
        parent_manifest=dict(path=str(PARENT_MANIFEST),sha256=PARENT_SHA),
        parent_completed_outer=5,parent_training_attempts=5,parent_optimizer_calls_per_rank=COUNTS,
        parent_scheduler_step=5,parent_sampler_cursor=2560,parent_sampler_epoch=0,
        next_outer_update=6,completed_outer_goal=200,planned_new_completed_outer=195,
        environment=ENV,automatic_resume_supported=False,exact_executor_resume=False,
        executor_equivalence=False,actual_migration_executed=False,initial_evaluation_added=False,
        step5_approval_hold_added=False,parent_failed_attempt=dict(attempt_id=6,completed_outer_credit=0,
            optimizer_writes=None,published_rank_optimizer_record=None,error_type='HumanRequestedStop',failure_component='manual_stop_slow3335'))


def validate_chain_spec(package):
    require(package.get('scope')==CHAIN_SCOPE and package.get('packet_revision')==13
        and package.get('current_chain_resume')==spec(),'V13 exact current-chain3335 native5 contract differs')
    require(package.get('fresh_independent_200') is False and package.get('resume') is True
        and package.get('expected_initialized_completed_outer')==5
        and package.get('exact_executor_resume') is False,'Prospective continuation must not become fresh0 or executor equivalence')
    require(package['run_prefix']==RUN_PREFIX and package['code_commit']!=PARENT_SOURCE,
        'Current-chain child run/source must identify the new segment')
    return spec()


def validate_chain_environment(package,environ=None):
    environ=os.environ if environ is None else environ;validate_chain_spec(package)
    actual={key:value for key,value in environ.items() if key.startswith('TREE_CURRENT_CHAIN_RESUME_')}
    require(actual==ENV and not any(key.startswith('TREE_RECOVERY_') for key in environ),
        'Only the exact three current-chain environment fields are permitted')
    return dict(ENV)


def install_chain_environment(package,environ):
    validate_chain_spec(package)
    # Refuse inherited conflicting/extra current-chain fields, then set exactly3.
    inherited={key:value for key,value in environ.items() if key.startswith('TREE_CURRENT_CHAIN_RESUME_')}
    require(not inherited or inherited==ENV,'Inherited current-chain environment conflicts with the immutable packet')
    environ.update(ENV);return validate_chain_environment(package,environ)


def chain_env_for_ray():
    tools=Path(os.environ['TREE_INSTRUCT_TOOLS'])
    package,_,_=read_bound(tools/'manifest.json',os.environ['TREE_INSTRUCT_PACKAGE_SHA256'])
    return validate_chain_environment(package)


def install_ray_chain_environment():
    """Driver audit environment feeds actor/ref workers; propagate only the pinned3."""
    chain_env_for_ray()
    from search_r1.baselines import runtime_audit
    if getattr(runtime_audit,'_v13_current_chain_env_installed',False):return
    original=runtime_audit.runtime_env_vars
    def environment():
        result=original();result.update(chain_env_for_ray());return result
    runtime_audit.runtime_env_vars=environment
    runtime_audit._v13_current_chain_env_installed=True
    from verl.single_controller.ray.base import RayClassWithInitArgs
    original_options=RayClassWithInitArgs.update_options
    def update_options(self,options):
        options=dict(options)
        if 'runtime_env' in options:
            runtime=dict(options['runtime_env']);variables=dict(runtime.get('env_vars',{}))
            variables.update(chain_env_for_ray());runtime['env_vars']=variables;options['runtime_env']=runtime
        return original_options(self,options)
    RayClassWithInitArgs.update_options=update_options


def parent_metadata(package):
    validate_chain_spec(package)
    value,digest,_=read_bound(PARENT_MANIFEST,PARENT_SHA);identity=value['identity']
    expected=dict(job_id='3335',code_commit=PARENT_SOURCE,config_sha256=PARENT_CONFIG,
        data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'],
        completed_step=5,actual_update_limit=200)
    require(value['status']=='COMPLETE_SAME_WORLD_STATE_SAVED'
        and all(identity.get(key)==v for key,v in expected.items()),'Actual parent metadata/source identity differs')
    accounting=identity['guard_continue_accounting']
    require(accounting==dict(protocol='tree_guard_continue_fresh_rollout_v1',training_attempts=5,
        completed_outer_updates=5,refused_attempts=0,guard_refused_attempts=0,
        partial_optimizer_calls_per_rank={},completed_optimizer_calls_per_rank=COUNTS,
        total_optimizer_calls_per_rank=COUNTS,partial_evidence=[]),'Actual parent complete counters differ')
    workers=sorted(value['worker_states'],key=lambda row:row['identity']['rank'])
    require(len(workers)==8 and [row['identity']['rank'] for row in workers]==list(range(8))
        and all(row['full_worker_state_finite'] is True and row['scheduler_last_epoch']==5
            and row['actual_total_optimizer_calls']==100 for row in workers)
        and all(row['exact_same_world_reload_verified'] is True for row in workers)
        and value['driver']['exact_reload_verified'] is True
        and value['driver']['next_step']==6,'Actual parent rank/driver receipts incomplete')
    # Native milestone identity has no run_id. Bind the run through its exact
    # manifest/driver/worker paths and the already published ancestry receipt.
    require(all(Path(row.get('path', str(PARENT_MANIFEST.parent/f"rank-{row['identity']['rank']:02d}.pt")))
        == PARENT_MANIFEST.parent/f"rank-{row['identity']['rank']:02d}.pt" for row in workers)
        and Path(value['driver'].get('path', str(PARENT_MANIFEST.parent/'driver.pt')))
        == PARENT_MANIFEST.parent/'driver.pt', 'Parent native state paths differ')
    receipt,_,_=read_bound(ROOT/'runs'/PARENT_RUN/'output/current-chain-migrated.json',
        '07c19b89289ba2116f874447c7bfbf1dfc93b8d115644be5fe4c2cbf24bb543c')
    ancestry=receipt['provenance']
    require(receipt['status']=='COLD_CURRENT_CHAIN_STATE_MIGRATED'
        and ancestry['child_job_id']=='3335' and ancestry['child_run_id']==PARENT_RUN
        and ancestry['child_code_commit']==PARENT_SOURCE and ancestry['child_config_sha256']==PARENT_CONFIG,
        'Parent native run binding differs')
    return value,digest


def validate_cold_initialization(run,package,config_hash):
    from calibration_contract import expected_runtime,PROFILE
    value,digest,_=read_bound(Path(run)/'output/fresh-initialization.json')
    expected=expected_runtime(run,package,config_hash);require_identity(value,expected)
    require(value['status']=='COLD_OFFICIAL_INITIAL_STATE_OBSERVED_BEFORE_CURRENT_CHAIN_MIGRATION'
        and value['training_protocol']==PROFILE and value['calibration_profile']=='fresh_base_zero_v1'
        and value['initialized_global_steps']==value['parent_completed_step']==value['parent_cursor']==value['driver_completed_outer']==0
        and value['checkpoint_loaded'] is False,'Actual cold official initialization differs')
    workers=sorted(value['workers'],key=lambda row:row['rank'])
    require(len(workers)==8 and [row['rank'] for row in workers]==list(range(8)),'Cold official rank coverage differs')
    for row in workers:
        require_identity(row,expected)
        require(row['world_size']==8 and row['training_protocol']==PROFILE and row['calibration_profile']=='fresh_base_zero_v1'
            and row['checkpoint_recovery_environment_absent'] is True and row['optimizer_state_entries']==0
            and row['optimizer_step_values']==[] and row['scheduler_step']==row['optimizer_committed']==0
            and row['scheduler_advanced'] is False and row['complete_model_parameter_bytes_verified'] is False
            and Path(row['configured_model_path']).resolve()==Path(row['expected_model_path']).resolve()==Path(package['official_model_path']).resolve(),
            'Cold official worker state/path differs')
    return dict(status='ACTUAL_COLD_OFFICIAL_INITIALIZATION_OBSERVED_BEFORE_MIGRATION',evidence_sha256=digest,
        resumed_current_chain=False,parent_state_loaded_at_this_cold_boundary=False,workers=8,
        independent_acceptance=False,complete_model_parameter_bytes_verified=False)


def validate_actual_migration(run,package,config_hash):
    parent,parent_sha=parent_metadata(package);cold=validate_cold_initialization(run,package,config_hash)
    path=Path(run)/'output/current-chain-migrated.json';value,digest,_=read_bound(path)
    provenance=value['provenance'];expected=dict(child_code_commit=package['code_commit'],child_config_sha256=config_hash,
        child_job_id=Path(run).name.rsplit('-',1)[1],child_run_id=Path(run).name,
        protocol=PROTOCOL,parent_identity=parent['identity'],parent_manifest_path=str(PARENT_MANIFEST),
        parent_manifest_sha256=parent_sha,parent_completed_step=5,parent_cursor=2560,parent_epoch=0,
        new_batch_size=512,same_fresh200_chain=True,automatic_resume_supported=False,
        executor_equivalence=False,exact_executor_resume=False,parent_failed_attempt=dict(attempt_id=6,completed_outer_credit=0,
            optimizer_writes=None,published_rank_optimizer_record=None,error_type='HumanRequestedStop',failure_component='manual_stop_slow3335',
            scope='User stopped slow3335 after completed5; interrupted6 partial writes remain unknown; only completed native5 is migrated'))
    require(value['status']=='COLD_CURRENT_CHAIN_STATE_MIGRATED' and value['protocol']==PROTOCOL
        and all(provenance.get(key)==expected_value for key,expected_value in expected.items()) and value['executor_equivalence'] is False
        and value['initial_evaluation_added'] is False and value['step5_approval_hold_added'] is False,
        'Actual cold migration provenance differs from parent/child chain')
    ancestry=provenance['saved_parent_provenance']
    parent_receipt,_,_=read_bound(ROOT/'runs'/PARENT_RUN/'output/current-chain-migrated.json',
        '07c19b89289ba2116f874447c7bfbf1dfc93b8d115644be5fe4c2cbf24bb543c')
    require(ancestry==parent_receipt['provenance']
        and ancestry['protocol']==PROTOCOL and ancestry['same_fresh200_chain'] is True
        and ancestry['child_job_id']=='3335' and ancestry['child_run_id']==PARENT_RUN
        and ancestry['child_code_commit']==PARENT_SOURCE and ancestry['child_config_sha256']==PARENT_CONFIG,
        'Actual complete3335 ancestry differs from bound native source/config receipt')
    cold_record,_,_=read_bound(Path(run)/'output/fresh-initialization.json')
    require(value['cold_official_initialization']==cold_record,'Actual migration drops cold official boundary')
    workers=sorted(value['workers'],key=lambda row:row['rank'])
    require(len(workers)==8 and [row['rank'] for row in workers]==list(range(8)),'Actual migrated rank coverage differs')
    originals={row['identity']['rank']:row for row in parent['worker_states']}
    for row in workers:
        original=originals[row['rank']]
        require(row['status']=='COLD_CURRENT_CHAIN_WORKER_STATE_MIGRATED' and row['protocol']==PROTOCOL
            and row['world_size']==8 and row['parent_identity']==original['identity']
            and row['parent_manifest_sha256']==parent_sha and row['state_file_sha256']==original['sha256']
            and Path(row['state_path'])==PARENT_MANIFEST.parent/f"rank-{row['rank']:02d}.pt"
            and row['actual_file_sha256_verified'] is True and row['actual_cold_native_state_equal'] is True
            and row['serialized_logical_state_sha256']==row['actual_loaded_logical_state_sha256']==original['logical_state_sha256']
            and row['adam_steps']==[100] and row['scheduler_step']==5 and row['actual_total_optimizer_calls']==100
            and row['child_code_commit']==package['code_commit'] and row['executor_equivalence'] is False
            and row['automatic_resume_supported'] is False,'Actual migrated worker equality/counter receipt differs')
    driver=value['driver'];original=parent['driver']
    require(Path(driver['state_path'])==PARENT_MANIFEST.parent/'driver.pt'
        and driver['state_file_sha256']==original['sha256'] and driver['actual_file_sha256_verified'] is True
        and driver['actual_cold_driver_state_equal'] is True and driver['completed_outer_updates']==5
        and driver['next_outer_update']==6 and driver['sampler_cursor']==2560 and driver['sampler_epoch']==0
        and driver['training_attempts']==5 and driver['completed_optimizer_calls_per_rank']==COUNTS,
        'Actual migrated driver/native sampler receipt differs')
    return dict(status='ACTUAL_CURRENT_FRESH200_CHAIN_COLD_MIGRATION_OBSERVED',path=str(path),sha256=digest,
        parent_manifest=dict(path=str(PARENT_MANIFEST),sha256=parent_sha),parent_source=PARENT_SOURCE,
        child_source=package['code_commit'],parent_completed_outer=5,next_outer_update=6,
        parent_actual_optimizer_calls_per_rank=COUNTS,parent_scheduler_step=5,resumed_current_chain=True,
        exact_executor_resume=False,executor_equivalence=False,automatic_resume_supported=False,
        initial_evaluation_added=False,step5_approval_hold_added=False,cold_initialization=cold,
        metadata_only_wrapper_observation=True,raw_payloads_reopened=False,pt_deserialized=False,independent_acceptance=False)


def chain_progress(run,package,config_hash):
    """Only persisted decisions count; absence of a migration receipt is not resume."""
    run=Path(run);errors=[];migrated=False;complete=0;attempts=0;totals={str(rank):0 for rank in range(8)};new_totals=totals.copy();refused=0
    if (run/'output/current-chain-migrated.json').is_file():
        try:validate_actual_migration(run,package,config_hash);migrated=True;complete=5;attempts=5;totals=COUNTS.copy()
        except Exception as exc:errors.append(dict(path=str(run/'output/current-chain-migrated.json'),error=repr(exc)))
    files=sorted((run/'checkpoints/training_state/attempt-results').glob('attempt-*.json'))
    for expected_attempt,path in enumerate(files,6):
        try:
            require(migrated,'Child decisions exist without a verified actual migration receipt')
            value,_,_=read_bound(path);require(path.name==f'attempt-{expected_attempt:06d}.json','Persisted child attempt gap')
            if value['status']=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED':
                require(value['accepted_outer']==complete+1,'Persisted chain completed counter gap')
                ranks=value['all_rank_result']['ranks'];require(len(ranks)==8 and {row['rank'] for row in ranks}==set(range(8)),'Persisted chain rank gap')
                complete+=1
                for rank in ranks:
                    key=str(rank['rank']);totals[key]+=rank['optimizer_calls'];new_totals[key]+=rank['optimizer_calls']
            elif value['status']=='ALL_OPPORTUNITIES_SKIPPED_NO_OUTER_CREDIT':refused+=1
            else:raise ValueError('Fault/unrecognized persisted child decision')
            attempts+=1
        except Exception as exc:errors.append(dict(path=str(path),error=repr(exc)));break
    partial=[]
    for path in sorted((run/'checkpoints/training_state').glob('partial_attempt_*/manifest.json')):
        try:value,digest,_=read_bound(path);partial.append(dict(path=str(path),sha256=digest,value=value))
        except Exception as exc:errors.append(dict(path=str(path),error=repr(exc)))
    return dict(scope=CHAIN_SCOPE,resumed_current_chain=migrated,parent_completed_outer=5 if migrated else 0,
        planned_parent_completed_outer=5,actual_completed_outer_from_attempt_results=complete,new_completed_outer=max(complete-5,0),
        training_attempts_seen=attempts,new_training_attempts_seen=max(attempts-5,0),all_skipped_attempts=refused,
        parent_actual_optimizer_calls_per_rank=COUNTS if migrated else {str(rank):0 for rank in range(8)},
        completed_actual_optimizer_calls_per_rank=totals,new_completed_actual_optimizer_calls_per_rank=new_totals,
        partial_attempt_evidence=partial,errors=errors,exact_resume_supported=False,exact_executor_resume=False,
        automatic_rollback=False,unrecorded_partial_optimizer_calls=None,
        metadata_only_progress_not_independent_acceptance=True,partial_writes_not_credited_as_completed=True)
