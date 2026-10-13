import os,json,time
from pathlib import Path
from production_contract import require,read_bound
def guard_health(run,job):
    with (Path(run)/'resource-samples.jsonl').open('rb') as stream:
        stream.seek(0,os.SEEK_END);size=stream.tell();stream.seek(max(0,size-131072));tail=stream.read()
    rows=tail.splitlines()
    if tail and not tail.endswith(b'\n'):rows=rows[:-1]
    require(rows,'Missing complete actual guard row')
    last=json.loads(rows[-1]);require(last.get('job_id')==job and not last.get('event') and time.time()-last['epoch']<15,'Actual guard unhealthy/stale')


def progress_evidence(run):
    """Direct persisted decisions only; unmatched/unrecorded writes stay unknown."""
    root=Path(run)/'checkpoints/training_state';completed=0;attempts=0;totals={str(r):0 for r in range(8)};partial=[];errors=[];refused=0
    native_step=5 if (Path(run)/'output/nested-native5-loaded.json').is_file() else 1
    native_start=Path(run)/'output'/f'nested-native{native_step}-loaded.json'
    inherited=None
    if native_start.is_file():
        value,receipt_sha,_=read_bound(native_start)
        driver=value['driver'];workers=value.get('workers',[])
        boundary_calls,boundary_cursor=(100,2560) if native_step==5 else (20,512)
        require(value['status']==f'COLD_NESTED_NATIVE{native_step}_STATE_LOADED'
                and driver['completed_outer_updates']==native_step and driver['training_attempts']==native_step
                and driver['sampler_cursor']==boundary_cursor and driver['sampler_epoch']==0
                and driver['completed_optimizer_calls_per_rank']=={str(r):boundary_calls for r in range(8)}
                and driver['actual_cold_driver_state_equal'] is True and driver['actual_file_sha256_verified'] is True
                and len(workers)==8 and sorted(w.get('rank') for w in workers)==list(range(8))
                and all(w.get('status')==f'COLD_NESTED_NATIVE{native_step}_WORKER_LOADED'
                    and w.get('actual_cold_native_state_equal') is True and w.get('actual_file_sha256_verified') is True
                    and w.get('actual_total_optimizer_calls')==boundary_calls and w.get('scheduler_step')==native_step
                    for w in workers),
                'Progress mother boundary differs')
        completed=driver['completed_outer_updates'];attempts=driver['training_attempts']
        totals=dict(driver['completed_optimizer_calls_per_rank']);provenance=value['provenance']
        inherited=dict(completed_outer=completed,receipt_sha256=receipt_sha,
            failed_parent_outer2=provenance['failed_parent_outer2'],
            failed_parent_outer6=provenance.get('failed_parent_outer6'),
            effective_ancestor_segments=provenance.get('effective_ancestor_segments'),
            parent_source_commit=provenance['parent_identity']['code_commit'])
    for path in sorted((root/'attempt-results').glob('attempt-*.json')):
        try:
            value,sha,_=read_bound(path);attempts+=1
            if value['status']=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED':
                require(value['accepted_outer']==completed+1,'Persisted completed counter gap')
                ranks=value['all_rank_result']['ranks'];require(len(ranks)==8,'Persisted rank coverage missing')
                completed+=1
                for rank in ranks:totals[str(rank['rank'])]+=rank['optimizer_calls']
            elif value['status']=='ALL_OPPORTUNITIES_SKIPPED_NO_OUTER_CREDIT':refused+=1
            else:raise ValueError('Unknown/fault persisted decision')
        except Exception as exc:errors.append(dict(path=str(path),error=repr(exc)));break
    paths=list(root.glob('partial_outer_*/**/manifest.json'))+list(root.glob('partial_attempt_*/**/manifest.json'))
    for path in sorted(paths):
        try:
            value,sha,_=read_bound(path);partial.append(dict(path=str(path),sha256=sha,value=value))
        except Exception as exc:errors.append(dict(path=str(path),error=repr(exc)))
    return dict(actual_completed_outer_from_attempt_results=completed,training_attempts_seen=attempts,all_skipped_attempts=refused,
        completed_actual_optimizer_calls_per_rank=totals,partial_attempt_evidence=partial,errors=errors,
        inherited_native_restore=inherited,inherited_native1=inherited if native_step==1 else None,
        failed_parent_outer2_in_effective_history=False,failed_parent_outer6_in_effective_history=False,
        exact_resume_supported=False,automatic_rollback=False,unrecorded_partial_optimizer_calls=None,
        metadata_only_progress_not_independent_acceptance=True,partial_writes_not_credited_as_completed=True)


def verify_terminal(run,package,config_hash):
    from production_contract import direct_rank_records,tree_and_safety_records,validation_reports,checkpoint_manifest
    from calibration_contract import validate_fresh_start
    fresh=validate_fresh_start(run,package,config_hash)
    native=None
    if package.get('nested_native_continuation') is not None:
        from calibration_contract import validate_native_start
        native=validate_native_start(run,package,config_hash)
    ranks=direct_rank_records(run,package,config_hash,200)
    safety=tree_and_safety_records(run,200,package,config_hash)
    inherited_outer=native['loaded_completed_outer'] if native is not None else 0
    parent_development=None;parent_milestone=None
    if inherited_outer==5:
        from search_r1.baselines.tree_nested_chain_resume import (PARENT_ROOT,PARENT_COMMIT,
            PARENT_CONFIG_SHA256,PARENT_MANIFEST,PARENT_MANIFEST_SHA256,parent_manifest)
        parent_package=dict(package,code_commit=PARENT_COMMIT)
        parent_package.pop('nested_native_continuation',None)
        parent_development=validation_reports(PARENT_ROOT,5,package=parent_package,config_hash=PARENT_CONFIG_SHA256)
        preserved_parent=parent_manifest();bound_parent,parent_milestone,_=read_bound(PARENT_MANIFEST)
        require(parent_milestone==PARENT_MANIFEST_SHA256 and bound_parent==preserved_parent,
                'Inherited native5 permanent manifest binding differs')
    development={str(step):(parent_development if step==5 and inherited_outer==5
        else validation_reports(run,step,package=package,config_hash=config_hash)) for step in range(5,201,5)}
    final=validation_reports(run,200,final=True,package=package,config_hash=config_hash)
    milestones={str(step):(parent_milestone if step==5 and inherited_outer==5
        else checkpoint_manifest(run,package,config_hash,step)[1]) for step in (5,50,100,150,200)}
    require(ranks['completed_outer']==200 and final['evaluated_rows']==51713,'Terminal coverage missing')
    return dict(status='FRESH_NESTED_BRANCH200_JSON_ACCOUNTING_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE',
        actual_rank_optimizer_returns=ranks['actual_optimizer_returns'],opportunities_processed=ranks['opportunities_processed'],
        actual_calls_per_rank=ranks['actual_writes_per_rank'],training_attempts=ranks['training_attempts'],
        fresh_initialization=fresh,native_continuation=native,
        inherited_completed_outers=inherited_outer,child_completed_outers=200-inherited_outer,
        effective_ancestor_segments=(native or {}).get('effective_ancestor_segments'),
        failed_parent_outer2_in_effective_history=False,failed_parent_outer6_in_effective_history=False,
        rank_accounting=ranks,safety=safety,development=development,final=final,milestones=milestones,
        independent_acceptance=False,full_hf_reload_independently_verified=False)
