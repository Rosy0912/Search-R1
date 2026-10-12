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
        exact_resume_supported=False,automatic_rollback=False,unrecorded_partial_optimizer_calls=None,
        metadata_only_progress_not_independent_acceptance=True,partial_writes_not_credited_as_completed=True)


def verify_terminal(run,package,config_hash):
    from production_contract import direct_rank_records,tree_and_safety_records,validation_reports,checkpoint_manifest
    from calibration_contract import validate_fresh_start
    fresh=validate_fresh_start(run,package,config_hash)
    ranks=direct_rank_records(run,package,config_hash,200)
    safety=tree_and_safety_records(run,200,package,config_hash)
    development={str(step):validation_reports(run,step,package=package,config_hash=config_hash) for step in range(5,201,5)}
    final=validation_reports(run,200,final=True,package=package,config_hash=config_hash)
    milestones={str(step):checkpoint_manifest(run,package,config_hash,step)[1] for step in (5,50,100,150,200)}
    require(ranks['completed_outer']==200 and final['evaluated_rows']==51713,'Terminal coverage missing')
    return dict(status='FRESH_NESTED_BRANCH200_JSON_ACCOUNTING_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE',
        actual_rank_optimizer_returns=ranks['actual_optimizer_returns'],opportunities_processed=ranks['opportunities_processed'],
        actual_calls_per_rank=ranks['actual_writes_per_rank'],training_attempts=ranks['training_attempts'],
        fresh_initialization=fresh,rank_accounting=ranks,safety=safety,development=development,final=final,milestones=milestones,
        independent_acceptance=False,full_hf_reload_independently_verified=False)
