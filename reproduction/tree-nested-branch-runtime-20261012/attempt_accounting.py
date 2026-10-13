from production_contract import run_job_id
"""JSON-only consumer of source-owner DRAFT opportunity/attempt interface.

Requires actual source freeze/review before release. Never equates opportunities
with writes, drops allskip/partial evidence, or infers sampler cursor from credit.
"""
from pathlib import Path
from production_contract import require,read_bound,require_identity

PROTOCOL='tree_guard_continue_fresh_rollout_v1'
RESULT_PROTOCOL='tree_fresh_actor_result_v1'


def ledger_writes(ledger):
    require(type(ledger) is list and len(ledger)==20,'Incomplete twenty-opportunity ledger')
    writes=0;braked=False
    allowed={'SUCCESS_ADAM_WRITE','SUCCESS_ADAM_WRITE_BRAKE_AFTER_WRITE','SKIPPED_EMPTY',
             'SKIPPED_SAFETY_GUARD_BEFORE_WRITE','SKIPPED_BY_PRIOR_ROLLOUT_BRAKE'}
    for index,entry in enumerate(ledger,1):
        success=entry['status'].startswith('SUCCESS')
        require(type(entry['index']) is int and entry['index']==index and type(entry['rows']) is int and entry['rows']==16 and entry['local_row_indices']==list(range((index-1)*16,index*16))
                and entry['status'] in allowed and type(entry['actual_writes']) is int and entry['actual_writes'] in (0,1)
                and success==bool(entry['actual_writes']), 'Actual opportunity rows/status/write mismatch')
        require(not braked or entry['status']=='SKIPPED_BY_PRIOR_ROLLOUT_BRAKE','A write occurred after the first rollout brake')
        if entry['status'] in ('SUCCESS_ADAM_WRITE_BRAKE_AFTER_WRITE','SKIPPED_SAFETY_GUARD_BEFORE_WRITE'):braked=True
        writes+=entry['actual_writes']
    return writes


def attempt_results(run,package,config_hash,last):
    run=Path(run);expected=dict(job_id=run_job_id(run),run_id=run.name,code_commit=package['code_commit'],
        config_sha256=config_hash,data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'])
    complete=0;refused=0;totals={str(r):0 for r in range(8)};completed_totals=totals.copy();partial_totals=totals.copy()
    bindings={};credited=[];skipped=[];opportunities=0
    files=sorted((run/'checkpoints/training_state/attempt-results').glob('attempt-*.json'))
    for attempt,path in enumerate(files,1):
        value,sha,_=read_bound(path);require(path.name==f'attempt-{attempt:06d}.json','Attempt result sequence has a gap/duplicate')
        bindings[str(path)]=sha
        if value['status']=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED':
            require(value['protocol']==RESULT_PROTOCOL and value['attempt_id']==attempt and value['accepted_outer']==complete+1
                    and value['automatic_rollback'] is False,'Actual credited candidate/attempt mismatch')
            result=value['all_rank_result'];ranks=result['ranks']
            require_identity(result['identity'],expected)
            require(result['identity']['training_protocol']=='tree_fresh200_five_leaf_v1' and result['identity']['attempt_id']==attempt
                    and result['identity']['attempted_outer']==complete+1 and result['identity']['completed_outer']==complete, 'Credited gathered identity differs from current attempt')
            require(result['protocol']==RESULT_PROTOCOL and result['decision']=='COMPLETED_OUTER'
                    and result['attempt_id']==attempt and result['accepted_outer']==complete+1 and result['completed_outer_before']==complete
                    and len(ranks)==8 and [x['rank'] for x in ranks]==list(range(8)), 'Explicit actual all-rank completion missing')
            writes={}
            for rank,record in enumerate(ranks):
                actual=ledger_writes(record['opportunity_ledger'])
                require(1<=actual<=20 and record['optimizer_calls']==actual and record['opportunities_processed']==20
                        and record['opportunities_skipped']==20-actual and record['scheduler_before']==complete
                        and record['scheduler_after']==complete+1 and record['attempt_id']==attempt
                        and record['accepted_outer']==complete+1 and record['completed_outer_before']==complete,'Credited rank opportunity/write/scheduler mismatch')
                direct=record['direct_rank_record'];direct_path=run/'output/fresh-state'/f'outer-{complete+1:06d}-attempt-{attempt:06d}-rank{rank:02d}.json'
                require(Path(direct['path']).resolve()==direct_path.resolve() and direct['rank']==rank
                        and direct['actual_optimizer_calls']==actual and direct['scheduler_step_before']==complete
                        and direct['scheduler_step_after']==complete+1,'Gathered actual direct-rank descriptor differs')
                physical,dsha,_=read_bound(direct_path,direct['sha256']);bindings[str(direct_path)]=dsha
                require_identity(physical,expected)
                require(physical['rank']==rank and physical['world_size']==8 and physical['attempt_id']==attempt
                        and physical['observed_outer']==complete+1 and physical['actual_optimizer_calls']==actual
                        and physical['scheduler_step_before']==complete and physical['scheduler_step_after']==complete+1
                        and physical['optimizer_opportunity_ledger']==record['opportunity_ledger'],'Physical rank ledger differs from gathered result')
                writes[str(rank)]=actual;totals[str(rank)]+=actual;completed_totals[str(rank)]+=actual
                require(record['total_optimizer_calls']==totals[str(rank)],'Actual cumulative Adam count omits history')
            require(len(set(writes.values()))==1 and value['actual_completed_optimizer_calls_per_rank']==completed_totals,'Matched complete accounting differs across ranks')
            complete+=1;credited.append(dict(outer=complete,attempt=attempt,actual_writes_per_rank=writes,result_sha256=sha))
        elif value['status']=='ALL_OPPORTUNITIES_SKIPPED_NO_OUTER_CREDIT':
            require(value['protocol']=='tree_partial_state_v1','Unknown no-credit protocol')
            ids=value['identity'];require_identity(ids,expected)
            require(ids['attempt_id']==attempt and ids['attempted_outer']==complete+1 and ids['completed_outer']==complete,'Allskip candidate credit changed')
            evidence=value['partial_evidence'];partial_path=Path(evidence['path'])
            require(partial_path.resolve()==(run/'checkpoints/training_state'/f'partial_attempt_{attempt:06d}_outer_{complete+1:06d}'/'manifest.json').resolve(),'Allskip evidence outside actual attempt')
            partial,psha,_=read_bound(partial_path,evidence['sha256']);bindings[str(partial_path)]=psha
            from partial_raw_evidence import verify_partial_raw_files
            bindings.update(verify_partial_raw_files(run,partial))
            require(partial['identity']==ids and partial['worker_states_verified'] is True and partial['verification_error'] is None
                    and partial['continuation_decision']=='CONTINUE_FRESH_ROLLOUT' and partial['outer_completed'] is False
                    and partial['completed_outer_updates']==complete and partial['credited_outer_candidate'] is False
                    and partial['automatic_rollback'] is False,'Uncredited partial evidence is unsafe/incomplete')
            workers=partial['worker_outcome']['worker_states'];require(len(workers)==8 and {x['rank'] for x in workers}==set(range(8)),'Allskip rank coverage missing')
            for worker in workers:
                rank=str(worker['rank']);actual=ledger_writes(worker['optimizer_opportunity_ledger'])
                require(actual==0 and worker['committed_optimizer_calls']==0 and worker['safety_guard'] is True
                        and worker['model_adam_finite'] is True and worker['scheduler_step_before_actual']==complete
                        and worker['scheduler_step_after_actual']==complete,'Allskip made a write/scheduler change')
                partial_totals[rank]+=actual;totals[rank]+=actual
            require(value['actual_optimizer_calls_per_rank']=={str(r):0 for r in range(8)},'No-credit allskip write totals inconsistent')
            epoch,cursor=sampler_position(attempt)
            require(value['consumed_cursor']==cursor and value['consumed_epoch']==epoch,'Allskip consumed sampler does not include the fetched attempt')
            refused+=1;skipped.append(dict(attempt=attempt,candidate_outer=complete+1,partial_manifest_sha256=psha))
        else:raise ValueError('Fault/unrecognized attempt cannot become completed outer credit')
        opportunities+=20
        if complete==last:break
    require(complete==last and last>=1,'Actual completed outer coverage missing')
    return dict(protocol=PROTOCOL,training_attempts=len(credited)+len(skipped),refused_attempts=refused,completed_outer_updates=complete,
        opportunities_per_rank=opportunities,successful_writes_per_rank=totals,completed_optimizer_calls_per_rank=completed_totals,
        partial_optimizer_calls_per_rank=partial_totals,total_optimizer_calls_per_rank=totals,
        skipped_opportunities_per_rank={str(r):opportunities-totals[str(r)] for r in range(8)},credited=credited,all_skipped=skipped,
        record_sha256=bindings,uncredited_partial_writes_not_inferred=True,automatic_rollback=False)


def checkpoint_accounting(value,observed,outer,run=None):
    require(value['protocol']==PROTOCOL and value['completed_outer_updates']==outer
            and value['training_attempts']==observed['training_attempts'] and value['refused_attempts']==observed['refused_attempts']
            and all(value[key]==observed[key] for key in ('completed_optimizer_calls_per_rank','total_optimizer_calls_per_rank'))
            and {str(r):value['partial_optimizer_calls_per_rank'].get(str(r),0) for r in range(8)}==observed['partial_optimizer_calls_per_rank'],
            'Saved state differs from actual attempt opportunity/write history')
    require(len({x['attempt_id'] for x in value['partial_evidence']})==len(value['partial_evidence'])
            and {x['attempt'] for x in observed['all_skipped']}<={x['attempt_id'] for x in value['partial_evidence']}
            and len(value['partial_evidence'])==value['guard_refused_attempts']
            and sum(not x['credited_outer_candidate'] for x in value['partial_evidence'])==value['refused_attempts'], 'Saved guard histories omit uncredited attempts')
    if run is not None:
        events={x['attempt']:x for x in observed['credited']+observed['all_skipped']}
        for entry in value['partial_evidence']:
            event=events[entry['attempt_id']];candidate=event.get('outer',event.get('candidate_outer'))
            path=Path(entry['path']);require(path.resolve()==(Path(run)/'checkpoints/training_state'/f"partial_attempt_{entry['attempt_id']:06d}_outer_{candidate:06d}"/'manifest.json').resolve(),'Saved partial path differs from actual attempt')
            partial,_,_=read_bound(path,entry['sha256'])
            from partial_raw_evidence import verify_partial_raw_files
            verify_partial_raw_files(run,partial)
            require(partial['continuation_decision']=='CONTINUE_FRESH_ROLLOUT' and partial['worker_states_verified'] is True
                    and partial['verification_error'] is None and partial['credited_outer_candidate']==('outer' in event)
                    and partial['actual_optimizer_calls_per_rank']==entry['optimizer_calls_per_rank']==event.get('actual_writes_per_rank',{str(r):0 for r in range(8)})
                    and len(partial['worker_outcome']['worker_states'])==8
                    and {x['rank'] for x in partial['worker_outcome']['worker_states']}==set(range(8))
                    and all(x['model_adam_finite'] is True for x in partial['worker_outcome']['worker_states']), 'Unsafe/fault partial cannot pass normal training acceptance')
    return value


def sampler_position(attempts):
    require(type(attempts) is int and attempts>0,'Actual consumed attempt count required')
    batches=169028//512
    return (attempts-1)//batches,((attempts-1)%batches+1)*512
