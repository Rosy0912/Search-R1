"""Owned CPU acceptance or explicit human-authorized JSON-only training accounting."""
import json
import math
import os
from pathlib import Path
import sys
import time

from production_contract import (require,read_bound,checkpoint_manifest,hf_reload_report,state_cpu_report,
    direct_rank_records,tree_and_safety_records,validation_reports,publish_external_approval)

DIRECT_MODE='HUMAN_AUTHORIZED_DIRECT_TRAIN200'
DIRECT_SOURCE='a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98'


def direct_training_authority(package):
    """Human instruction is authority to train, never independent acceptance."""
    from production_contract import direct_authority
    pin=package['human_training_authority']
    authority,digest=direct_authority(package)
    require(package['code_commit']==DIRECT_SOURCE,'Human direct-training source differs')
    for record in (package,authority):
        require(all(value is False for key,value in record.items() if 'accepted' in key or key.endswith('_acceptance')),
            'Human training authority cannot claim independent acceptance')
    return dict(path=pin['path'],sha256=digest),authority


def _direct_partial_json(run,entry,expected,attempt,outer,completed,bindings):
    """Bind the original JSON receipt and retain payload references without opening PT."""
    from partial_raw_evidence import raw_file_records
    path=Path(entry['path'])
    require(path.resolve()==(Path(run)/'checkpoints/training_state'/f'partial_attempt_{attempt:06d}_outer_{outer:06d}'/'manifest.json').resolve(),
        'Partial JSON belongs to a different actual attempt')
    partial,digest,_=read_bound(path,entry['sha256']);bindings[str(path)]=digest
    from production_contract import require_identity
    require_identity(partial['identity'],expected)
    require(partial['identity']['attempt_id']==attempt and partial['identity']['attempted_outer']==outer
        and partial['identity']['completed_outer']==completed and partial['completed_outer_updates']==completed
        and partial['continuation_decision']=='CONTINUE_FRESH_ROLLOUT','Partial JSON counter/continuation differs')
    references={str(payload):dict(record) for payload,record,_ in raw_file_records(run,partial)}
    require(all(worker['model_adam_finite'] is True for worker in partial['worker_outcome']['worker_states']),
        'Recorded partial model/Adam finite state missing')
    return partial,dict(path=str(path),sha256=digest,raw_payload_references=references,
        raw_payloads_reopened=False,pt_deserialized=False,independently_accepted=False)


def direct_training_evidence(run,package,config_hash,outer):
    """Actual same-run opportunity/write JSON; allskip attempts earn no outer/warmup credit."""
    authority_pin,_=direct_training_authority(package)
    from attempt_accounting import ledger_writes,sampler_position
    from production_contract import require_identity
    run=Path(run);expected=dict(job_id=run.name.rsplit('-',1)[1],run_id=run.name,code_commit=package['code_commit'],
        config_sha256=config_hash,data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'])
    require(type(outer) is int and 1<=outer<=200,'Actual credited outer boundary required')
    complete=0;totals={str(rank):0 for rank in range(8)};bindings={};credited=[];skipped=[];partials={}
    files=sorted((run/'checkpoints/training_state/attempt-results').glob('attempt-*.json'))
    for attempt,path in enumerate(files,1):
        require(path.name==f'attempt-{attempt:06d}.json','Actual attempt result sequence has a gap')
        value,digest,_=read_bound(path);bindings[str(path)]=digest;candidate=complete+1
        if value['status']=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED':
            result=value['all_rank_result'];require_identity(result['identity'],expected)
            require(value['protocol']==result['protocol']=='tree_fresh_actor_result_v1' and value['automatic_rollback'] is False
                and result['identity']['training_protocol']=='tree_fresh200_five_leaf_v1'
                and result['identity']['attempt_id']==attempt and result['identity']['attempted_outer']==candidate
                and result['identity']['completed_outer']==complete and value['attempt_id']==result['attempt_id']==attempt
                and value['accepted_outer']==result['accepted_outer']==candidate and result['completed_outer_before']==complete
                and result['decision']=='COMPLETED_OUTER','Actual credited attempt/source boundary differs')
            ranks=result['ranks'];require(len(ranks)==8 and [record['rank'] for record in ranks]==list(range(8)),
                'Actual complete rank coverage missing')
            writes={}
            for rank,record in enumerate(ranks):
                actual=ledger_writes(record['opportunity_ledger'])
                require(1<=actual<=20 and record['optimizer_calls']==actual and record['opportunities_processed']==20
                    and record['opportunities_skipped']==20-actual and record['scheduler_before']==complete
                    and record['scheduler_after']==candidate and record['attempt_id']==attempt
                    and record['accepted_outer']==candidate and record['completed_outer_before']==complete,
                    'Credited rank opportunity/write/scheduler mismatch')
                descriptor=record['direct_rank_record'];physical_path=run/'output/fresh-state'/f'outer-{candidate:06d}-attempt-{attempt:06d}-rank{rank:02d}.json'
                require(Path(descriptor['path']).resolve()==physical_path.resolve() and descriptor['rank']==rank
                    and descriptor['actual_optimizer_calls']==actual and descriptor['scheduler_step_before']==complete
                    and descriptor['scheduler_step_after']==candidate,'Gathered direct-rank reference differs')
                physical,dsha,_=read_bound(physical_path,descriptor['sha256']);bindings[str(physical_path)]=dsha
                require_identity(physical,expected)
                require(physical['status']=='DIRECT_RANK_COMPLETE_OUTER_OBSERVED' and physical['training_protocol']=='tree_fresh200_five_leaf_v1'
                    and physical['rank']==rank and physical['world_size']==8 and physical['attempt_id']==attempt
                    and physical['observed_outer']==candidate and physical['actual_optimizer_calls']==physical['returned_optimizer_calls']==actual
                    and physical['opportunities_processed']==20 and physical['opportunities_skipped']==20-actual
                    and physical['optimizer_opportunity_ledger']==record['opportunity_ledger']
                    and physical['scheduler_step_before']==complete and physical['scheduler_step_after']==candidate
                    and physical['scheduler_step_invoked'] is True and physical['all_rank_before_admitted'] is True
                    and physical['all_rank_scheduler_after_valid'] is True and physical['local_before_error'] is None
                    and physical['local_scheduler_error'] is None,'Direct rank JSON actual counts differ')
                require(physical['learning_rates_before']==[1e-6*min(complete/57,1)]
                    and physical['learning_rates_after']==[1e-6*min(candidate/57,1)],'Warmup advanced outside completed credit')
                writes[str(rank)]=actual;totals[str(rank)]+=actual
                require(record['total_optimizer_calls']==totals[str(rank)],'Cumulative actual Adam return counter differs')
            require(len(set(writes.values()))==1 and value['actual_completed_optimizer_calls_per_rank']==totals,
                'Actual all-rank completed write accounting differs')
            complete=candidate;credited.append(dict(outer=complete,attempt=attempt,actual_writes_per_rank=writes,result_sha256=digest))
        elif value['status']=='ALL_OPPORTUNITIES_SKIPPED_NO_OUTER_CREDIT':
            ids=value['identity'];require_identity(ids,expected)
            require(value['protocol']=='tree_partial_state_v1' and ids['attempt_id']==attempt
                and ids['attempted_outer']==candidate and ids['completed_outer']==complete,'Allskip altered completed credit')
            partial,reference=_direct_partial_json(run,value['partial_evidence'],expected,attempt,candidate,complete,bindings)
            require(partial['identity']==ids and partial['credited_outer_candidate'] is False,'Allskip cannot earn candidate credit')
            for worker in partial['worker_outcome']['worker_states']:
                require(ledger_writes(worker['optimizer_opportunity_ledger'])==worker['committed_optimizer_calls']==0
                    and worker['safety_guard'] is True and worker['scheduler_step_before_actual']==complete
                    and worker['scheduler_step_after_actual']==complete,'Allskip changed Adam/scheduler')
            require(value['actual_optimizer_calls_per_rank']=={str(rank):0 for rank in range(8)},'Allskip actual writes are not zero')
            epoch,cursor=sampler_position(attempt)
            require(value['consumed_cursor']==cursor and value['consumed_epoch']==epoch,'Allskip sampler attempt omitted')
            partials[str(reference['path'])]=reference
            skipped.append(dict(attempt=attempt,candidate_outer=candidate,partial_manifest_sha256=reference['sha256']))
            writes={str(rank):0 for rank in range(8)}
        else:raise ValueError('Fault/abandoned attempt cannot become a completed outer')
        for rank in range(8):
            audit_path=run/'output/tree-rank-audit'/f'step-{candidate:06d}-attempt-{attempt:06d}-rank-{rank:02d}.json'
            audit,asha,_=read_bound(audit_path);bindings[str(audit_path)]=asha
            require(all(audit[key]==expected[key] for key in ('job_id','run_id','code_commit','config_sha256','data_identity'))
                and audit['outer_step']==candidate and audit['attempt_id']==attempt and audit['rank']==rank and audit['world_size']==8
                and audit['local_rows']==320 and audit['local_minibatch']==16 and audit['local_microbatch']==2 and audit['epochs']==1
                and audit['successful_optimizer_calls']==writes[str(rank)] and audit['skipped_empty_minibatches']==20-writes[str(rank)]
                and audit['finite_grad_norms']==writes[str(rank)] and len(audit['existing_grad_norms'])==writes[str(rank)]
                and all(type(number) in (int,float) and math.isfinite(number) for number in audit['existing_grad_norms'])
                and audit['local_shard_sample_layout_unchanged'] is True,'Actual same-run rank finite/write accounting differs')
        if complete==outer:break
    require(complete==outer,'Actual completed outer coverage missing')
    attempts=len(credited)+len(skipped);opportunities=attempts*20
    observed=dict(protocol='tree_guard_continue_fresh_rollout_v1',training_attempts=attempts,refused_attempts=len(skipped),
        completed_outer_updates=complete,opportunities_per_rank=opportunities,successful_writes_per_rank=totals,
        completed_optimizer_calls_per_rank=totals,partial_optimizer_calls_per_rank={str(rank):0 for rank in range(8)},
        total_optimizer_calls_per_rank=totals,skipped_opportunities_per_rank={str(rank):opportunities-totals[str(rank)] for rank in range(8)},
        credited=credited,all_skipped=skipped,record_sha256=bindings,uncredited_partial_writes_not_inferred=True,automatic_rollback=False)
    return dict(completed_outer=outer,training_attempts=attempts,direct_rank_records=outer*8,tree_rank_records=attempts*8,
        actual_optimizer_returns=sum(totals.values()),opportunities_processed=opportunities*8,actual_writes_per_rank=totals,
        skipped_opportunities_per_rank=observed['skipped_opportunities_per_rank'],attempt_accounting=observed,
        record_sha256=bindings,partial_evidence=partials,human_training_authority=authority_pin,
        evidence_scope='actual same-run JSON counters and finite receipts; raw payload references retained',
        cpu_checks_run=False,pt_deserialized=False,full_hf_reloaded=False,independently_accepted=False)


class CpuChecks:
    def __init__(self,owned,run,tools,package,config_hash,outer):
        require(package.get('execution_mode')!=DIRECT_MODE,'Direct human-authorized training does not run CPU acceptance checks')
        self.owned,self.run,self.tools,self.package,self.config_hash,self.outer=owned,Path(run),Path(tools),package,config_hash,outer
        self.directory=self.run/'output'/f'cpu-acceptance-{outer}';self.directory.mkdir()
        self.manifest,self.manifest_sha=checkpoint_manifest(self.run,package,config_hash,outer)
        manifest_path=self.run/'checkpoints/training_state'/f'global_step_{outer}'/'manifest.json'
        root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
        self.commands=[('full-hf',[sys.executable,str(root/'tools/check_real_checkpoint.py'),'--base',package['official_model_path'],
            '--checkpoint',self.manifest['hf_checkpoint'],'--output',str(self.directory/'full-hf.json')]),
            ('complete-state',[sys.executable,str(tools/'check_training_state_cpu.py'),'--manifest',str(manifest_path),
             '--manifest-sha',self.manifest_sha,'--output',str(self.directory/'complete-state.json')]),
            ('behavior-mu',[sys.executable,str(tools/'check_behavior_mu_cpu.py'),'--run',str(self.run),'--package',str(tools),
             '--config-sha',config_hash,'--outer',str(outer),'--output',str(self.directory/'behavior-mu.json')]),
            ('partial-raw',[sys.executable,str(tools/'check_partial_raw_cpu.py'),'--run',str(self.run),'--package',str(tools),
             '--config-sha',config_hash,'--outer',str(outer),'--output',str(self.directory/'partial-raw.json')])]
        self.index=0;self.process=None;self.complete=False;self.start_epoch=time.time();self.spawn_next()

    def spawn_next(self):
        name,command=self.commands[self.index]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',TREE_FIXED_ACTOR_PROFILE='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONOPTIMIZE='0')
        self.process=self.owned.spawn(command,self.directory/(name+'.log'),environment)

    def poll(self):
        if self.complete:return True
        if self.process.poll() is None:return False
        require(self.process.returncode==0,'Actual same-node CPU checker failed: '+self.commands[self.index][0])
        self.index+=1
        if self.index==len(self.commands):
            self.complete=True
            with (self.directory/'cost-scope.json').open('x') as stream:json.dump(dict(start_epoch=self.start_epoch,end_epoch=time.time(),outer=self.outer,
                scope='Serial actual full HF CPU reload, complete saved state, and real mu binding validation in own8GPU allocation; no additional GPU model forward/backward/optimizer',gpu_forward_calls=0),stream)
            return True
        self.spawn_next();return False

    def proofs(self):
        require(self.complete,'Actual CPU checkers have not completed')
        hfsha=hf_reload_report(self.directory/'full-hf.json',self.package['official_model_path'],self.manifest['hf_checkpoint'],self.manifest['hf_identity'])
        statesha=state_cpu_report(self.directory/'complete-state.json',self.manifest_sha,self.outer)
        mu,musha,_=read_bound(self.directory/'behavior-mu.json')
        require(mu['status']=='FRESH200_REAL_BEHAVIOR_MU_CPU_BINDING_VERIFIED' and mu['completed_outers_checked']==self.outer
                and mu['real_DataProto'] is True and mu['optimizer_writes']==mu['model_forwards']==0
                and mu['cuda_initialized'] is False and mu['original_sampler_actor_gate_accepted'] is False,'Actual full mu CPU binding evidence missing')
        partial,partial_sha,_=read_bound(self.directory/'partial-raw.json')
        from production_contract import require_identity
        expected=dict(job_id=self.run.name.rsplit('-',1)[1],run_id=self.run.name,code_commit=self.package['code_commit'],
            config_sha256=self.config_hash,data_identity=self.package['common_data_identity'],model_identity=self.package['model_identity']['sha256'])
        require_identity(partial['identity'],expected)
        require(partial['status']=='FRESH200_PARTIAL_RAW_CPU_REVERIFIED' and partial['completed_outer']==self.outer
            and partial['checkpoint_manifest_sha256']==self.manifest_sha and partial['same_fd_io'] is True
            and partial['optimizer_writes']==partial['model_forwards']==0 and partial['cuda_initialized'] is False,
            'Retained historical raw partial byte/logical verification absent')
        from partial_raw_evidence import verify_partial_raw_files
        raw_mapping={}
        entries=self.manifest['identity']['guard_continue_accounting']['partial_evidence']
        for entry in entries:
            saved,_,_=read_bound(entry['path'],entry['sha256']);raw_mapping.update(verify_partial_raw_files(self.run,saved))
        require(partial['raw_files_sha256']==raw_mapping and partial['raw_files_checked']==10*len(entries)
            and partial['partial_attempts_checked']==len(entries) and len(partial['logical_records'])==len(entries)
            and partial['pt_deserialization_count']==10*len(entries)
            and partial['actual_pt_deserialization']==bool(entries),'Actual raw report coverage/deserialization differs from retained checkpoint history')
        return dict(partial_raw_cpu_report_sha256=partial_sha,partial_raw_files_sha256=partial['raw_files_sha256'],full_state_manifest_sha256=self.manifest_sha,hf_reload_report_sha256=hfsha,state_cpu_report_sha256=statesha,
                    behavior_mu_cpu_report_sha256=musha,rank_counts=direct_rank_records(self.run,self.package,self.config_hash,self.outer),
                    tree_and_safety=tree_and_safety_records(self.run,self.outer,self.package,self.config_hash),training_cost=training_cost_reports(self.run,self.package,self.config_hash,self.outer),development=validation_reports(self.run,self.outer,package=self.package,config_hash=self.config_hash))


def training_cost_reports(run,package,config_hash,last,observed=None):
    if observed is None:
        from attempt_accounting import attempt_results
        observed=attempt_results(run,package,config_hash,last)
    expected=dict(job_id=Path(run).name.rsplit('-',1)[1],run_id=Path(run).name,code_commit=package['code_commit'],
        config_sha256=config_hash,data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'])
    from production_contract import require_identity
    import math
    groups={'credited':{},'all_skipped':{}};records={}
    for event in sorted(observed['credited']+observed['all_skipped'],key=lambda x:x['attempt']):
        path=Path(run)/'output/training-rollout-cost'/f"attempt-{event['attempt']:06d}.json"
        value,sha,_=read_bound(path);require_identity(value,expected)
        candidate=event.get('outer',event.get('candidate_outer'))
        require(value['status']=='TRAINING_ROLLOUT_COST_COMPLETE' and value['attempt_id']==event['attempt']
            and value['candidate_outer']==candidate and value['completed_outer_before']==candidate-1
            and value['question_rows']==512 and value['actual_optimizer_writes'] is None and value['outer_credit_not_inferred'] is True,
            'Real training rollout cost belongs to a different actual attempt/credit')
        costs=value['costs'];require({'generated_tokens','prefill_tokens','retrieval_calls','raw_emitted_tokens','padded_rows'}<=set(costs)
            and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in costs.values()),'Actual training rollout costs are incomplete/nonfinite')
        group=groups['credited' if 'outer' in event else 'all_skipped']
        for key,number in costs.items():group[key]=group.get(key,0)+number
        records[str(path)]=sha
    return dict(training_attempts=observed['training_attempts'],complete_outer_credits=last,costs_by_persisted_credit=groups,
        record_sha256=records,source='original Controller.generate metrics',failed_unrecorded_costs_not_inferred=True,
        optimizer_writes_taken_from_direct_records_separately=True)


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


def _direct_checkpoint_json(run,package,config_hash,outer,rank_counts):
    """Saved JSON references/finite counters, without PT reload or HF file reread."""
    from attempt_accounting import ledger_writes
    run=Path(run);directory=run/'checkpoints/training_state'/f'global_step_{outer}'
    manifest,digest,_=read_bound(directory/'manifest.json')
    expected=dict(job_id=run.name.rsplit('-',1)[1],code_commit=package['code_commit'],config_sha256=config_hash,
        data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'],
        completed_step=outer,actual_update_limit=200)
    identity=manifest['identity'];accounting=identity['guard_continue_accounting']
    require(manifest['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and identity==dict(expected,guard_continue_accounting=accounting),
        'Saved checkpoint JSON source/split/completed boundary differs')
    history=rank_counts['attempt_accounting'];last_attempt=history['credited'][outer-1]['attempt']
    events={event['attempt']:event for event in history['credited']+history['all_skipped'] if event['attempt']<=last_attempt}
    complete_totals={str(rank):sum(event['actual_writes_per_rank'][str(rank)] for event in history['credited'] if event['outer']<=outer)
        for rank in range(8)}
    refused={event['attempt'] for event in history['all_skipped'] if event['attempt']<=last_attempt}
    require(accounting['protocol']=='tree_guard_continue_fresh_rollout_v1' and accounting['completed_outer_updates']==outer
        and accounting['training_attempts']==last_attempt and accounting['refused_attempts']==len(refused)
        and accounting['completed_optimizer_calls_per_rank']==accounting['total_optimizer_calls_per_rank']==complete_totals
        and {str(rank):accounting['partial_optimizer_calls_per_rank'].get(str(rank),0) for rank in range(8)}=={str(rank):0 for rank in range(8)},
        'Saved checkpoint JSON actual attempt/write accounting differs')
    partials=accounting['partial_evidence'];partial_ids={entry['attempt_id'] for entry in partials}
    require(len(partial_ids)==len(partials)==accounting['guard_refused_attempts'] and refused<=partial_ids
        and sum(not entry['credited_outer_candidate'] for entry in partials)==len(refused),
        'Saved checkpoint JSON drops actual guard/partial history')
    run_identity=dict(expected,run_id=run.name);references={}
    for entry in partials:
        event=events[entry['attempt_id']];candidate=event.get('outer',event.get('candidate_outer'))
        partial,reference=_direct_partial_json(run,entry,run_identity,entry['attempt_id'],candidate,candidate-1,rank_counts['record_sha256'])
        writes=event.get('actual_writes_per_rank',{str(rank):0 for rank in range(8)})
        require(partial['credited_outer_candidate']==entry['credited_outer_candidate']==('outer' in event)
            and partial['actual_optimizer_calls_per_rank']==entry['optimizer_calls_per_rank']==writes,
            'Saved partial JSON cannot alter credited/uncredited actual writes')
        for worker in partial['worker_outcome']['worker_states']:
            require(ledger_writes(worker['optimizer_opportunity_ledger'])==worker['committed_optimizer_calls']==writes[str(worker['rank'])]
                and worker['scheduler_step_before_actual']==candidate-1
                and worker['scheduler_step_after_actual']==candidate-1+int(writes[str(worker['rank'])]>0),
                'Saved partial JSON opportunity/scheduler accounting differs')
        references[str(reference['path'])]=reference
    workers=sorted(manifest['worker_states'],key=lambda record:record['identity']['rank'])
    require(len(workers)==8 and [record['identity']['rank'] for record in workers]==list(range(8)),
        'Saved JSON rank coverage incomplete')
    for rank,record in enumerate(workers):
        worker_identity=record['identity'];visible=worker_identity.get('cuda_visible_devices')
        require(type(visible) is str and visible and worker_identity==dict(identity,rank=rank,world_size=8,cuda_visible_devices=visible)
            and record['full_worker_state_finite'] is True and record['scheduler_last_epoch']==outer
            and Path(record['path']).resolve()==(directory/f'rank-{rank:02d}.pt').resolve(),
            'Saved rank JSON finite/scheduler/payload reference differs')
    require(manifest['driver']['next_step']==outer+1 and Path(manifest['driver']['path']).resolve()==(directory/'driver.pt').resolve()
        and Path(manifest['hf_checkpoint']).resolve()==(run/'checkpoints/actor'/f'global_step_{outer}').resolve(),
        'Saved driver/HF reference belongs to another actual boundary')
    return dict(manifest_sha256=digest,hf_identity=manifest['hf_identity'],hf_checkpoint=manifest['hf_checkpoint'],
        worker_payload_references=workers,driver_payload_reference=manifest['driver'],partial_evidence=references,
        json_finite_counters_observed=True,raw_payloads_reopened=False,pt_deserialized=False,
        full_hf_reloaded=False,independently_accepted=False)


def _direct_initial_validation(run,package,config_hash):
    overrides=package['configuration_overrides']
    if 'trainer.val_before_train=false' not in overrides:
        return validation_reports(run,0,package=package,config_hash=config_hash)
    require('trainer.val_before_train=true' not in overrides,'Conflicting direct initial evaluation configuration')
    historical=package['historical_initial_validation'];pin=historical['source_report']
    historical_sha='9a7cf5a3251aa54ea68b5bfdebc1c6a6a2eced2f88fdd4404af8ec8dc24bd947'
    source_commit='53082e1667a6ab131a036ca2ef66e004989ed1a9'
    require(historical['kind']=='HISTORICAL_OFFICIAL_FRESH_BASE_DEV512' and historical['job_id']=='3317'
        and historical['source_code_commit']==source_commit and historical['evaluated_rows']==512
        and historical['model_identity']==package['model_identity']['sha256']
        and historical['data_identity']==package['common_data_identity']
        and historical['current_run_evaluation_executed'] is False and historical['independent_acceptance'] is False
        and pin['sha256']==historical_sha,'Historical official fresh baseline descriptor differs')
    from production_contract import ROOT
    require(Path(pin['path']).resolve()==(ROOT/'runs/tree-fresh200-3317/output/validation-cost/validation-1377126-1.json').resolve(),
        'Historical baseline source report reference differs')
    packaged_path=Path(__file__).resolve().parent/'historical-inputs/initial-dev3317.json'
    require(package['files']['historical-inputs/initial-dev3317.json']==historical_sha,
        'Historical baseline exact package copy is not pinned')
    observed,digest,_=read_bound(packaged_path,historical_sha)
    require(observed['status']=='VALIDATION_COST_COMPLETE' and observed['job_id']=='3317'
        and observed['code_commit']==source_commit and observed['model_identity']==historical['model_identity']
        and observed['data_identity']==historical['data_identity'] and observed['actual_completed_outer']==observed['global_step']==0
        and observed['phase']=='fresh_initial' and observed['evaluated_rows']==observed['dataset_rows']==512
        and observed['metrics']['val/qa_count/all']==512
        and historical['metrics']=={'val/f1/all':0.23400209323528842,'val/em/all':0.177734375}
        and all(observed['metrics'][key]==value for key,value in historical['metrics'].items()),
        'Historical baseline report identity/count/metrics differ')
    counts={key.rsplit('/',1)[1]:value for key,value in observed['metrics'].items() if key.startswith('val/evaluated_count/')}
    require(sum(counts.values())==512,'Historical baseline dataset row counts incomplete')
    return dict(kind=historical['kind'],phase='historical_fresh_initial',observed_job_id='3317',source_code_commit=source_commit,
        source_report=dict(pin),packaged_source_report=dict(path=str(packaged_path),sha256=digest),evaluated_rows=512,
        per_dataset=counts,metrics=observed['metrics'],model_identity=historical['model_identity'],data_identity=historical['data_identity'],
        current_run_evaluation_executed=False,independent_acceptance=False,
        scope='Historical official fresh-base dev512; no new initial evaluation in this run')


def verify_direct_terminal(run,package,config_hash):
    """Completion evidence only; human continuation never substitutes reviewer PASS."""
    authority_pin,_=direct_training_authority(package)
    ranks=direct_training_evidence(run,package,config_hash,200)
    final=validation_reports(run,200,final=True,package=package,config_hash=config_hash)
    initial=_direct_initial_validation(run,package,config_hash)
    periodic=[validation_reports(run,outer,package=package,config_hash=config_hash) for outer in range(5,201,5)]
    require(final['evaluated_rows']==51713 and len(final['per_dataset'])==7
        and all(type(count) is int and count>0 for count in final['per_dataset'].values())
        and sum(final['per_dataset'].values())==51713,'Actual final seven-dataset row counts incomplete')
    saved={outer:_direct_checkpoint_json(run,package,config_hash,outer,ranks) for outer in (5,50,100,150,200)}
    continuation_path=Path(run)/'output/acceptance-step5/human-authorized-continuation.json'
    continuation,csha,_=read_bound(continuation_path)
    from production_contract import require_identity
    expected=dict(job_id=Path(run).name.rsplit('-',1)[1],run_id=Path(run).name,code_commit=package['code_commit'],
        config_sha256=config_hash,data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'])
    require_identity(continuation,expected)
    ready,rsha,_=read_bound(continuation_path.with_name('ready.json'),continuation['ready_sha256'])
    require_identity(ready,expected,include_run=False)
    require(continuation['status']=='HUMAN_AUTHORIZED_SAME_INSTANCE_6_TO_200' and continuation['completed_outer_updates']==5
        and continuation['next_outer_update']==6 and continuation['actual_update_limit']==200
        and continuation['same_instance_continuation'] is True
        and continuation['human_training_authority_sha256']==authority_pin['sha256']
        and ready['completed_outer_updates']==ready['completed_step']==5 and ready['next_outer_update']==6
        and ready['actual_update_limit']==200,
        'Actual step5 same-instance human continuation missing')
    require(all(value is False for key,value in continuation.items() if 'accepted' in key or key.endswith('_acceptance')),
        'Direct continuation cannot claim independent five acceptance')
    progress=progress_evidence(run)
    require(progress['actual_completed_outer_from_attempt_results']==200 and not progress['errors'],
        'Complete direct run contains incomplete/fault attempt evidence')
    cost=training_cost_reports(run,package,config_hash,200,observed=ranks['attempt_accounting'])
    return dict(status='ACTUAL_HUMAN_AUTHORIZED_DIRECT_FRESH200_TRAINING_COMPLETE',execution_mode=DIRECT_MODE,
        actual_completed_outer=200,direct_rank_records=ranks['direct_rank_records'],tree_rank_records=ranks['tree_rank_records'],
        actual_rank_optimizer_returns=ranks['actual_optimizer_returns'],opportunities_processed=ranks['opportunities_processed'],
        actual_calls_per_rank=ranks['actual_writes_per_rank'],skipped_opportunities_per_rank=ranks['skipped_opportunities_per_rank'],
        training_attempts=ranks['training_attempts'],same_instance_continuation=True,human_training_authority=authority_pin,
        first5_human_continuation_path=str(continuation_path),first5_human_continuation_sha256=csha,first5_ready_sha256=rsha,
        first5_independently_accepted=False,saved_checkpoints=saved,final=final,initial=initial,periodic=periodic,
        rank_counts=ranks,training_cost=cost,progress=progress,json_only_terminal_accounting=True,
        cpu_checks_run=False,pt_deserialized=False,full_hf_reloaded=False,actual_zero_ready=False,
        actual_calibration_accepted=False,bootstrap_thresholds_accepted=False,production_thresholds_accepted=False,
        independent_acceptance=False,training_accepted=False,actual_training_accepted=False,
        final_effect_independently_accepted=False,original_sampler_actor_probability_gate_accepted=False,GPU_capacity_proven=False)


def verify_terminal(run,package,config_hash,checks=None):
    if package.get('execution_mode')==DIRECT_MODE:
        return verify_direct_terminal(run,package,config_hash)
    require(checks.outer==200 and checks.complete,'Final actual200 CPU checks incomplete')
    proofs=checks.proofs();final=validation_reports(run,200,final=True,package=package,config_hash=config_hash)
    initial=validation_reports(run,0,package=package,config_hash=config_hash)
    periodic=[validation_reports(run,outer,package=package,config_hash=config_hash) for outer in range(5,201,5)]
    saved={}
    for outer in (5,50,100,150,200):
        manifest,digest=checkpoint_manifest(run,package,config_hash,outer)
        from search_r1.baselines.select_teacher import model_manifest
        require(model_manifest(manifest['hf_checkpoint'])==manifest['hf_identity'],'Saved HF checkpoint changed after training')
        saved[outer]=dict(manifest_sha256=digest,hf_identity=manifest['hf_identity'])
    released,rsha,_=read_bound(Path(run)/'output/acceptance-step5/released.json')
    require(released['completed_outer_updates']==5 and released['next_outer_update']==6,'No actual same-instance five-gate release')
    approval,asha,_=read_bound(Path(run)/'output/acceptance-step5/approved.json')
    require(released['approval_sha256']==asha,'Actual live release does not bind independent approval')
    first5_proofs,_,_=read_bound(Path(run)/'output/acceptance-step5/wrapper-evidence.json')
    first5=publish_external_approval(run,package,config_hash,first5_proofs)
    require(first5 and first5['full_run_thresholds_accepted'] is True and first5['review_sha256']==asha,'Final continuation lacks actual short-training threshold acceptance')
    progress=progress_evidence(run);require(progress['actual_completed_outer_from_attempt_results']==200 and not progress['errors'],'Complete run contains partial/unknown failure evidence')
    return dict(status='ACTUAL_FRESH200_TRAINING_COMPLETE_PENDING_FINAL_INDEPENDENT_REVIEW',actual_completed_outer=200,
                direct_rank_records=proofs['rank_counts']['direct_rank_records'],tree_rank_records=proofs['rank_counts']['tree_rank_records'],actual_rank_optimizer_returns=proofs['rank_counts']['actual_optimizer_returns'],opportunities_processed=proofs['rank_counts']['opportunities_processed'],actual_calls_per_rank=proofs['rank_counts']['actual_writes_per_rank'],training_attempts=proofs['rank_counts']['training_attempts'],first5_gate_release_sha256=rsha,first5_threshold_assessment_sha256=first5['threshold_assessment_sha256'],
                first5_gate_legacy_literal_status=released['status'],same_instance_continuation=True,saved_checkpoints=saved,
                final=final,initial=initial,periodic=periodic,final_cpu_proofs=proofs,progress=progress,
                final_effect_independently_accepted=False,original_sampler_actor_probability_gate_accepted=False,GPU_capacity_proven=False)
