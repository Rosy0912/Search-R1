"""Owned child CPU acceptance queue and JSON-only live/final accounting."""
import json
import os
from pathlib import Path
import sys
import time

from production_contract import (require,read_bound,checkpoint_manifest,hf_reload_report,state_cpu_report,
    direct_rank_records,tree_and_safety_records,validation_reports,publish_external_approval)


class CpuChecks:
    def __init__(self,owned,run,tools,package,config_hash,outer):
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


def training_cost_reports(run,package,config_hash,last):
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


def verify_terminal(run,package,config_hash,checks):
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
