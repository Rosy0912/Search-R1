"""Strict JSON contracts for a future actual fresh200 run; no model/PT imports.

New acceptance/threshold envelopes here are proposed APIs, not existing receipts.
The worker/state/HF/validation JSON consumers below use actual frozen APIs.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PROFILE = 'tree_fresh200_five_leaf_v1'
SOURCE = '62271081ba1bb74d8fc3f4958d623758efa74fa5'
CHAIN_SCOPE = 'CURRENT_FRESH200_CHAIN_CONTINUATION'
DIRECT_MODE = 'HUMAN_AUTHORIZED_DIRECT_TRAIN200'
DIRECT_THRESHOLDS = dict(target_old_policy_kl=0.2,negative_tail_limit=0.01,initial_actor_repeat_max_abs_delta=0.0)
THRESHOLD_FIELDS = ('target_old_policy_kl','negative_tail_limit','initial_actor_repeat_max_abs_delta')
INPUT_NAMES = ('actual_calibration_ready','actual_calibration_acceptance','threshold_acceptance',
               'source_context_free_review')


def require(condition, message):
    if not condition: raise ValueError(message)


def read_bound(path, expected=None):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), 'Pinned evidence must be ordinary file: '+str(path))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno()); data = stream.read(); after = os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                'Evidence changed while held: '+str(path))
    digest = hashlib.sha256(data).hexdigest()
    require(expected is None or digest == expected, 'Pinned evidence bytes changed: '+str(path))
    return json.loads(data), digest, data


def descriptor(value, trusted_root=ROOT):
    require(type(value) is dict and set(value) == {'path','sha256'}, 'Actual path/SHA descriptor required')
    require(type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}',value['sha256']), 'Actual SHA pin required')
    path = Path(value['path'])
    require(path.is_absolute() and path.resolve().is_relative_to(Path(trusted_root).resolve()), 'Evidence must remain on this own h100-10 task filesystem')
    return read_bound(path,value['sha256'])


def require_identity(record, expected, *, include_run=True):
    keys = ('job_id','code_commit','config_sha256','data_identity','model_identity') + (('run_id',) if include_run else ())
    require(all(record.get(key) == expected[key] for key in keys), 'Actual source/run identity differs')


def observed_production_inputs(package,values):
    """Validate four actual JSON byte-bound inputs before package review exists."""
    require(set(values)==set(INPUT_NAMES), 'Actual four observation/review inputs required')
    ready = values['actual_calibration_ready'][0]
    require(ready['status'] == 'FRESH_BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES'
            and ready['calibration_profile'] == 'fresh_base_zero_v1'
            and ready['initialized_global_steps'] == ready['parent_completed_outer'] == ready['new_completed_outer'] == ready['optimizer_committed'] == 0
            and ready['scheduler_advanced'] is False and ready['observed_outer'] == 1
            and ready['behavior_mu_corrected_profile'] is True
            and ready['original_sampler_actor_probability_gate_accepted'] is False,
            'Actual official fresh zero-write calibration is not proven')
    require(ready['code_commit'] == package['code_commit'] and type(package['code_commit']) is str and re.fullmatch('[0-9a-f]{40}',package['code_commit'])
            and ready['data_identity'] == package['common_data_identity']
            and ready['model_identity'] == package['model_identity']['sha256'], 'Calibration source/model/split differs from training candidate')
    workers = sorted(ready['workers'],key=lambda x:x['rank'])
    require(len(workers)==8 and [x['rank'] for x in workers]==list(range(8)), 'Actual calibration rank coverage incomplete')
    for worker in workers:
        require(worker['world_size']==8 and worker['optimizer_committed']==0
                and worker['optimizer_state_entries']==0 and worker['restored_optimizer_steps']==[]
                and worker['scheduler_step']==0 and worker['scheduler_advanced'] is False
                and worker['fresh_post_monitor_state']==dict(calibration_profile='fresh_base_zero_v1',restored_optimizer_steps=[],optimizer_state_entries=0,scheduler_step=0),
                'Actual calibration worker inherited/changed state')
    accepted = values['actual_calibration_acceptance'][0]
    require(accepted['status']=='INDEPENDENT_ACTUAL_FRESH200_MU_CALIBRATION_ACCEPTED'
            and accepted['fork_turns']=='none' and accepted['calibration_ready_sha256']==values['actual_calibration_ready'][1]
            and accepted['actual_zero_writes_accepted'] is True and accepted['actual_behavior_mu_binding_accepted'] is True
            and accepted['fresh_initialization_accepted'] is True and accepted['original_sampler_actor_gate_accepted'] is False,
            'Actual calibration independent acceptance missing')
    thresholds = values['threshold_acceptance'][0]
    require(thresholds['status']=='INDEPENDENT_FRESH200_MU_BOOTSTRAP_THRESHOLDS_ACCEPTED'
            and thresholds['fork_turns']=='none' and thresholds['bootstrap_only'] is True and thresholds['first_credited_outers']==[1,2,3,4,5]
            and thresholds['production_thresholds_accepted'] is False and thresholds['continuation_6_200_accepted'] is False
            and thresholds['calibration_ready_sha256']==values['actual_calibration_ready'][1]
            and thresholds['calibration_acceptance_sha256']==values['actual_calibration_acceptance'][1]
            and thresholds['code_commit']==package['code_commit'] and thresholds['training_protocol']==PROFILE
            and thresholds['original_sampler_actor_gate_accepted'] is False,
            'Only explicit actual-zero-bound first-five bootstrap candidates may initialize training')
    parameters = thresholds['bootstrap_thresholds']
    require(set(parameters)==set(THRESHOLD_FIELDS) and all(type(x) in (int,float) and math.isfinite(x) for x in parameters.values()), 'Production thresholds cannot be null/default/nonfinite')
    require(parameters['target_old_policy_kl']==0.2 and 0<=parameters['negative_tail_limit']<=1 and parameters['initial_actor_repeat_max_abs_delta']>=0, 'Production KL must use latest explicit user value 0.2; remaining threshold range invalid')
    monitor_hashes={str(worker['rank']):worker['monitor_sha256'] for worker in workers}
    require(thresholds['zero_monitor_sha256']==monitor_hashes,'Bootstrap candidates do not bind all actual zero monitors')
    observations=thresholds['zero_rank_observations']
    require(type(observations) is list and [x['rank'] for x in observations]==list(range(8)), 'Actual zero threshold observations missing')
    for record in observations:
        require(record['monitor_sha256']==monitor_hashes[str(record['rank'])]
                and type(record['effective_tokens']) is int and record['effective_tokens']>0
                and all(type(record[k]) in (int,float) and math.isfinite(record[k]) for k in
                        ('sampled_old_policy_kl','negative_tail_fraction','initial_actor_repeat_max_abs_delta'))
                and 0<=record['negative_tail_fraction']<=1 and record['initial_actor_repeat_max_abs_delta']>=0,
                'Bootstrap observation metrics are missing/invalid')
    require(type(thresholds['assessment_rationale']) is str and thresholds['assessment_rationale'].strip(), 'Bootstrap candidate rationale absent')
    authority=thresholds['target_kl_authority']
    require(authority['kind']=='explicit_latest_human_goal_edit' and authority['value']==0.2
            and type(authority['original_instruction_time']) is str and authority['original_instruction_time']
            and type(authority['source']) is str and authority['source'], 'Original user KL instruction time/source must be recorded, not inferred from receipt replay')
    review = values['source_context_free_review'][0]
    compact=(review.get('status')=='CONTEXT_FREE_FRESH200_SOURCE_ACCEPTED' and review.get('fork_turns')=='none'
             and review.get('code_commit')==package['code_commit'] and review.get('blocking_findings')==[])
    sealed=(values['source_context_free_review'][1]=='6f2ceaebd1ff6c04e9330b404e17ff7db9df7493db5c686cbc3366cb3253fb44'
            and review.get('status')=='FRESH200_F7_V9_V6_SOURCE_CPU_CONTEXT_FREE_REVIEWED'
            and review['reviewer_context']['fork_turns']=='none' and review['reviewer_context']['inherited_parent_conversation'] is False
            and review['source']['audited_clean_head']==package['code_commit'] and review['source']['final_clean'] is True
            and review['source']['source_review_accepted'] is True and review['source_interface_review_accepted'] is True and review['findings']==[])
    require(compact or sealed,'Context-free actual source acceptance missing; zero/helper approval never substitutes production review')
    return parameters


def production_inputs(package, *, manifest_sha256, dispatch_review_descriptor, trusted_root=ROOT):
    if package.get('execution_mode') == DIRECT_MODE:
        return human_direct_inputs(package,manifest_sha256=manifest_sha256,
                                   release_descriptor=dispatch_review_descriptor,trusted_root=trusted_root)
    inputs = package['production_inputs']
    require(type(inputs) is dict and set(inputs) == set(INPUT_NAMES), 'All four manifest production inputs required; fifth dispatch review is separately pinned to avoid self-reference')
    values = {name:descriptor(inputs[name],trusted_root) for name in INPUT_NAMES}
    parameters=observed_production_inputs(package,values)
    dispatch,dispatch_sha,_ = descriptor(dispatch_review_descriptor,trusted_root)
    require(dispatch['status']=='READY_FOR_FRESH200_BEHAVIOR_MU_FIRST5_BOOTSTRAP' and dispatch['bootstrap_only'] is True
            and dispatch['full_run_thresholds_accepted'] is False and dispatch['fork_turns']=='none'
            and dispatch['manifest_sha256']==manifest_sha256 and dispatch['code_commit']==package['code_commit']
            and dispatch['bootstrap_thresholds_sha256']==values['threshold_acceptance'][1]
            and dispatch['source_review_sha256']==values['source_context_free_review'][1]
            and dispatch['actual_calibration_acceptance_sha256']==values['actual_calibration_acceptance'][1]
            and dispatch['original_sampler_actor_gate_accepted'] is False,
            'Independent whole-package first-five bootstrap release missing')
    require(dispatch['cluster_name']=='wuhan-h100' and dispatch['node_list']=='h100-10'
            and dispatch['dispatch_renderer_sha256']==package['files']['render_dispatch.py']
            and dispatch['dispatch_submitter_sha256']==package['files']['submit_once.py'],'Independent release does not bind own production dispatch routing/source')
    # No resume assets or inherited sampler/Adam are accepted by these inputs.
    return parameters, dict({name:value[1] for name,value in values.items()},package_context_free_review=dispatch_sha)


def direct_authority(package, trusted_root=ROOT):
    """Human execution authority is separate from observed/independent acceptance."""
    require(package.get('execution_mode')==DIRECT_MODE and package.get('scope')==CHAIN_SCOPE and package.get('packet_revision')==13,
            'Explicit human direct execution branch required')
    require(package['code_commit']==SOURCE and package['stage']=='fresh_mu_train200'
            and package['training_protocol']==PROFILE,'Direct frozen source/stage/profile differs')
    from current_chain_runtime import validate_chain_spec
    validate_chain_spec(package)
    authority,digest,_=descriptor(package['human_training_authority'],trusted_root)
    require(authority['scope']==CHAIN_SCOPE and authority['packet_revision']==13
            and authority['current_chain_resume']==package['current_chain_resume'],
            'Exact human current-chain continuation authority required')
    require(authority['status']==DIRECT_MODE and authority['code_commit']==SOURCE
            and authority['same_instance_continuation'] is True
            and authority['completed_outer_goal']==200 and authority['final_evaluation_rows']==51713,
            'Human direct200 authority identity/range differs')
    for record in (authority,package):
        require(all(record.get(k) is False for k in ('production_thresholds_accepted','independent_acceptance',
                'training_accepted','original_sampler_actor_gate_accepted')),
                'Human authorization must not claim independent/numerical training acceptance')
    require(package['bootstrap_thresholds_accepted'] is False and package['actual_zero_ready'] is False
            and package['actual_calibration_accepted'] is False,'Unobserved calibration/bootstrap cannot be relabelled accepted')
    require(authority['configured_prospective_thresholds']==package['configured_prospective_thresholds']==DIRECT_THRESHOLDS,
            'Configured prospective thresholds differ from explicit direct branch')
    require(type(authority['human_instruction']) is str and authority['human_instruction'].strip(),
            'Original explicit human direct training instruction missing')
    return authority,digest


def human_direct_inputs(package, *, manifest_sha256, release_descriptor, trusted_root=ROOT):
    authority,authority_sha=direct_authority(package,trusted_root)
    release,release_sha,_=descriptor(release_descriptor,trusted_root)
    require(release['status']==DIRECT_MODE and release['execution_mode']==DIRECT_MODE
            and release['scope']==CHAIN_SCOPE and release['packet_revision']==13
            and release['current_chain_resume']==package['current_chain_resume']
            and release['manifest_sha256']==manifest_sha256 and release['code_commit']==SOURCE
            and release['human_training_authority_sha256']==authority_sha
            and release['cluster_name']=='wuhan-h100' and release['node_list']=='h100-10',
            'Direct release must bind exact human-authorized package and own allocation')
    require(all(release.get(k) is False for k in ('production_thresholds_accepted','independent_acceptance',
                'training_accepted','original_sampler_actor_gate_accepted')),
            'Direct release cannot invent acceptance')
    require(release['dispatch_renderer_sha256']==package['files']['render_dispatch.py']
            and release['dispatch_submitter_sha256']==package['files']['submit_once.py']
            and release['dispatch_intent_path']==package['dispatch']['once_intent'],
            'Direct release dispatcher/intent binding differs')
    return dict(DIRECT_THRESHOLDS),dict(human_training_authority=authority_sha,human_direct_release=release_sha)


def configuration_overrides(spec, thresholds):
    require(set(thresholds)==set(THRESHOLD_FIELDS), 'Explicit three accepted production thresholds required')
    items=list(spec['configuration_overrides']); prefix='actor_rollout_ref.actor.policy_safety.'
    for key in THRESHOLD_FIELDS:
        path=prefix+('sampling_probability_correction.' if key=='initial_actor_repeat_max_abs_delta' else '')+key
        if key=='target_old_policy_kl':
            require(thresholds[key]==.2 and [item for item in items if item.startswith(path+'=')]==[path+'=0.2'], 'Latest explicit user KL override missing/drifting')
            continue
        require(not any(item.startswith(path+'=') for item in items), 'Trial/default threshold survived production spec')
        items.append(path+'='+str(thresholds[key]))
    return items


def direct_rank_records(run, package, config_hash, last):
    from attempt_accounting import attempt_results
    observed=attempt_results(run,package,config_hash,last)
    expected=dict(job_id=Path(run).name.rsplit('-',1)[1],run_id=Path(run).name,code_commit=package['code_commit'],config_sha256=config_hash,
                  data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'])
    bindings=observed['record_sha256'].copy()
    for credited in observed['credited']:
        outer,attempt=credited['outer'],credited['attempt']
        for rank in range(8):
            writes=credited['actual_writes_per_rank'][str(rank)]
            path=Path(run)/'output/fresh-state'/f'outer-{outer:06d}-attempt-{attempt:06d}-rank{rank:02d}.json'
            record,digest,_=read_bound(path);require_identity(record,expected)
            require(record['status']=='DIRECT_RANK_COMPLETE_OUTER_OBSERVED' and record['training_protocol']==PROFILE
                    and record['rank']==rank and record['world_size']==8 and record['observed_outer']==outer and record['attempt_id']==attempt
                    and record['actual_optimizer_calls']==record['returned_optimizer_calls']==writes
                    and record['opportunities_processed']==20 and record['opportunities_skipped']==20-writes
                    and record['scheduler_step_before']==outer-1 and record['scheduler_step_after']==outer and record['scheduler_step_invoked'] is True
                    and record['all_rank_before_admitted'] is True and record['all_rank_scheduler_after_valid'] is True,
                    'Direct actual opportunity/write/scheduler evidence differs')
            from attempt_accounting import ledger_writes
            require(ledger_writes(record['optimizer_opportunity_ledger'])==writes and record['local_before_error'] is None and record['local_scheduler_error'] is None,'Direct ledger/error mismatch')
            require(record['learning_rates_before']==[1e-6*min((outer-1)/57,1)] and record['learning_rates_after']==[1e-6*min(outer/57,1)],'Warmup advances on completed credit only')
            bindings[str(path)]=digest
    for event in observed['credited']+observed['all_skipped']:
        outer=event.get('outer',event.get('candidate_outer'));attempt=event['attempt']
        for rank in range(8):
            writes=event.get('actual_writes_per_rank',{}).get(str(rank),0)
            path=Path(run)/'output/tree-rank-audit'/f'step-{outer:06d}-attempt-{attempt:06d}-rank-{rank:02d}.json'
            audit,digest,_=read_bound(path)
            require(all(audit[k]==expected[k] for k in ('job_id','run_id','code_commit','config_sha256','data_identity')), 'Tree rank source/run differs')
            require(audit['outer_step']==outer and audit['attempt_id']==attempt and audit['rank']==rank and audit['world_size']==8
                    and audit['local_rows']==320 and audit['local_minibatch']==16 and audit['local_microbatch']==2 and audit['epochs']==1
                    and audit['successful_optimizer_calls']==writes and audit['skipped_empty_minibatches']==20-writes
                    and audit['finite_grad_norms']==writes and len(audit['existing_grad_norms'])==writes
                    and all(math.isfinite(x) for x in audit['existing_grad_norms']) and audit['local_shard_sample_layout_unchanged'] is True,'Actual Tree returned-write numerics incomplete')
            bindings[str(path)]=digest
    return dict(completed_outer=last,training_attempts=observed['training_attempts'],direct_rank_records=last*8,
        tree_rank_records=observed['training_attempts']*8,actual_optimizer_returns=sum(observed['successful_writes_per_rank'].values()),
        opportunities_processed=observed['opportunities_per_rank']*8,actual_writes_per_rank=observed['successful_writes_per_rank'],
        skipped_opportunities_per_rank=observed['skipped_opportunities_per_rank'],attempt_accounting=observed,record_sha256=bindings)


def tree_and_safety_records(run,last,package,config_hash):
    from attempt_accounting import attempt_results
    observed=attempt_results(run,package,config_hash,last);bindings={};brakes=0
    events=sorted(observed['credited']+observed['all_skipped'],key=lambda x:x['attempt'])
    for event in events:
        outer=event.get('outer',event.get('candidate_outer'));attempt=event['attempt']
        candidates=[]
        for path in (Path(run)/'output/tree-audit').glob(f'step_{outer}_rollout_*.json'):
            if path.name.endswith('_policy_batch.json'):continue
            value,sha,_=read_bound(path)
            if value.get('attempt_id')==attempt:candidates.append((path,value,sha))
        require(len(candidates)==1,'Exactly one actual global Tree ledger per attempt required')
        path,rollout,digest=candidates[0];bindings[str(path)]=digest
        require(rollout['status']=='ROLLOUT_DIAGNOSTIC_COMPLETE' and rollout['global_step']==outer and rollout['leaf_count']==2560
                and rollout['root_leaf_quotas']==[3,2] and rollout['leaves_per_question']==5 and len(rollout['question_groups'])==512
                and all(x['roots']==2 and x['leaves']==5 for x in rollout['question_groups']) and all(x['inherited'] is True for x in rollout['branches'])
                and all(x['relative_leaf_mass']==.2 and x['context_matches'] is True for x in rollout['leaves']),'Actual Tree prefix/leaf/quota differs')
        policy_path=path.with_name(path.stem+'_policy_batch.json');policy,psha,_=read_bound(policy_path);bindings[str(policy_path)]=psha
        require(policy['status']=='POLICY_BATCH_DIAGNOSTIC_COMPLETE' and policy['rollout_file_sha256']==digest and policy['leaf_count']==2560 and len(policy['rows'])==2560,'Tree policy-weight evidence incomplete')
        for rank in range(8):
            writes=event.get('actual_writes_per_rank',{}).get(str(rank),0);records=[]
            for path in sorted((Path(run)/'output/tree-rank-audit/policy-safety').glob(f'step-{outer:06d}-attempt-{attempt:06d}-rank-{rank:02d}-event-*.json')):
                value,sha,_=read_bound(path);bindings[str(path)]=sha;records.append(value)
                require(value['outer_step']==outer and value['attempt_id']==attempt and value['rank']==rank and value['local_outer_rows']==320
                        and value['monitor_rows']==32 and value['probability_denominator']=='actual_behavior_mu'
                        and value['original_sampler_actor_consistency_accepted'] is False and value['automatic_rollback'] is False
                        and value['config']['target_old_policy_kl']==.2,'Actual safety profile/count mismatch')
                brakes+=int(value['decision']['allowed'] is False)
            require(records,'Missing real before-write safety observation')
            commits=[x['committed_optimizer_calls'] for x in records if x['phase']=='optimizer_committed']
            require(commits==list(range(1,writes+1)) and all(x['committed_optimizer_calls']<=writes for x in records),'Safety events have unaccounted writes after brake')
    return dict(completed_outers=last,training_attempts=observed['training_attempts'],observed_refusal_events=brakes,
        all_allowed_not_claimed=True,original_sampler_actor_gate_accepted=False,evidence_sha256=bindings)


def validation_reports(run, expected_outer, *, final=False, package=None,config_hash=None):
    candidates=[]
    for path in sorted((Path(run)/'output/validation-cost').glob('validation-*.json')):
        value,digest,_=read_bound(path)
        if value.get('actual_completed_outer')==expected_outer and (value.get('phase')=='final')==final:
            candidates.append((path,value,digest))
    require(len(candidates)==1, 'Exactly one actual validation report required at this phase/outer')
    path,value,digest=candidates[0]; count=51713 if final else 512
    require(value['job_id']==Path(run).name.rsplit('-',1)[1] and Path(value['run']).resolve()==(Path(run)/'output').resolve()
            and value['training_protocol']==PROFILE and value['global_step']==(201 if final else expected_outer), 'Actual evaluation counter/run/protocol differs')
    require(package is not None and config_hash is not None and value['code_commit']==package['code_commit']
            and value['config_sha256']==config_hash and value['data_identity']==package['common_data_identity']
            and value['model_identity']==package['model_identity']['sha256'], 'Actual evaluation source/model/split identity differs')
    require(value['status']=='VALIDATION_COST_COMPLETE' and value['evaluated_rows']==value['dataset_rows']==count, 'Actual evaluation coverage incomplete')
    ledger=path.with_suffix('.jsonl')
    require(hashlib.sha256(ledger.read_bytes()).hexdigest()==value['ledger_sha256'], 'Actual validation call ledger changed')
    metrics=value['metrics']; counts={k.rsplit('/',1)[1]:v for k,v in metrics.items() if k.startswith('val/evaluated_count/')}
    require(sum(counts.values())==count and (not final or len(counts)==7), 'Actual source dataset evaluation counts incomplete')
    require(metrics['val/qa_count/all']==count and math.isfinite(metrics['val/f1/all']) and math.isfinite(metrics['val/em/all']), 'Actual F1/EM/QA coverage missing')
    return dict(path=str(path),sha256=digest,evaluated_rows=count,per_dataset=counts,metrics=metrics,phase=value['phase'])


def checkpoint_manifest(run, package, config_hash, outer):
    directory=Path(run)/'checkpoints/training_state'/f'global_step_{outer}'
    manifest,digest,_=read_bound(directory/'manifest.json')
    expected=dict(job_id=Path(run).name.rsplit('-',1)[1],code_commit=package['code_commit'],config_sha256=config_hash,
        data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'],completed_step=outer,actual_update_limit=200)
    from attempt_accounting import attempt_results,checkpoint_accounting
    actual=checkpoint_accounting(manifest['identity']['guard_continue_accounting'],attempt_results(run,package,config_hash,outer),outer,run)
    expected['guard_continue_accounting']=actual
    require(manifest['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and manifest['identity']==expected,'Saved full-state actual identity differs')
    workers=sorted(manifest['worker_states'],key=lambda x:x['identity']['rank'])
    require(len(workers)==8 and [x['identity']['rank'] for x in workers]==list(range(8)), 'Saved state rank coverage incomplete')
    for rank,record in enumerate(workers):
        worker_identity=record['identity']; visible=worker_identity.get('cuda_visible_devices')
        require(type(visible) is str and visible!=''
                and worker_identity==dict(expected,rank=rank,world_size=8,cuda_visible_devices=visible)
                and record['full_worker_state_finite'] is True and record['scheduler_last_epoch']==outer
                and (outer!=5 or record['exact_same_world_reload_verified'] is True), 'Actual worker state finite/reload/counter proof differs')
        require(Path(record['path']).resolve()==(directory/f'rank-{rank:02d}.pt').resolve(), 'Actual saved shard must match own rank/path')
    require(manifest['driver']['next_step']==outer+1 and (outer!=5 or manifest['driver']['exact_reload_verified'] is True), 'Driver state boundary/reload proof missing')
    require(Path(manifest['driver']['path']).resolve()==(directory/'driver.pt').resolve(), 'Actual driver state path outside own checkpoint')
    require(Path(manifest['hf_checkpoint']).resolve()==(Path(run)/'checkpoints/actor'/f'global_step_{outer}').resolve(), 'HF path must match own completed outer')
    return manifest,digest


def hf_reload_report(path, base, checkpoint, hf_identity):
    record,digest,_=read_bound(path)
    require(record['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED'
            and Path(record['base']).resolve()==Path(base).resolve() and Path(record['checkpoint']).resolve()==Path(checkpoint).resolve()
            and record['tokenizer_vocab_equal'] is True and record['bos_eos_equal'] is True
            and record['all_parameters_finite'] is True and record['cpu_forward_logits_finite'] is True
            and record['changed_parameters']>0 and record['max_abs_parameter_delta']>0
            and not any(record['hf_loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
            and record['checkpoint_file_sha256']==hf_identity['files'], 'Actual full HF reload/tokenizer/delta checker has not accepted exact checkpoint')
    return digest


def state_cpu_report(path, manifest_sha256, outer):
    value,digest,_=read_bound(path)
    from attempt_accounting import sampler_position
    epoch,cursor=sampler_position(value['training_attempts'])
    require(value['status']=='FRESH200_COMPLETE_SAVED_STATE_CPU_VERIFIED' and value['manifest_sha256']==manifest_sha256
            and value['completed_outer']==value['driver_actual_outer']==outer and value['sampler_cursor']==cursor and value['sampler_epoch']==epoch
            and value['actual_pt_deserialization'] is True and value['all_saved_state_finite'] is True
            and value['cuda_initialized'] is False and value['resume_permitted'] is False, 'Actual complete CPU state acceptance missing')
    workers=sorted(value['workers'],key=lambda x:x['rank'])
    require(len(workers)==8 and [x['rank'] for x in workers]==list(range(8))
            and all(x['world_size']==8 and x['adam_step_values']==[value['actual_optimizer_calls_per_rank'][str(x['rank'])]]
                    and x['scheduler_step']==outer and x['all_saved_state_finite'] is True for x in workers), 'Actual saved CPU rank Adam/scheduler coverage differs')
    return digest


def publish_external_approval(run, package, config_hash, proofs):
    gate=Path(run)/'output/acceptance-step5'; envelope=gate/'independent-envelope.json'
    if not envelope.exists(): return False
    value,envelope_sha,_=read_bound(envelope)
    review,review_sha,raw=descriptor(value['review'],Path(run))
    ready,ready_sha,_=read_bound(gate/'ready.json')
    require(review['status']=='PASS' and review['fork_turns']=='none' and review['ready_sha256']==ready_sha, 'No genuine context-free actual step5 review')
    require_identity(review,dict(job_id=Path(run).name.rsplit('-',1)[1],run_id=Path(run).name,code_commit=package['code_commit'],config_sha256=config_hash,data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256']))
    require(ready['completed_outer_updates']==5 and ready['next_outer_update']==6 and ready['actual_update_limit']==200, 'Live actual five gate boundary differs')
    require(all(review[k] is True for k in ('full_state_reload_accepted','full_hf_reload_accepted','rank_tree_numerics_accepted','fixed512_evaluation_accepted','behavior_mu_binding_accepted')), 'Independent five acceptance incomplete')
    require(review['full_state_manifest_sha256']==ready['full_state_manifest_sha256']==proofs['full_state_manifest_sha256']
            and review['hf_identity_sha256']==ready['hf_identity']['sha256']
            and review['hf_reload_report_sha256']==proofs['hf_reload_report_sha256']
            and review['state_cpu_report_sha256']==proofs['state_cpu_report_sha256']
            and review['behavior_mu_cpu_report_sha256']==proofs['behavior_mu_cpu_report_sha256']
            and review['development_report_sha256']==proofs['development']['sha256']
            and review['direct_rank_record_sha256']==proofs['rank_counts']['record_sha256'], 'Independent review actual evidence bindings differ')
    assessment,assessment_sha,assessment_raw=descriptor(value['threshold_assessment'],Path(run))
    require(review['threshold_assessment_sha256']==assessment_sha and review['short_training_thresholds_accepted'] is True,
            'Actual five review lacks independent unchanged-threshold acceptance')
    validate_short_threshold_assessment(run,package,config_hash,proofs,assessment,ready_sha)
    require(review['partial_raw_cpu_report_sha256']==proofs['partial_raw_cpu_report_sha256'],
            'Actual independent review omitted retained partial raw IO verification')
    approved=gate/'approved.json'
    # This publishes the reviewer's original bytes; it never synthesizes PASS.
    if approved.exists(): require(read_bound(approved)[1]==review_sha,'Existing approved receipt differs from external independent review')
    else:
        pending=gate/'approved.wrapper-pending.json'
        with pending.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(pending,approved);pending.unlink()
        from slurm_identity import fsync_directory
        fsync_directory(gate)
    return dict(review_sha256=review_sha,envelope_sha256=envelope_sha,ready_sha256=ready_sha,threshold_assessment_sha256=assessment_sha,
                full_run_thresholds_accepted=True,unchanged_thresholds=package['bootstrap_thresholds'])


def short_threshold_observations(run,proofs,thresholds):
    """Read actual first-five safety JSON; no simulated measurements or PT imports."""
    bindings=proofs['tree_and_safety']['evidence_sha256'];records={};kl=[];tail=[];repeat=[];coverage=set();missing=[]
    for name,expected_sha in sorted(bindings.items()):
        path=Path(name)
        if 'policy-safety' not in path.parts:continue
        require(path.is_absolute() and path.resolve().is_relative_to((Path(run)/'output/tree-rank-audit/policy-safety').resolve()),'Short monitor outside own run')
        value,digest,_=read_bound(path,expected_sha);decision=value['decision'];config=value['config']
        require(config['target_old_policy_kl']==thresholds['target_old_policy_kl'] and config['negative_tail_limit']==thresholds['negative_tail_limit']
                and config['sampling_probability_correction']['initial_actor_repeat_max_abs_delta']==thresholds['initial_actor_repeat_max_abs_delta'],
                'Short safety actually consumed different bootstrap values')
        require(value['outer_step'] in range(1,6) and value['rank'] in range(8) and value['monitor_rows']==32 and value['probability_denominator']=='actual_behavior_mu'
                and value['original_sampler_actor_consistency_accepted'] is False,'First-five actual monitor coordinates/count differ')
        coverage.add((value['outer_step'],value['attempt_id'],value['rank']));records[name]=digest
        for key,output in (('sampled_old_policy_kl',kl),('negative_tail_fraction',tail)):
            number=decision.get(key)
            if number is None:missing.append(dict(path=name,metric=key,reasons=decision.get('reasons',[])))
            else:
                require(type(number) in (int,float) and math.isfinite(number),'Invalid actual short safety metric')
                output.append(number)
        if decision.get('initial_actor_repeat_checked') is True:
            number=decision['global_monitor_actor_repeat_max_abs_delta']
            require(type(number) in (int,float) and math.isfinite(number) and number>=0,'Actual initial actor repeat metric invalid')
            repeat.append(number)
    attempts=proofs['rank_counts']['attempt_accounting']
    expected={(event.get('outer',event.get('candidate_outer')),event['attempt'],rank)
              for event in attempts['credited']+attempts['all_skipped'] for rank in range(8)}
    require(records and coverage==expected and kl and tail and len(repeat)>=len(expected),'Actual first-five all-attempt/rank metrics incomplete')
    return dict(completed_outer=5,training_attempts=proofs['rank_counts']['training_attempts'],monitor_rows_per_rank=32,
                evidence_sha256=records,max_sampled_old_policy_kl=max(kl),max_negative_tail_fraction=max(tail),
                max_initial_actor_repeat_abs_delta=max(repeat),missing_metric_records=missing,
                guard_refusal_events=proofs['tree_and_safety']['observed_refusal_events'],
                actual_writes_per_rank=proofs['rank_counts']['actual_writes_per_rank'],
                opportunities_processed=proofs['rank_counts']['opportunities_processed'],thresholds=thresholds)

def validate_short_threshold_assessment(run,package,config_hash,proofs,assessment,ready_sha):
    expected=dict(job_id=Path(run).name.rsplit('-',1)[1],run_id=Path(run).name,code_commit=package['code_commit'],config_sha256=config_hash,
                  data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'])
    require_identity(assessment,expected)
    require(assessment['status']=='INDEPENDENT_SAME_INSTANCE_FIRST5_THRESHOLDS_ACCEPTED' and assessment['fork_turns']=='none'
            and assessment['training_protocol']==PROFILE and assessment['completed_outer']==5 and assessment['continue_range']==[6,200]
            and assessment['unchanged_thresholds_accepted'] is True and assessment['original_sampler_actor_gate_accepted'] is False,
            'No actual same-instance short-training threshold acceptance')
    require(assessment['ready_sha256']==ready_sha and assessment['bootstrap_thresholds_sha256']==package['production_inputs']['threshold_acceptance']['sha256']
            and assessment['calibration_ready_sha256']==package['production_inputs']['actual_calibration_ready']['sha256']
            and assessment['calibration_acceptance_sha256']==package['production_inputs']['actual_calibration_acceptance']['sha256']
            and assessment['values']==package['bootstrap_thresholds'],'Threshold assessment zero/bootstrap/value binding differs')
    require(assessment['observations']==short_threshold_observations(run,proofs,package['bootstrap_thresholds']),
            'Threshold assessment does not bind actual first-five monitor/guard/count measurements')
    require(assessment['partial_raw_cpu_report_sha256']==proofs['partial_raw_cpu_report_sha256']
            and type(assessment['assessment_rationale']) is str and assessment['assessment_rationale'].strip(),
            'Threshold assessment lacks actual retained raw proof or judgment')
    return True


def submitted_intent(path,manifest_sha,review_sha,job,*,cluster_name,node_list,run_id,wait_seconds=30,clock=None,sleep=None):
    """Read-only bounded wait; cluster-qualified unknowns never execute/retry."""
    import time
    from slurm_identity import submitted_lineage,CLUSTER,NODE
    require(cluster_name==CLUSTER and node_list==NODE and run_id=='tree-current200-continue-'+str(job),'Runtime own tuple missing')
    clock=time.monotonic if clock is None else clock;sleep=time.sleep if sleep is None else sleep
    require(type(wait_seconds) in (int,float) and 0<=wait_seconds<=30,'Intent read-only wait must stay bounded')
    deadline=clock()+wait_seconds
    while True:
        value,_,_=read_bound(path)
        require(value['manifest_sha256']==manifest_sha and value['review_sha256']==review_sha
                and value.get('cluster_name')==CLUSTER and value.get('dispatch_intent_path')==str(path),'Once journal own lineage differs')
        if value['status']=='SUBMITTED':return submitted_lineage(value,str(job),manifest_sha,review_sha,path)
        require(value['status']=='SUBMITTING' and clock()<deadline,'Unknown, failed or unresolved submission cannot execute or retry')
        sleep(.2)
