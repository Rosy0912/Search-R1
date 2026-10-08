"""Actual two-outer warmup measurements; never formal updates or inherited state."""
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from reproduction.common8.calibration_runtime import training_input_rows,write_once,verify_live_optimizer_workers
from reproduction.common8.formal_contract import assert_scientific_run,consumed_lr,verify_scientific_config
from reproduction.common8.policy_calibration_plan import canonical_bytes,document_sha256
from reproduction.common8.verify_data import digest,verify
from reproduction.common8.warmup_measurement_plan import ROLE,OUTER_COUNT,WRITES,MONITORS,load_plan,need

SEGMENT_PATH='logs/warmup57-measurement-segment.json'


def scheduler_projection(rows):
    """Preserve durable completed rows; final cleanup changes console/ledger hashes."""
    return {rank:{name:row[name] for name in ('receipt_path','selected_rows','selected_rows_sha256','observed_file_sha256')}
            for rank,row in rows.items()}


def active_plan():
    if os.environ.get('SEARCH_G1_PROFILE')!='common8':return None
    base=Path(os.environ['SEARCH_G1_BASE']).resolve()
    run=json.loads((base/'logs/run-manifest.json').read_text())
    if run.get('policy_calibration_role')!=ROLE:return None
    plan,sha=load_plan(base,run,expected_slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    return base,run,plan,sha


def training_inputs(batch,phase,parents=None,*,logical_update):
    base,run,plan,sha=phase
    need(type(logical_update) is int and logical_update in (1,2),'Wrong actual measurement outer')
    rows=training_input_rows(batch)
    if parents is None:
        need(len(rows)==512,'Actual512 parents required')
        return rows
    need(len(parents)==512 and len(rows)==2560 and rows==[row for row in parents for _ in range(5)],'ActualG5 expansion differs')
    result=dict(run_id=run['run_id'],plan_sha256=sha,slurm_job_id=plan['slurm_job_id'],logical_update=logical_update,
        execution_role=ROLE,actual_parent_count=512,actual_trajectory_count=2560,n_agent=5,
        train_sha256=run['input_sha256']['train.parquet'],parents=parents,
        repeated_group_sha256=[row['upstream_group_sha256'] for row in rows],production_admission=False)
    write_once(base/'logs'/f'warmup57-measurement-inputs-{logical_update}.json',result)
    return result


def training_groups_before_update(batch,phase,*,logical_update):
    base,run,plan,sha=phase
    saved=json.loads((base/'logs'/f'warmup57-measurement-inputs-{logical_update}.json').read_text())
    actual=[]
    for value in batch.non_tensor_batch['uid']:
        if hasattr(value,'item'):value=value.item()
        actual.append(hashlib.sha256(canonical_bytes(value)).hexdigest())
    need(len(batch)==2560 and saved['run_id']==run['run_id'] and saved['plan_sha256']==sha and
         Counter(actual)==Counter(saved['repeated_group_sha256']),'Actual balanced groups differ')
    result=dict(run_id=run['run_id'],plan_sha256=sha,slurm_job_id=plan['slurm_job_id'],execution_role=ROLE,
        logical_update=logical_update,actual_trajectory_count=2560,balanced_group_sha256=actual,
        upstream_group_multiset_unchanged=True,production_admission=False)
    write_once(base/'logs'/f'warmup57-measurement-groups-{logical_update}.json',result)
    return result


def evaluation(base,run,plan,sha,step,rows,original_rewards):
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.validation_identity import expected
    need(type(step) is int and step in (0,2),'Only paired before0/after2 evaluations allowed')
    record,frame,ids=expected(base,'validation');result=aggregate(rows,ids,frame['data_source'].tolist())
    need([r['em'] for r in rows]==list(original_rewards),'Independent measurements differ from original EM')
    result.update(schema_version=2,phase_role=ROLE,run_id=run['run_id'],parent_production_run_id=plan['parent_production_run_id'],
        slurm_job_id=plan['slurm_job_id'],plan_sha256=sha,step=step,validation_sha256=record['sha256'],checked_unix=time.time())
    suffix='before' if step==0 else 'after'
    write_once(base/'logs'/f'warmup57-measurement-evaluation-{suffix}.json',result)
    return result


def verify_optimizer_states(states,run,plan,sha,allocation,initial_states,initial_sha256):
    need(set(states)==set(range(8)) and all(type(rank) is int for rank in states),'Eight real state proofs required')
    need(set(initial_states)==set(initial_sha256)==set(range(8)),'Eight independent actual initial-empty-state proofs required')
    need(allocation['accepted'] is True and allocation['run_id']==run['run_id'] and
         allocation['slurm_job_id']==plan['slurm_job_id'],'Wrong captured native allocation')
    uuids=allocation['actual_cuda_runtime_uuids'];need(len(uuids)==len(set(uuids))==8,'Wrong real CUDA UUIDs')
    devices=[];births=[]
    for rank in range(8):
        row=states[rank]
        initial=initial_states[rank]
        for state,logical,rates in ((initial,0,[consumed_lr(1)]),(row,2,[consumed_lr(2)])):
            need(state['schema_version']==1 and state['logical_update']==logical and
                 state['run_id']==run['run_id'] and state['execution_role']==ROLE and state['plan_sha256']==sha and
                 state['parent_production_run_id']==run['measurement_parent_run_id'] and state['slurm_job_id']==plan['slurm_job_id'] and
                 type(state['rank']) is int and state['rank']==rank and
                 type(state['initial_optimizer_writes']) is int and state['initial_optimizer_writes']==0 and
                 type(state['initial_adam_state_entries']) is int and state['initial_adam_state_entries']==0,
                 'Wrong original/final actual state binding')
            need(isinstance(state['learning_rates'],list) and bool(state['learning_rates']) and
                 all(type(value) in (int,float) and value==rates[0] for value in state['learning_rates']),
                 'Wrong actual initial/final consumed LR')
            need(state['all_parameters_and_adam_finite'] is True,'Actual original/final state not finite')
        for name in ('pid','start_ticks','cuda_uuid','parameter_tensor_count','parameter_numel'):
            need(row[name]==initial[name],'Original/final worker identity or parameter layout changed')
        need(initial['final_optimizer_writes']==0 and type(initial['final_optimizer_writes']) is int and
             type(initial['adam_state_entries']) is int and initial['adam_state_entries']==0 and
             initial['adam_steps']==[] and initial['adam_state_tensor_count']==0 and initial['adam_state_numel']==0 and
             initial['initial_optimizer_proof_sha256'] is None and
             row['initial_optimizer_proof_sha256']==initial_sha256[rank],'Original Adam was not empty/bound before firstwrite')
        for name,value in (('rank',rank),('initial_optimizer_writes',0),('initial_adam_state_entries',0),('final_optimizer_writes',40)):
            need(type(row[name]) is int and row[name]==value,'Wrong terminal Adam counter '+name)
        for name in ('pid','start_ticks','parameter_tensor_count','parameter_numel','adam_state_entries','adam_state_tensor_count','adam_state_numel'):
            need(type(row[name]) is int and row[name]>0,'Missing actual state count '+name)
        need(row['all_parameters_and_adam_finite'] is True and row['adam_state_tensor_count']>=3*row['adam_state_entries'],
             'Actual parameters/Adam not finite or complete')
        steps=row['adam_steps'];need(isinstance(steps,list) and len(steps)==row['adam_state_entries'] and
             all(type(value) in (int,float) and value==40 for value in steps),'Actual Adam steps must all be40')
        need(row['cuda_uuid'] in uuids,'Wrong captured actual worker CUDA device')
        devices.append(row['cuda_uuid']);births.append(row['pid'])
    need(set(devices)==set(uuids) and len(set(births))==8,'Duplicated actual device/worker')
    return [states[rank] for rank in range(8)]


def verify_measurement(base,run=None,*,candidate=None,require_live_workers=False):
    """Historical CPU audit. Native terminal state is authenticated by the caller.

    No RUNNING/CUDA query occurs here. Optional live PID checks are only for the
    actual on-node finisher before cleanup, never historical authentication.
    """
    from reproduction.common8.behavior_runtime import read_gate_receipt,verify_write_receipt_links
    from reproduction.common8.worker_evidence import read_console_form
    from reproduction.common8.runtime import budget_rows,verify_measurement_scheduler_receipts
    from reproduction.common8.warmup_measurement_evidence import verify_segment
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.validation_identity import expected
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    base=Path(base).resolve()
    current=json.loads((base/'logs/run-manifest.json').read_text())
    need(run is None or run==current,'Manifest changed during historical measurement audit');run=current
    assert_scientific_run(run,expected_role=ROLE);plan,sha=load_plan(base,run)
    result=candidate if candidate is not None else json.loads((base/SEGMENT_PATH).read_text())
    need(result['schema_version']==2 and result['role']=='warmup57_measurement_evidence' and
         result['run_id']==run['run_id'] and result['plan_sha256']==sha and result['identity']==plan['identity'] and
         result['parent_production_run_id']==plan['parent_production_run_id'] and result['slurm_job_id']==plan['slurm_job_id'] and
         result['resolved_config_sha256']==plan['resolved_config_sha256'] and
         result['initial_reward_fingerprint']==plan['initial_reward_fingerprint'] and
         result['completed_outer_updates']==2 and result['optimizer_writes_per_rank']==40 and
         result['updates_counted_as_production']==0 and result['production_admission'] is False and
         result['healthy_checkpoint_accepted'] is False,
         'Wrong typed actual measurement segment')
    graph={}
    def read(relative):
        from reproduction.common8.warmup_measurement_plan import _regular
        path=_regular(base/relative,base)
        graph[relative]=digest(path);return json.loads(path.read_text())
    allocation=read('logs/native-allocation-admission.json')
    claim=read('logs/training-launch-claim.json')
    need(claim['run_id']==run['run_id'] and claim['slurm_job_id']==plan['slurm_job_id'] and
         type(claim['claimed_unix']) in (int,float) and math.isfinite(claim['claimed_unix']) and
         claim['claimed_unix']>=plan['created_unix'],'Real plan must predate actual trainer launch/Adam')
    launch=read('logs/resolved-training-config.json')
    need(launch['run_id']==run['run_id'] and launch['slurm_job_id']==plan['slurm_job_id'] and
         launch['resolved_config_sha256']==document_sha256(launch['resolved_config'])==plan['resolved_config_sha256'],
         'Original actual launch configuration changed')
    verify_scientific_config(launch['resolved_config'],run,base)
    updates=[json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    graph['logs/actual-updates.jsonl']=digest(base/'logs/actual-updates.jsonl')
    need([row['step'] for row in updates]==[1,2] and updates==result['actual_updates'],'Exactly two actual completed updates required')
    for row in updates:
        need(row['run_id']==run['run_id'] and row['execution_role']==ROLE and row['probe_version']==run['initial_probe_version'] and
             all(math.isfinite(value) for value in row['metrics'].values()) and row['metrics']['actor/optimizer_steps_this_update']==20,
             'Actual update identity/numerical metrics differ')
    gates={str(step):read_gate_receipt(base,run,step,update_record=updates[step-1]) for step in (1,2)}
    verify_write_receipt_links(base,gates,1,2)
    need(gates==result['behavior_probability_gates'],'Actual all2560 behavior evidence differs')
    from reproduction.common8.behavior_trace import audit_outer
    for step in (1,2):
        receipt=read(f'logs/behavior-probability-outer-{step}.json')
        trace=read(f'artifacts/behavior-traces/outer-{step}.json')
        need(graph[f'artifacts/behavior-traces/outer-{step}.json']==gates[str(step)]['private_trace_sha256'],
             'Original private full2560 trace bytes changed')
        parents=trace['parent_group_ids']
        ids=[run['run_id']+'-'+str(step)+'-'+parent+'-sample-'+str(sample) for parent in parents for sample in range(5)]
        audit=audit_outer(trace,expected_run_id=run['run_id'],expected_source_sha256=run['source_sha256'],
            expected_slurm_job_id=plan['slurm_job_id'],expected_logical_update=step,
            expected_parent_group_ids=parents,expected_trajectory_ids=ids,
            expected_policy_snapshot_id=receipt['policy_snapshot_id'],expected_sampling=receipt['sampling'])
        need(audit['trace_sha256']==receipt['trace_sha256']==receipt['private_trace_sha256'] and
             audit['included_token_count']==receipt['included_actor_token_count'] and
             audit['actor_rows_sha256']==receipt['actor_rows_sha256'] and document_sha256(audit)==receipt['audit_sha256'] and
             audit['trajectory_ids']==receipt['trajectory_ids'] and audit['actor_token_counts']==receipt['actor_token_counts'],
             'Original full2560 token/context/mask audit differs')
    budgets=budget_rows(read_console_form(base,run['run_id']),1,2)
    schedulers=scheduler_projection(verify_measurement_scheduler_receipts(base,run,1,2,gates))
    need(budgets==result['budget_rows'] and schedulers==result['scheduler_receipts'],'Actual budget/scheduler evidence differs')
    logs={}
    for rank in range(8):
        name=f'logs/optimizer-rank-{rank}.jsonl';graph[name]=digest(base/name)
        logs[rank]=[json.loads(line) for line in (base/name).read_text().splitlines()]
        scheduler=f'logs/scheduler-rank-{rank}.jsonl';graph[scheduler]=digest(base/scheduler)
        actual_scheduler=[json.loads(line) for line in (base/scheduler).read_text().splitlines()]
        need([(row['logical_update'],row['kind']) for row in actual_scheduler]==[
             (step,kind) for step in (1,2) for kind in ('BEGIN','COMMIT_PREPARED','COMPLETE')],
             'Extra, missing or rejected measurement scheduler events')
    structural=verify_segment(logs,dict(schema_version=2,role=ROLE,run_id=run['run_id'],calibration_sha256=sha,
                                     outer_count=2,initial_optimizer_writes=0))
    need(structural==result['structural'] and len(structural['global_monitors'])==80,'Actual80 global monitors differ')
    for step in (1,2):
        need(structural['behavior_gate_sha256'][str(step)]==gates[str(step)]['proof_sha256'],'Monitor/write gate link differs')
        parents=read(f'logs/warmup57-measurement-inputs-{step}.json');groups=read(f'logs/warmup57-measurement-groups-{step}.json')
        for row in (parents,groups):
            need(row['run_id']==run['run_id'] and row['plan_sha256']==sha and row['logical_update']==step and
                 row['execution_role']==ROLE and row['actual_trajectory_count']==2560,'Wrong loaded training outer identity')
        need(parents['actual_parent_count']==512 and parents['n_agent']==5 and parents['train_sha256']==run['input_sha256']['train.parquet'],
             'Wrong actual original512/G5 train product')
        need(len(parents['parents'])==512 and parents['repeated_group_sha256']==[
             item['upstream_group_sha256'] for item in parents['parents'] for _ in range(5)],'Actual G5 expansion changed')
        need(groups['upstream_group_multiset_unchanged'] is True and
             Counter(groups['balanced_group_sha256'])==Counter(parents['repeated_group_sha256']),'Actual GRPO groups changed')
        raw=read(f'artifacts/behavior-traces/outer-{step}.json')
        need([value.split('-',1)[1] for value in raw['parent_group_ids']]==[
             item['upstream_group_sha256'] for item in parents['parents']],
             'Raw actual trajectories do not bind these512 parent occurrences')
    states={rank:read(f'logs/warmup57-measurement-optimizer-rank-{rank}.json') for rank in range(8)}
    initial_states={rank:read(f'logs/warmup57-measurement-initial-optimizer-rank-{rank}.json') for rank in range(8)}
    initial_shas={rank:graph[f'logs/warmup57-measurement-initial-optimizer-rank-{rank}.json'] for rank in range(8)}
    states=verify_optimizer_states(states,run,plan,sha,allocation,initial_states,initial_shas)
    need(states==result['actual_optimizer_states'],'Actual post40 state proof differs')
    need([initial_states[rank] for rank in range(8)]==result['initial_optimizer_states'],'Independent actual initial0 states differ')
    if require_live_workers:verify_live_optimizer_workers(states,base,plan['slurm_job_id'])
    record,frame,ids=expected(base,'validation');evaluations={}
    for suffix,step in (('before',0),('after',2)):
        proof=read(f'logs/warmup57-measurement-evaluation-{suffix}.json')
        measured=aggregate(proof['rows'],ids,frame['data_source'].tolist())
        need(proof['phase_role']==ROLE and proof['run_id']==run['run_id'] and proof['plan_sha256']==sha and
             proof['step']==step and proof['validation_sha256']==record['sha256'] and
             all(proof[name]==value for name,value in measured.items()),'Actual paired512 scalar/ID/count proof differs')
        need(proof==result['evaluation_'+suffix],'Frozen paired512 proof differs');evaluations[suffix]=proof
    passed=all(evaluations['after'][key]>=evaluations['before'][key] for key in plan['method']['health_rule']['allowed_drop'])
    need(result['health_measurement_passed']==passed,'Measured health cannot be relabeled')
    checkpoint=Path(run['checkpoint_dir'])/'actor/global_step_2'
    saved=read(str((checkpoint/'complete.json').relative_to(base)))
    need(saved==result['checkpoint'] and saved['run_id']==run['run_id'] and saved['step']==2 and
         saved['finite_all_saved_tensors'] is True and Path(saved['checkpoint']).resolve()==checkpoint,'Wrong actual checkpoint2')
    count=validate_saved_checkpoint(checkpoint,Path(run['initial_model']),run['started_unix'])
    need(count==result['saved_tensor_count'],'Full canonical/finite tensor count differs')
    for name,value in saved['files_sha256'].items():
        need(digest(checkpoint/name)==value,'Checkpoint2 changed');graph[str((checkpoint/name).relative_to(base))]=value
    for name,value in run['initial_model_sha256'].items():
        path=Path(run['initial_model'])/name;need(digest(path)==value,'Official initial weights changed')
        if path.resolve().is_relative_to(base):graph[str(path.relative_to(base))]=value
    import torch
    from safetensors import safe_open
    initial=Path(run['initial_model']);old=json.loads((initial/'model.safetensors.index.json').read_text())['weight_map']
    new=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map'];changed=[]
    for key in old:
        with safe_open(str(initial/old[key]),framework='pt',device='cpu') as f:a=f.get_tensor(key)
        with safe_open(str(checkpoint/new[key]),framework='pt',device='cpu') as f:z=f.get_tensor(key)
        if not torch.equal(a,z):changed.append(key)
        del a,z
    need(bool(changed) and changed==result['changed_parameters'],'Freshcheckpoint2 must have real parameter delta')
    live=read('artifacts/live/manifest.json');need(live['run_id']==run['run_id'] and live['round']==0 and
         live['version']==run['initial_probe_version']==plan['initial_probe_version'] and
         live['head_sha256']==plan['initial_reward_fingerprint']['head_sha256'] and
         live['encoder_sha256']==plan['initial_reward_fingerprint']['encoder_sha256'],'Actual initial0 probes changed')
    for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
        path=Path(live[key]);need(digest(path)==live['head_sha256'][name],'Actual initial fitted head changed')
        import joblib
        from reproduction.a800.recovery_binding import finite_fitted_values
        fitted=joblib.load(path);finite_fitted_values(vars(fitted['clf']))
        graph[str(path.relative_to(base))]=digest(path)
    heads=Path(live['commit']).parent;dump=heads.parent/'dumps'
    need(heads.parent.name.startswith('round_0-attempt-') and
         {path.name for path in dump.iterdir() if path.is_file()}==set(live['dump_sha256']),
         'Fresh initial0 probes must bind the complete new intervention set')
    for name,value in live['dump_sha256'].items():need(digest(dump/name)==value,'Initial actual interventions changed');graph[str((dump/name).relative_to(base))]=value
    report=read(str((heads/'calibration-report.json').relative_to(base)))
    archive=read(str((heads.parent/'version-manifest.json').relative_to(base)))
    need(report==live['calibration_report'] and graph[str((heads/'calibration-report.json').relative_to(base))]==live['calibration_report_sha256'] and
         archive==live,'Fresh initial0 original report/archive differs')
    labels=live['label_proof'];need(report['unique_question_count']==labels['question_count']==300 and
         report['question_grouped_cv'] is True and report['require_all_cv_folds'] is True and report['configured_cv_folds']==2 and
         labels['group_size']==5 and set(labels['question_ids'])==set(run['calibration_question_ids']) and
         len(set(labels['question_ids']))==300 and labels['searched_valid_trajectory_count']>0 and
         labels['closed_book_label']=='exact_match_no_retrieval' and
         labels['reliance_label']=='normalized_answer_change_after_evidence_deletion' and
         labels['model_authored_answer_spans'] is True and labels['boundary_all_reserved_questions'] is True and
         labels['calibration_protocol']['rollout_max_new']==500 and labels['calibration_protocol']['correction_feedback'] is True and
         labels['evidence_deletion']=='typed_retrieval_ranges_only_preserve_model_and_correction',
         'Original real fresh300/G5 two-probe calibration differs')
    for key in ('commit','boundary'):
        need(len(report[key+'_class_counts'])==2 and min(report[key+'_class_counts'])>0 and
             math.isfinite(report[key+'_cv_auc']) and 0<=report[key+'_cv_auc']<=1,'Original real groupedCV failed')
    verify(base/'data')
    for name,value in run['input_sha256'].items():
        need(digest(base/'data'/name)==value,'Shared original data changed');graph['data/'+name]=value
    for name,value in run['source_sha256'].items():
        need(digest(base/'repo'/name)==value,'Original measured source changed');graph['repo/'+name]=value
    for name in ('logs/run-manifest.json','logs/warmup57-measurement-plan.json',
                 'preparation/measurement-parent-run-manifest.json','preparation/warmup57-measurement-method.json',
                 'logs/resolved-training-config.json','logs/training-launch-claim.json','logs/process-ledger.json'):
        graph[name]=digest(base/name)
    console=Path(run['train_log']);need(console.resolve().is_relative_to(base) and not console.is_symlink(),'Foreign training console')
    graph[str(console.relative_to(base))]=digest(console)
    graph.update(plan['prerequisite_sha256'])
    if candidate is None:graph[SEGMENT_PATH]=digest(base/SEGMENT_PATH)
    return dict(result,source_file_sha256=graph)


def finish_after_update(trainer,phase):
    from reproduction.common8.behavior_runtime import read_gate_receipt
    from reproduction.common8.runtime import checkpoint_complete,budget_rows,verify_measurement_scheduler_receipts
    from reproduction.common8.worker_evidence import read_console_form
    from reproduction.common8.warmup_measurement_evidence import verify_segment
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    base,run,plan,sha=phase
    need(trainer.total_training_steps==201 and trainer.global_steps==2,'Measurement must stop after actual2 with original201 horizon')
    trainer._save_checkpoint();checkpoint=Path(run['checkpoint_dir'])/'actor/global_step_2'
    saved=checkpoint_complete(2,str(checkpoint));trainer._validate()
    updates=[json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    need([row['step'] for row in updates]==[1,2],'Two actual completed updates required')
    gates={str(step):read_gate_receipt(base,run,step,update_record=updates[step-1]) for step in (1,2)}
    budgets=budget_rows(read_console_form(base,run['run_id']),1,2)
    schedulers=scheduler_projection(verify_measurement_scheduler_receipts(base,run,1,2,gates))
    logs={rank:[json.loads(line) for line in (base/'logs'/f'optimizer-rank-{rank}.jsonl').read_text().splitlines()] for rank in range(8)}
    structural=verify_segment(logs,dict(schema_version=2,role=ROLE,run_id=run['run_id'],calibration_sha256=sha,outer_count=2,initial_optimizer_writes=0))
    allocation=json.loads((base/'logs/native-allocation-admission.json').read_text())
    initial_states={rank:json.loads((base/'logs'/f'warmup57-measurement-initial-optimizer-rank-{rank}.json').read_text()) for rank in range(8)}
    initial_shas={rank:digest(base/'logs'/f'warmup57-measurement-initial-optimizer-rank-{rank}.json') for rank in range(8)}
    states=verify_optimizer_states({rank:json.loads((base/'logs'/f'warmup57-measurement-optimizer-rank-{rank}.json').read_text()) for rank in range(8)},run,plan,sha,allocation,initial_states,initial_shas)
    before=json.loads((base/'logs/warmup57-measurement-evaluation-before.json').read_text())
    after=json.loads((base/'logs/warmup57-measurement-evaluation-after.json').read_text())
    import torch
    from safetensors import safe_open
    initial=Path(run['initial_model']);old=json.loads((initial/'model.safetensors.index.json').read_text())['weight_map']
    new=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map'];changed=[]
    for key in old:
        with safe_open(str(initial/old[key]),framework='pt',device='cpu') as f:a=f.get_tensor(key)
        with safe_open(str(checkpoint/new[key]),framework='pt',device='cpu') as f:z=f.get_tensor(key)
        if not torch.equal(a,z):changed.append(key)
        del a,z
    result=dict(schema_version=2,role='warmup57_measurement_evidence',run_id=run['run_id'],
        parent_production_run_id=plan['parent_production_run_id'],slurm_job_id=plan['slurm_job_id'],plan_sha256=sha,
        identity=plan['identity'],resolved_config_sha256=plan['resolved_config_sha256'],initial_reward_fingerprint=plan['initial_reward_fingerprint'],
        completed_outer_updates=2,optimizer_writes_per_rank=40,structural=structural,budget_rows=budgets,
        scheduler_receipts=schedulers,checkpoint=saved,saved_tensor_count=validate_saved_checkpoint(checkpoint,initial,run['started_unix']),
        changed_parameters=changed,initial_optimizer_states=[initial_states[rank] for rank in range(8)],
        actual_optimizer_states=states,actual_updates=updates,behavior_probability_gates=gates,
        evaluation_before=before,evaluation_after=after,
        health_measurement_passed=all(after[key]>=before[key] for key in plan['method']['health_rule']['allowed_drop']),
        production_admission=False,healthy_checkpoint_accepted=False,updates_counted_as_production=0,
        checked_unix=time.time(),semantics='Two actual guarded warmup measurements; native terminal/cleanup and independent health authentication still required')
    verified=verify_measurement(base,run,candidate=result,require_live_workers=True)
    write_once(base/SEGMENT_PATH,result)
    need(result['health_measurement_passed'] is True,'Real paired512 zero-drop health failed; preserve completed writes/checkpoint/evidence')
    print('SEARCH_G1_WARMUP57_MEASUREMENT_COMPLETE='+run['run_id'],flush=True)
    return verified


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    print(json.dumps(verify_measurement(a.base),allow_nan=False))
