"""Constructed owned scalar files exercise real readers; no native GPU claim."""
import copy
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8.formal_contract import formal_mathematics, consumed_lr
from reproduction.common8.formal_attempt_audit import verify_attempt_history
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.runtime import verify_scheduler_receipts
from reproduction.common8.scheduler import expected_lr
from reproduction.common8.test_formal_inline_safety import attempt_history_fixture
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_namespace import namespace_origin
from search_g1.actor_probe_recovery import identity, binding_digest
from tests.reproduction.test_actor_probe_recovery import documents


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def rows(path, values):
    path.write_text(''.join(json.dumps(value,sort_keys=True)+'\n' for value in values))


def fixture(root, start=50, recovery=True):
    root=Path(root).resolve();base=root/'child';base.mkdir()
    _,plan,updates,schedulers=attempt_history_fixture(base,last=2,skipped_before=(1,))
    attempts=[json.loads(line) for line in (base/'logs/actual-attempts.jsonl').read_text().splitlines()]
    parent,run,binding,receipt=documents();parent_base=root/'parent';parent_base.mkdir()
    for document in (parent,run):
        document.update(completed_updates=0,acceptance_update=5,source_sha256={'constructed.py':'2'*64})
        document['budget'].update(formal_mathematics(document),profile='common8',contract_kind='fresh200_warmup57',
            model_id='Qwen/Qwen2.5-3B-Instruct',model_revision=document['model_revision'],
            common_manifest_sha256=document['input_sha256']['manifest.json'])
    run.update(train_log=str(base/'logs/training.log'),checkpoint_dir=str(base/'checkpoints/search-g1-common8'))
    parent['checkpoint_dir']=str(parent_base/'checkpoints/search-g1-common8')
    binding.update(parent_base=str(parent_base),completed_outer=start,actual_optimizer_writes=start*20-7 if recovery else 0,
                   attempt_index=start+2 if recovery else 0,aborted_attempts=2 if recovery else 0)
    receipt.update(completed_outer=start,next_logical_update=start+1,actual_optimizer_writes=binding['actual_optimizer_writes'],
                   canonical_model_receipt_sha256='9'*64)
    receipt['workers']=[dict(rank=rank,completed_outer=start,optimizer_writes=binding['actual_optimizer_writes']) for rank in range(4)]
    if not recovery:
        run.pop('actor_probe_recovery');run.pop('actor_probe_experiment_root_run_id')
    parent_path=parent_base/'logs/run-manifest.json';write(parent_path,parent)
    receipt_path=Path(parent['checkpoint_dir'])/'actor'/f'global_step_{start}'/'full-state-complete.json'
    write(receipt_path,receipt)
    binding.update(parent_manifest_sha256=digest(parent_path),full_state_receipt_sha256=digest(receipt_path))
    origin=namespace_origin(run)
    admission=dict(role='actor_probe_full_recovery_launch_admission',accepted=True,
        identity=identity(run),origin=origin,recovery_binding_sha256=binding_digest(binding))
    admission_path=base/'logs/actor-probe-recovery-launch-admission.json';write(admission_path,admission)
    policy_rejections=3 if recovery else 0
    workers=[dict(rank=rank,identity=identity(run),last_attempt_index=origin['attempt_index'],policy_rejections=policy_rejections,
        completed_outer=start,optimizer_writes=origin['actual_optimizer_writes'],canonical_model_loaded=True,
        canonical_model_receipt_sha256='9'*64,recovery_binding_sha256=binding_digest(binding),
        original_worker_state_sha256='a'*64,rebound_worker_state_sha256='b'*64,
        local_actor_parameters_sha256='c'*64) for rank in range(4)]
    restored=dict(role='actor_probe_full_recovery_restored',accepted=True,identity=identity(run),origin=origin,
        recovery_binding_sha256=binding_digest(binding),parent_full_state_receipt_sha256=binding['full_state_receipt_sha256'],
        reference_unchanged=True,no_new_optimizer_updates=True,launch_admission_sha256=digest(admission_path),workers=workers)
    if recovery:write(base/'logs/actor-probe-recovery-restored.json',restored)
    write(base/'logs/run-manifest.json',run)
    (base/'logs/training.log').write_text('Constructed CPU scalar fixtures, not a native trainer\n')
    claim=dict(run_id=run['run_id'],execution_backend='ssh_local',execution_id=run['execution_id'],
        slurm_job_id=None,hostname=socket.gethostname(),shell_pid=90,shell_start_ticks=900)
    write(base/'logs/training-launch-claim.json',claim)
    write(base/'logs/native-allocation-admission.json',dict(accepted=True,run_id=run['run_id'],
        execution_backend='ssh_local',execution_id=run['execution_id'],slurm_job_id=None,node=socket.gethostname()))
    leader=dict(pid=90,start_ticks=900,sid=90,uid=0)
    write(base/'logs/process-ledger.json',dict(run_id=run['run_id'],jobs={'trainer':dict(leader=leader,
        ledger={str(rank):dict(pid=100+rank,start_ticks=1000+rank,sid=90,uid=0) for rank in range(4)})}))
    def adjust(value):
        value=copy.deepcopy(value);value['run_id']=run['run_id']
        for key in ('logical_update','step','completed_outer','completed_outer_before','completed_outer_after'):
            if key in value:value[key]+=start
        if 'attempt_index' in value:
            value['attempt_index']+=origin['attempt_index']
            value['attempt_id']=f"{value['attempt_index']+10000:032x}"
        for key in ('optimizer_writes_before','optimizer_writes_after','cumulative_writes','write_count',
                    'writes_before','writes_after'):
            if key in value:value[key]+=origin['actual_optimizer_writes']
        if 'cumulative_aborted_attempts' in value:value['cumulative_aborted_attempts']+=origin['aborted_attempts']
        if 'context' in value:value['context']=adjust(dict(value['context'],run_id=run['run_id']));value['context'].pop('run_id')
        if 'prior_optimizer_writes' in value:value['prior_optimizer_writes']+=origin['actual_optimizer_writes']
        context=value.get('context',value)
        if 'attempt_index' in context:
            gate=document_sha256(context['attempt_index'])
            if 'behavior_probability_gate_sha256' in value:value['behavior_probability_gate_sha256']=gate
            if 'batch_gate_sha256' in value:value['batch_gate_sha256']=gate
        if 'learning_rates' in value:value['learning_rates']=[consumed_lr(context['logical_update'])]
        if 'slurm_job_id' in value:
            value.update(slurm_job_id=None,execution_backend='ssh_local',execution_id=run['execution_id'])
        return value
    attempts=[adjust(value) for value in attempts]
    for value in attempts:value['real_rank_count']=4
    updates=[adjust(value) for value in updates]
    for value in updates:
        value.update(reward_schedule=run['reward_schedule'],reward_logical_update=value['step'],
            reward_mode='search_g1' if value['step']<=100 else 'binary_em')
        value['metrics']['critic/rewards/mean']=.4
    rows(base/'logs/actual-attempts.jsonl',attempts);rows(base/'logs/actual-updates.jsonl',updates)
    for rank in range(4):
        path=base/'logs'/f'optimizer-rank-{rank}.jsonl'
        values=[adjust(json.loads(line)) for line in path.read_text().splitlines()]
        for value in values:
            if value['kind']=='OUTER_BUDGET':value.update(local_trajectories=640,local_mini=32)
            if value['kind']=='POLICY_MONITOR':
                value['optimizer_writes']+=origin['actual_optimizer_writes']
                value['rejection_count']+=policy_rejections;value['monitoring_scope']='fixed eight trajectories per rank across length strata; not full rollout'
                m=value['monitor_identity'];m['selection_indices']+= [i+320 for i in m['selection_indices']]
                m['actor_token_lengths']=sorted(m['actor_token_lengths']*2)
        rows(path,values)
        def header(value,observed):
            return dict(value,schema_version=1,role='fresh200_outer_scheduler',contract_kind='fresh200_warmup57',
                rank=rank,world_size=4,run_id=run['run_id'],source_commit=run['source_commit'],
                source_map_sha256=document_sha256(run['source_sha256']),scheduler_contract_sha256=document_sha256(run['budget']),
                slurm_job_id=None,execution_backend='ssh_local',execution_id=run['execution_id'],hostname=socket.gethostname(),
                warmup_steps=57,base_learning_rate=1e-6,observed_unix=observed,process_pid=100+rank,process_start_ticks=1000+rank)
        anchor=header(dict(kind='RESTORE',completed_outer=start,actual_writes=origin['actual_optimizer_writes'],
            last_attempt_index=origin['attempt_index'],policy_rejections=policy_rejections,scheduler_advances=0,
            scheduler_last_epoch=start,scheduler_step_count=start+1,learning_rates=[expected_lr(start)],
            recovery_binding_sha256=binding_digest(binding),parent_run_id=parent['run_id'],parent_execution_id=parent['execution_id'],
            full_state_receipt_sha256=binding['full_state_receipt_sha256']),1.)
        values=[anchor] if recovery else []
        for index,raw in enumerate(schedulers[str(rank)]['selected_rows'],2):
            value=adjust(raw);step=value['logical_update'];advanced=value['kind']=='COMPLETE'
            if value['kind']=='ABORTED_ATTEMPT':
                value.update(recoverable_guard=True,retained_partial_adam_writes=True,
                             no_automatic_rollback=True,scheduler_unchanged_on_rejection=True)
            value.update(scheduler_last_epoch_before=step-1,scheduler_last_epoch_after=step if advanced else step-1,
                scheduler_step_count_before=step,scheduler_step_count_after=step+1 if advanced else step,
                consumed_learning_rates=[consumed_lr(step)],next_learning_rates=[expected_lr(step if advanced else step-1)])
            values.append(header(value,float(index)))
        rows(base/'logs'/f'scheduler-rank-{rank}.jsonl',values)
    return base,run,plan,updates


class NamespaceAuditTests(unittest.TestCase):
    def audit(self,base,run,plan,updates):
        gates={str(row['step']):{'proof_sha256':document_sha256(row['attempt_index'])} for row in updates}
        def gate(_base,_run,step,**kwargs):
            return dict(proof_sha256=document_sha256(kwargs['update_record']['attempt_index']
                if 'update_record' in kwargs else kwargs['attempt_index']))
        with patch('reproduction.common8.behavior_runtime.read_gate_receipt',side_effect=gate):
            schedulers=verify_scheduler_receipts(base,run,updates[0]['step'],updates[-1]['step'],gates)
            return verify_attempt_history(base,run,updates,schedulers,last=updates[-1]['step'],plan_sha256=plan),schedulers

    def test_real_readers_bind_saved_counters_warmup_and_retry_monitors(self):
        for start in (5,50,100):
            with self.subTest(start=start),tempfile.TemporaryDirectory() as root:
                base,run,plan,updates=fixture(root,start)
                proof,schedulers=self.audit(base,run,plan,updates)
                self.assertEqual(proof['actual_attempt_count'],3)
                self.assertEqual(proof['fully_skipped_attempt_count'],1)
                self.assertEqual(proof['processed_opportunities_per_rank'],60)
                self.assertEqual(proof['actual_optimizer_writes_per_rank'],start*20-7+40)
                self.assertEqual(proof['unique_global_monitor_count'],120) # Four replicas count once.
                self.assertFalse(proof['production_admission'])
                self.assertTrue(all(p['namespace_new_complete_outers']==2 for p in schedulers.values()))

    def test_fresh_four_rank_readers_keep_original_zero_counter_origin(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,plan,updates=fixture(root,start=0,recovery=False)
            proof,schedulers=self.audit(base,run,plan,updates)
            self.assertEqual(proof['actual_optimizer_writes_per_rank'],40)
            self.assertEqual(proof['namespace_new_complete_outers'],2)
            self.assertIsNone(proof['restore_proof_sha256'])
            self.assertTrue(all(p['restore_anchor'] is None for p in schedulers.values()))

    def test_existing_eight_rank_attempt_reader_keeps_original_monitor_accounting(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve();run,plan,updates,schedulers=attempt_history_fixture(base,last=2,skipped_before=(1,))
            run.update(profile='common8',training_world_size=8,budget={'world_size':8})
            def gate(_base,_run,step,**kwargs):
                return dict(proof_sha256=document_sha256(kwargs['update_record']['attempt_index']
                    if 'update_record' in kwargs else kwargs['attempt_index']))
            with patch('reproduction.common8.behavior_runtime.read_gate_receipt',side_effect=gate):
                proof=verify_attempt_history(base,run,updates,schedulers,last=2,plan_sha256=plan)
            self.assertEqual((proof['processed_opportunities_per_rank'],proof['actual_optimizer_writes_per_rank']),(60,40))
            self.assertEqual(proof['unique_global_monitor_count'],120)

    def test_forged_origin_missing_anchor_wrong_birth_or_inherited_lost_writes_rejected(self):
        for failure in ('missing_anchor','wrong_cursor','wrong_lr','wrong_birth','lost_writes','changed_parent','old_prefix'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as root:
                base,run,plan,updates=fixture(root)
                path=base/'logs/scheduler-rank-2.jsonl';values=[json.loads(line) for line in path.read_text().splitlines()]
                if failure=='missing_anchor':values.pop(0)
                elif failure=='wrong_cursor':values[0]['last_attempt_index']+=1
                elif failure=='wrong_lr':values[1]['consumed_learning_rates']=[0.]
                elif failure=='wrong_birth':values[1]['process_start_ticks']+=1
                elif failure=='lost_writes':values[0]['actual_writes']+=20
                elif failure=='changed_parent':
                    parent=Path(run['actor_probe_recovery']['parent_base'])/'logs/run-manifest.json';parent.write_text(parent.read_text()+' ')
                else:
                    updates=[dict(updates[0],step=1)]+updates;rows(base/'logs/actual-updates.jsonl',updates)
                rows(path,values)
                with self.assertRaises((AssertionError,ValueError)):self.audit(base,run,plan,updates)


if __name__=='__main__':unittest.main()
