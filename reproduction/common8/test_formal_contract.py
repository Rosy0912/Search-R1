"""CPU contract/receipt rejection tests, never native training evidence."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8.formal_contract import (
    CONTRACT_KIND, FORMAL_MATHEMATICS, PROBE_ROUNDS, SAVE_STEPS,
    assert_formal_run, consumed_lr, verify_formal_config,
)
from reproduction.common8.policy_calibration_plan import MATHEMATICS, canonical_bytes
from reproduction.common8.runtime import budget_rows, refresh_after_checkpoint, verify_scheduler_receipts
from reproduction.common8 import test_configuration


class FormalContractTests(unittest.TestCase):
    def test_new_contract_keeps_original_diagnostic_math_immutable(self):
        self.assertEqual(MATHEMATICS['target_actual_outer_updates'],300)
        self.assertEqual(MATHEMATICS['learning_rate_schedule'],'constant')
        self.assertEqual(MATHEMATICS['reference_kl_coefficient'],.001)
        self.assertEqual(FORMAL_MATHEMATICS['target_actual_outer_updates'],200)
        self.assertEqual(FORMAL_MATHEMATICS['lr_warmup_steps'],57)
        self.assertEqual(FORMAL_MATHEMATICS['entropy_coefficient'],0.)
        self.assertEqual(FORMAL_MATHEMATICS['reference_kl_coefficient'],0.)
        self.assertEqual(PROBE_ROUNDS,(0,50,100,150))
        self.assertEqual(SAVE_STEPS,(5,50,100,150,200))

    def test_exact_consumed_lr_boundaries_and_bad_outer_rejection(self):
        for step,value in ((1,0.),(2,1e-6*(1/57)),(57,1e-6*(56/57)),(58,1e-6),(200,1e-6)):
            self.assertEqual(consumed_lr(step),value)
        for step in (0,201,True,1.0):
            with self.subTest(step=step),self.assertRaises(AssertionError):consumed_lr(step)

    def test_formal_config_rejects_horizon_objective_warmup_micro_and_old_scope(self):
        from reproduction.common8.configuration import resolved
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as root:
            base,run=test_configuration.ConfigurationTests().fixture(root)
            _,config,_=resolved(base)
            document=OmegaConf.to_container(config,resolve=True)
            proof=verify_formal_config(document,run,base)
            self.assertEqual(proof['contract_kind'],CONTRACT_KIND)
            for path,value in ((('trainer','total_training_steps'),301),
                    (('actor_rollout_ref','actor','optim','lr_warmup_steps'),0),
                    (('actor_rollout_ref','actor','optim','lr_warmup_steps_ratio'),57/200),
                    (('actor_rollout_ref','actor','entropy_coeff'),.001),
                    (('actor_rollout_ref','actor','kl_loss_coef'),.001),
                    (('actor_rollout_ref','actor','use_kl_loss'),False),
                    (('actor_rollout_ref','actor','ppo_micro_batch_size'),128),
                    (('actor_rollout_ref','rollout','temperature'),.5)):
                bad=copy.deepcopy(document);part=bad
                for key in path[:-1]:part=part[key]
                part[path[-1]]=value
                with self.subTest(path=path),self.assertRaises(AssertionError):verify_formal_config(bad,run,base)
            for key,value in (('contract_kind','constant300'),('target_updates',300),('completed_updates',1)):
                bad=copy.deepcopy(run);bad[key]=value
                with self.subTest(key=key),self.assertRaises(AssertionError):assert_formal_run(bad)

    def scheduler_fixture(self,root,last=2):
        base,run=test_configuration.ConfigurationTests().fixture(root)
        run.update(source_commit='a'*40)
        source=hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest()
        contract=hashlib.sha256(canonical_bytes(run['budget'])).hexdigest()
        gates={str(i):dict(proof_sha256=str(i)*64) for i in range(1,last+1)}
        (base/'logs/training-launch-claim.json').write_text(json.dumps(dict(run_id=run['run_id'],slurm_job_id='123',
            shell_pid=10,shell_start_ticks=999)))
        leader=dict(pid=10,start_ticks=999,sid=10,uid=123)
        ledger={str(100+rank):dict(pid=100+rank,start_ticks=1000+rank,sid=10,uid=123) for rank in range(8)}
        (base/'logs/process-ledger.json').write_text(json.dumps(dict(run_id=run['run_id'],
            jobs={'trainer':dict(leader=leader,ledger=ledger,state='STARTED')})))
        (base/'logs/run-manifest.json').write_text(json.dumps(run))
        for rank in range(8):
            rows=[]
            for step in range(1,last+1):
                for ordinal,kind in enumerate(('BEGIN','COMMIT_PREPARED','COMPLETE')):
                    after=step if kind=='COMPLETE' else step-1
                    written=kind!='BEGIN'
                    rows.append(dict(schema_version=1,kind=kind,role='fresh200_outer_scheduler',
                        contract_kind=CONTRACT_KIND,run_id=run['run_id'],source_commit=run['source_commit'],
                        source_map_sha256=source,scheduler_contract_sha256=contract,slurm_job_id='123',rank=rank,world_size=8,
                        logical_update=step,warmup_steps=57,base_learning_rate=1e-6,
                        process_pid=100+rank,process_start_ticks=1000+rank,
                        observed_unix=100.+step+ordinal*.1,
                        batch_gate_sha256=gates[str(step)]['proof_sha256'],
                        completed_outer_before=step-1,completed_outer_after=after,
                        scheduler_last_epoch_before=step-1,scheduler_last_epoch_after=after,
                        scheduler_step_count_before=step,scheduler_step_count_after=after+1,
                        writes_before=(step-1)*20,writes_after=(step if written else step-1)*20,
                        actual_optimizer_writes=20 if written else 0,scheduler_advances=int(kind=='COMPLETE'),
                        consumed_learning_rates=[consumed_lr(step)],next_learning_rates=[1e-6*min(after/57,1.)]))
            (base/'logs'/f'scheduler-rank-{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return base,run,gates

    def test_durable_scheduler_binding_rejects_duplicate_rejected_wrong_counter_or_lr(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root)
            proof=verify_scheduler_receipts(base,run,1,2,gates)
            self.assertEqual(len(proof),8)
            path=base/'logs/scheduler-rank-3.jsonl';original=path.read_bytes()
            for field,value in (('kind','REJECTED'),('source_commit','b'*40),
                    ('scheduler_step_count_after',20),('scheduler_last_epoch_after',20),
                    ('actual_optimizer_writes',19),('writes_after',19),
                    ('consumed_learning_rates',[1e-6]),('batch_gate_sha256','wrong'),
                    ('process_pid',999),('contract_kind','old')):
                rows=[json.loads(line) for line in original.splitlines()];rows[1][field]=value
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.subTest(field=field),self.assertRaises(AssertionError):verify_scheduler_receipts(base,run,1,2,gates)
            path.write_bytes(original+original.splitlines()[0]+b'\n')
            with self.assertRaises(AssertionError):verify_scheduler_receipts(base,run,1,2,gates)

    def test_fatal_console_invalidates_otherwise_complete_scheduler_triplets(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root)
            self.assertEqual(len(verify_scheduler_receipts(base,run,1,2,gates)),8)
            Path(run['train_log']).write_text('SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='+json.dumps(dict(
                run_id=run['run_id'],actual_scheduler_last_epoch=2,
                actual_scheduler_step_count=3,writes_after=40,fatal_no_automatic_resume=True))+'\n')
            with self.assertRaisesRegex(AssertionError,'Fatal scheduler console'):
                verify_scheduler_receipts(base,run,1,2,gates)

    def test_first_outer_twenty_actual_zero_lr_writes_not_constant_lr(self):
        from reproduction.common8.test_contract import evidence
        lines=evidence();self.assertEqual(len(budget_rows('\n'.join(lines),1,1)),8)
        changed=lines[0].replace('[0.0]','[1e-06]')
        with self.assertRaises(AssertionError):budget_rows('\n'.join([changed]+lines[1:]),1,1)

    def test_scheduler_births_must_match_independent_owned_ledger_and_distinct_ranks(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root)
            for mode in ('unknown','shared'):
                path=base/'logs/scheduler-rank-3.jsonl';original=path.read_bytes()
                rows=[json.loads(line) for line in original.splitlines()]
                for row in rows:
                    row['process_pid']=999 if mode=='unknown' else 100
                    row['process_start_ticks']=9999 if mode=='unknown' else 1000
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.subTest(mode=mode),self.assertRaises(AssertionError):
                    verify_scheduler_receipts(base,run,1,2,gates)
                path.write_bytes(original)

    def test_actual_five_scheduler_interval_matches_full_two_hundred_receipts(self):
        from reproduction.common8.verify_full import verify_five_scheduler_prefix
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root,last=200)
            five=verify_scheduler_receipts(base,run,1,5,gates)
            full=verify_scheduler_receipts(base,run,1,200,gates)
            for rank in range(8):
                selected=verify_five_scheduler_prefix(five[str(rank)],full[str(rank)])
                self.assertEqual(len(selected),15)
                bad=copy.deepcopy(five[str(rank)]);bad['selected_rows']=bad['selected_rows'][:10]
                with self.assertRaises(AssertionError):verify_five_scheduler_prefix(bad,full[str(rank)])

    def test_terminal200_never_starts_unused_probe_calibration(self):
        with patch('subprocess.run') as subprocess_run,patch('requests.Session') as session:
            self.assertIsNone(refresh_after_checkpoint(200,'unused'))
        subprocess_run.assert_not_called();session.assert_not_called()


if __name__=='__main__':unittest.main()
