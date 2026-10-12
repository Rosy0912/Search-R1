"""Scalar receipt and birth/claim mechanics; not a native A800 launch test."""
import importlib
import os
import tempfile
import unittest
from unittest.mock import patch

from reproduction.common8.actor_probe_launch_admission import (
    ADMISSION, build, verify, verify_claim_binding, verify_current,
)
from reproduction.common8.actor_probe_restore_guard import build as build_guard, _unclaimed
from reproduction.common8.formal_inline_safety import PLAN
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.verify_data import digest
from tests.reproduction.test_actor_probe_restore_guard import fixture, write


def admitted(root, step=50):
    base, run, document = fixture(root, step)
    guard = build_guard(base, run, document, 102.)
    write(base/PLAN, guard)
    admission = build(base, run, guard, 103.)
    write(base/ADMISSION, admission)
    claim = dict(run_id=run['run_id'], execution_id=run['execution_id'],
        execution_backend='ssh_local', slurm_job_id=None, shell_pid=23, shell_start_ticks=24,
        claimed_unix=104., recovery_launch_admission_sha256=digest(base/ADMISSION),
        recovery_binding_sha256=admission['recovery_binding_sha256'], origin=admission['origin'],
        state_semantics='full_state_restore_required_before_rollout')
    write(base/'logs/training-launch-claim.json', claim)
    return base, run, guard, admission, claim


class LaunchAdmissionTests(unittest.TestCase):
    def setUp(self):
        # Constructed scalar fixtures cannot exercise target Hydra mathematics.
        # All saved-origin/Head/hash/guard reconstruction code itself remains real.
        gate = patch('reproduction.common8.formal_contract.verify_formal_config',
            side_effect=lambda doc, run, base:dict(resolved_config_sha256=document_sha256(doc)))
        gate.start(); self.addCleanup(gate.stop)

    def test_all_saved_milestones_preserve_origin_without_claiming_new_work(self):
        for step in (5,50,100,150):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as root:
                base, run, guard, admission, claim = admitted(root, step)
                self.assertEqual(verify(base, run), admission)
                self.assertEqual(verify_claim_binding(base, run, claim), claim)
                self.assertEqual(admission['inherited_optimizer_writes'], step*20)
                self.assertEqual(admission['new_namespace_optimizer_writes'], 0)
                self.assertEqual(admission['next_logical_update'], step+1)
                self.assertTrue(admission['full_state_restore_required_before_rollout'])
                self.assertFalse(admission['actual_behavior_probability_accepted'])
                self.assertFalse(admission['full_final_chain_accepted'])

    def test_self_asserted_acceptance_wrong_origin_and_guard_hash_cannot_bypass_gate(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, guard, admission, claim = admitted(root)
            for key, value in [('accepted',False),('schema_version',True),
                ('new_namespace_optimizer_writes',1),('inherited_optimizer_writes',0),
                ('next_logical_update',1),('guard_plan_sha256','f'*64),
                ('source_map_sha256','f'*64),('parent_full_state_receipt_sha256','f'*64),
                ('actual_behavior_probability_accepted',True),('full_state_restore_required_before_rollout',False),
                ('training_completion_accepted',True),('full_final_chain_accepted',True)]:
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    verify(base, run, dict(admission, **{key:value}))
            fresh = dict(run); fresh.pop('actor_probe_recovery')
            with self.assertRaises(AssertionError): verify(base, fresh, admission)

    def test_mutated_guard_and_actual_head_bytes_invalidate_existing_admission(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, guard, admission, claim = admitted(root)
            write(base/PLAN, dict(guard, negative_tail_limit=.02))
            with self.assertRaises(AssertionError): verify(base, run, admission)
            write(base/PLAN, guard)
            head = base/'artifacts/commit_probe.pkl'; head.write_text('changed scalar Head bytes')
            with self.assertRaises(AssertionError): verify(base, run, admission)

    def test_old_claim_hash_or_pre_admission_time_or_wrong_saved_count_reject(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, guard, admission, claim = admitted(root)
            for key, value in [('recovery_launch_admission_sha256','f'*64),
                ('recovery_binding_sha256','f'*64),('claimed_unix',102.),
                ('state_semantics','fresh0'),('execution_id','f'*32),
                ('origin',dict(admission['origin'],actual_optimizer_writes=0))]:
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    verify_claim_binding(base, run, dict(claim, **{key:value}), admission)
            with self.assertRaises(AssertionError): _unclaimed(base, run)

    def test_live_consumer_rejects_reused_pipeline_birth_and_unowned_trainer(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, guard, admission, claim = admitted(root)
            producer = run['pipeline_identity']; leader = dict(pid=23,start_ticks=24,sid=23,uid=os.getuid())
            write(base/'logs/process-ledger.json',dict(run_id=run['run_id'],jobs={'trainer':dict(leader=leader)}))
            births={producer['pid']:dict(producer,state='S'), leader['pid']:dict(leader,state='S')}
            with patch('reproduction.common8.actor_probe_head_admission.verify_training_gate',return_value=run), \
                 patch.object(importlib.import_module('reproduction.a800.owned_processes'),'identity',side_effect=lambda pid:births[pid]):
                self.assertIn('command',run['pipeline_identity'])
                self.assertEqual(set(admission['native_birth']['pipeline_identity']),{'pid','start_ticks','sid','uid'})
                births[producer['pid']].update(state='R',command='changed instantaneous scalar command')
                self.assertEqual(verify_current(base,run,training_claim=claim),admission)
                births[producer['pid']]['start_ticks']+=1
                with self.assertRaises(AssertionError): verify_current(base,run,training_claim=claim)
                births[producer['pid']]['start_ticks']-=1
                with self.assertRaises(AssertionError): verify_current(base,run,training_claim=dict(claim,shell_pid=25))
                births[leader['pid']]['state']='Z'
                with self.assertRaises(AssertionError): verify_current(base,run,training_claim=claim)


if __name__ == '__main__': unittest.main()
