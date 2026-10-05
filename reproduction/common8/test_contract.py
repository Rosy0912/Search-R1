"""Reject plausible but incorrect global/rank budgets and incomplete write evidence."""
import copy
import json
import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
from reproduction.common8.batch_contract import check_global, check_local, independent_worker_config
from reproduction.common8.runtime import budget_rows
from reproduction.common8.claim_training import claim
from reproduction.common8.worker_evidence import append, read_console_form


def config():
    return NS(actor=NS(ppo_mini_batch_size=128, ppo_micro_batch_size=16,
                       ppo_epochs=1, optim=NS(lr=1e-6,lr_warmup_steps_ratio=0.0)),
              rollout=NS(n=1, n_agent=5, log_prob_micro_batch_size=32),
              ref=NS(log_prob_micro_batch_size=32))


def evidence():
    lines=[]
    for rank in range(8):
        for mini in range(20):
            lines.append('SEARCH_G1_OPTIMIZER_WRITE='+json.dumps(dict(rank=rank,
                context=dict(logical_update=1,minibatch_index=mini),write_count=mini+1,
                learning_rates=[1e-6])))
        lines.append('SEARCH_G1_OUTER_BUDGET='+json.dumps(dict(rank=rank,
            logical_update=1,local_trajectories=320,local_mini=16,
            local_micro=2,optimizer_writes=20,cumulative_writes=20)))
    return lines


class ContractTests(unittest.TestCase):
    def test_durable_evidence_does_not_need_console_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,SEARCH_G1_BASE=tmp):
            base=Path(tmp);(base/'logs').mkdir()
            (base/'logs/run-manifest.json').write_text(json.dumps(dict(profile='common8',run_id='unit-test-only')))
            for line in evidence():
                marker,payload=line.split('=',1)
                append(marker.removeprefix('SEARCH_G1_'),json.loads(payload))
            text=read_console_form(base,'unit-test-only')
            self.assertEqual(len(budget_rows(text,1,1)),8)
            with self.assertRaises(AssertionError):read_console_form(base,'other-run')

    def test_fresh_claim_rejects_partial_attempt_evidence(self):
        for artifact in ('training-launch-claim.json','optimizer-rank-0.jsonl',
                         'trainer.log','checkpoint'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);logs=base/'logs';logs.mkdir()
                checkpoint=base/'checkpoints';checkpoint.mkdir()
                log=logs/'trainer.log'
                (logs/'run-manifest.json').write_text(json.dumps(dict(profile='common8',
                    completed_updates=0,checkpoint_dir=str(checkpoint),train_log=str(log))))
                if artifact=='checkpoint':(checkpoint/'partial').write_text('incomplete attempt')
                else:(logs/artifact).write_text('prior attempt evidence')
                with self.assertRaises(AssertionError):claim(base)

    def test_colocated_actor_ref_keep_independent_global_config(self):
        shared=config()
        actor=independent_worker_config(shared);check_global(actor,8,1)
        actor.actor.ppo_mini_batch_size//=8;actor.actor.ppo_micro_batch_size//=8
        actor.rollout.log_prob_micro_batch_size//=8
        check_local(actor,actor=True,rollout=True,reference=False)
        reference=independent_worker_config(shared);check_global(reference,8,1)
        reference.ref.log_prob_micro_batch_size//=8
        check_local(reference,actor=False,rollout=False,reference=True)
        check_global(shared,8,1)

    def test_global_then_local_all_roles(self):
        c=config();check_global(c,8,1)
        c.actor.ppo_mini_batch_size//=8;c.actor.ppo_micro_batch_size//=8
        c.rollout.log_prob_micro_batch_size//=8;c.ref.log_prob_micro_batch_size//=8
        check_local(c,actor=True,rollout=True,reference=True)

    def test_reject_mini_as_micro(self):
        c=config();c.actor.ppo_micro_batch_size=128
        with self.assertRaises(AssertionError):check_global(c,8,1)

    def test_reject_extra_rollout_expansion(self):
        c=config();c.rollout.n=5
        with self.assertRaises(AssertionError):check_global(c,8,1)

    def test_reject_four_rank_substitute(self):
        with self.assertRaises(AssertionError):check_global(config(),4,1)

    def test_reject_wrong_ref_normalization(self):
        c=config();c.ref.log_prob_micro_batch_size=32
        with self.assertRaises(AssertionError):check_local(c,actor=False,rollout=False,reference=True)

    def test_accept_complete_ansi_console_evidence(self):
        text='\n'.join('\x1b[36m(worker)\x1b[0m '+x+'\x1b[0m' for x in evidence())
        self.assertEqual(len(budget_rows(text,1,1)),8)

    def test_claimed_twenty_without_actual_writes_rejected(self):
        lines=[x for x in evidence() if 'OUTER_BUDGET' in x]
        with self.assertRaises(AssertionError):budget_rows('\n'.join(lines),1,1)

    def test_missing_rank_write_rejected(self):
        lines=evidence();del lines[0]
        with self.assertRaises(AssertionError):budget_rows('\n'.join(lines),1,1)

    def test_duplicate_minibatch_rejected(self):
        lines=evidence();lines[1]=lines[0]
        with self.assertRaises(AssertionError):budget_rows('\n'.join(lines),1,1)


if __name__=='__main__':unittest.main()
