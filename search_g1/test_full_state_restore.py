import unittest
from types import SimpleNamespace
import torch
from search_g1.mixed_runtime import CandidateStream
from search_g1.training_state import compact_stream_state
from search_g1.full_state_restore import equal_tree

class NeverSampler:
    def __iter__(self):raise AssertionError('Restoration drew a new sampler permutation')

class RestorationTests(unittest.TestCase):
    def test_restore_order_without_random_draw(self):
        loader=SimpleNamespace(dataset=list(range(4)),sampler=NeverSampler())
        saved=dict(candidate_order=[3,1,0,2],candidate_cursor=4,epoch=2,consumed_source_rows=12,pending_source=None,pending_offset=0)
        rng=torch.get_rng_state().clone();stream=CandidateStream(loader,saved)
        self.assertEqual(stream.state(),saved);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
    def test_partial_source_preserved_consumed_source_compacted(self):
        pending=[1,2,3]
        state=dict(pending_source=pending,pending_offset=1)
        self.assertIs(compact_stream_state(state)['pending_source'],pending)
        done=compact_stream_state(dict(state,pending_offset=3))
        self.assertIsNone(done['pending_source']);self.assertEqual(done['pending_offset'],0)
    def test_invalid_order_and_cursor_rejected(self):
        loader=SimpleNamespace(dataset=list(range(4)),sampler=NeverSampler())
        state=dict(candidate_order=[0,0,2,3],candidate_cursor=0,epoch=0,consumed_source_rows=0,pending_source=None,pending_offset=0)
        with self.assertRaises(AssertionError):CandidateStream(loader,state)
        state.update(candidate_order=[0,1,2,3],candidate_cursor=5)
        with self.assertRaises(AssertionError):CandidateStream(loader,state)
    def test_optimizer_exact_readback_rejects_changed_moment(self):
        a={'state':{0:{'step':torch.tensor(100.),'exp_avg':torch.tensor([1.,2.])}},'param_groups':[{'lr':1e-7}]}
        equal_tree(a,a)
        b={'state':{0:{'step':torch.tensor(100.),'exp_avg':torch.tensor([1.,3.])}},'param_groups':[{'lr':1e-7}]}
        with self.assertRaises(AssertionError):equal_tree(a,b)
class ResumedSchedulerTests(unittest.TestCase):
    def test_actual_adam100_scheduler5_continues_at6(self):
        import json,tempfile,torch
        from pathlib import Path
        from unittest.mock import patch
        from reproduction.common8.test_scheduler import ReceiptTests
        from reproduction.common8.scheduler import OuterSchedulerEvidence,expected_lr
        t=ReceiptTests(methodName='test_prepare_is_durable_and_does_not_advance');t.setUp()
        try:
            for outer in range(5):
                for mini in range(20):
                    t.parameter.grad=torch.ones_like(t.parameter);t.optimizer.step()
                t.scheduler.step()
            t.run.update(policy_safety_mode='inline_first_five',full_state_restore=dict(source_base=str(t.base),completed_outer=5))
            (t.base/'logs/run-manifest.json').write_text(json.dumps(t.run))
            (t.base/'logs/actual-attempts.jsonl').write_text('')
            prefix=dict(completed_outer=5,writes=100,attempt_index=5)
            with patch('search_g1.full_state_restore.verified_runtime_prefix',return_value=prefix):
                e=OuterSchedulerEvidence(t.base,t.optimizer,t.scheduler,rank=0,world_size=8,job_id='12345')
            try:
                self.assertEqual((e.completed_outer,e.actual_writes,e.last_attempt_index),(5,100,5))
                e.begin_outer(6,optimizer_writes=100,batch_gate_sha256='d'*64,attempt_id='e'*32,attempt_index=6)
                for mini in range(20):
                    self.assertEqual(t.optimizer.param_groups[0]['lr'],expected_lr(5))
                    t.parameter.grad=torch.ones_like(t.parameter);t.optimizer.step()
                e.complete_outer(optimizer_writes=120,opportunities_processed=20)
                self.assertEqual(t.scheduler.last_epoch,6)
                self.assertEqual(int(t.optimizer.state[t.parameter]['step']),120)
            finally:e.close()
        finally:t.tearDown()
if __name__=='__main__':unittest.main()
