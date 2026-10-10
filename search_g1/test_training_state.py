import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from search_g1.training_state import atomic_save,cpu_tree,driver_rng,compact_stream_state

class StateTests(unittest.TestCase):
    def test_partial_source_batch_retained_and_consumed_batch_removed(self):
        pending=['first','second','third']
        state=dict(pending_source=pending,pending_offset=1,candidate_cursor=512,
                   candidate_order=[3,1,2],consumed_source_rows=1,epoch=0)
        compact=compact_stream_state(state)
        self.assertIs(compact['pending_source'],pending)
        self.assertEqual(compact['pending_source'][compact['pending_offset']:],['second','third'])
        state['pending_offset']=3
        compact=compact_stream_state(state)
        self.assertIsNone(compact['pending_source'])
        self.assertEqual(compact['pending_offset'],0)
        self.assertEqual(compact['candidate_cursor'],512)
        self.assertIs(state['pending_source'],pending)
        for invalid in [-1,4]:
            with self.assertRaises(AssertionError):
                compact_stream_state(dict(state,pending_offset=invalid))
        with self.assertRaises(AssertionError):
            compact_stream_state(dict(state,pending_source=None,pending_offset=1))

    def test_post_generation_driver_rng_roundtrip_without_extra_draws(self):
        import random
        import numpy as np
        original=driver_rng()
        try:
            torch.rand(7);np.random.rand(7);random.random()
            post=driver_rng()
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'after-generation.pt'
                atomic_save(path,{'driver_rng':post})
                self.assertTrue(torch.equal(torch.get_rng_state(),post['torch_cpu']))
                expected=(torch.rand(8),np.random.rand(8),random.random())
                loaded=torch.load(path,map_location='cpu',weights_only=False)['driver_rng']
                torch.set_rng_state(loaded['torch_cpu']);np.random.set_state(loaded['numpy'])
                random.setstate(loaded['python'])
                self.assertTrue(torch.equal(expected[0],torch.rand(8)))
                self.assertTrue(np.array_equal(expected[1],np.random.rand(8)))
                self.assertEqual(expected[2],random.random())
        finally:
            torch.set_rng_state(original['torch_cpu']);np.random.set_state(original['numpy'])
            random.setstate(original['python'])

    def test_real_adam_scheduler_roundtrip_and_no_snapshot_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=torch.nn.Parameter(torch.tensor([2.]));opt=torch.optim.AdamW([p],lr=1e-6)
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:min(n/57,1))
            p.grad=torch.tensor([.3]);opt.step();scheduler.step()
            before=driver_rng();path=Path(temp)/'rank.pt'
            proof=atomic_save(path,{'optimizer':cpu_tree(opt.state_dict()),'scheduler':scheduler.state_dict(),'rng':before})
            loaded=torch.load(path,map_location='cpu',weights_only=False)
            q=torch.nn.Parameter(torch.tensor([9.]));other=torch.optim.AdamW([q],lr=1.)
            other.load_state_dict(loaded['optimizer'])
            self.assertEqual(int(other.state[q]['step']),1)
            self.assertTrue(torch.equal(other.state[q]['exp_avg'],opt.state[p]['exp_avg']))
            self.assertEqual(loaded['scheduler']['last_epoch'],scheduler.last_epoch)
            self.assertEqual(proof['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(torch.equal(torch.get_rng_state(),before['torch_cpu']))
            with self.assertRaises(AssertionError):atomic_save(path,{})

if __name__=='__main__':unittest.main()
