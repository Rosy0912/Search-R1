import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from search_g1.training_state import atomic_save,cpu_tree,driver_rng

class StateTests(unittest.TestCase):
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
