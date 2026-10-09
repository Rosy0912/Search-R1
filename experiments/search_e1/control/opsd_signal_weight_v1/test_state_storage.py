"""Physical sharing restores complete equal Adam and rejects unequal ranks."""
import tempfile
import unittest
from pathlib import Path

from checkpoint import fileproof
from state_storage import STORAGE,optimizer_payload,hydrate_rank_optimizer
from trainer import optimizer_tensor_digest


class ExactSharedAdam(unittest.TestCase):
    def optimizer(self):
        import torch
        return dict(state={0:dict(step=torch.tensor(5.),exp_avg=torch.tensor([.1,.2]),exp_avg_sq=torch.tensor([.3,.4]))},
                    param_groups=[dict(params=[0],lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01)])

    def test_all_four_actual_rank_states_reconstruct_same_full_adam(self):
        import torch
        value=self.optimizer();digest=optimizer_tensor_digest(value)
        proofs=[dict(rank=r,optimizer_digest=digest) for r in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'optimizer-shared.pt'
            torch.save(optimizer_payload(value,proofs,'run','config',5,digest),path)
            ref=fileproof(path);cp=dict(shared_optimizer=ref,all_rank_optimizer_tensor_sha256=[digest]*4)
            for rank in range(4):
                state=dict(rank=rank,run_id='run',config_sha256='config',optimizer_updates=5,
                    optimizer_tensor_sha256=digest,optimizer_storage=STORAGE,optimizer_ref=ref,rng=dict(rank_specific=rank))
                actual=hydrate_rank_optimizer(state,cp,lambda p:torch.load(p,map_location='cpu',weights_only=False),optimizer_tensor_digest)
                self.assertEqual(optimizer_tensor_digest(actual['optimizer']),digest)
                self.assertEqual(actual['rng']['rank_specific'],rank)

    def test_unequal_actual_optimizer_cannot_be_shared(self):
        value=self.optimizer();digest=optimizer_tensor_digest(value)
        proofs=[dict(rank=r,optimizer_digest=digest if r!=3 else 'different') for r in range(4)]
        with self.assertRaises(ValueError):optimizer_payload(value,proofs,'run','config',5,digest)


if __name__=='__main__':unittest.main()
