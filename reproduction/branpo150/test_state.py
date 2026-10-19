"""Small real Adam fixtures: loss of a moment/counter/nonfinite state must fail."""
import copy
import tempfile
import unittest
import contextlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .state import tensor_record,optimizer_record,rng_record


class StateTests(unittest.TestCase):
    def fixture(self):
        parameters=[torch.nn.Parameter(torch.ones(2)) for _ in range(37)]
        optimizer=torch.optim.AdamW(parameters,lr=1e-6)
        for _ in range(140):
            for p in parameters:p.grad=torch.ones_like(p)*0.1
            optimizer.step();optimizer.zero_grad()
        return optimizer
    def test_actual_adam140_roundtrip_exact(self):
        x=self.fixture();a=optimizer_record(x.state_dict(),140,37)
        y=self.fixture();y.load_state_dict(copy.deepcopy(x.state_dict()))
        self.assertEqual(a,optimizer_record(y.state_dict(),140,37))
    def test_wrong_step_rejected(self):
        x=self.fixture().state_dict();x['state'][0]['step'].fill_(139)
        with self.assertRaises(AssertionError):optimizer_record(x,140,37)
    def test_missing_moment_rejected(self):
        x=self.fixture().state_dict();del x['state'][0]['exp_avg']
        with self.assertRaises(AssertionError):optimizer_record(x,140,37)
    def test_nonfinite_moment_rejected(self):
        x=self.fixture().state_dict();x['state'][0]['exp_avg'][0]=float('nan')
        with self.assertRaises(AssertionError):optimizer_record(x,140,37)
    def test_missing_flat_parameter_rejected(self):
        x=self.fixture().state_dict();del x['state'][0]
        with self.assertRaises(AssertionError):optimizer_record(x,140,37)
    def test_actual_native_style_omegaconf_betas_without_mutation(self):
        from omegaconf import OmegaConf,ListConfig
        x=self.fixture().state_dict();x['param_groups'][0]['betas']=OmegaConf.create([0.9,0.999])
        a=optimizer_record(x,140,37)
        self.assertEqual(a['param_groups'][0]['betas'],[0.9,0.999])
        self.assertIsInstance(x['param_groups'][0]['betas'],ListConfig)
        x['param_groups'][0]['betas'][0]=0.8
        self.assertNotEqual(a,optimizer_record(x,140,37))
    def test_rng_record_does_not_advance_generators(self):
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.arange(16,dtype=torch.uint8),
                 numpy=np.random.get_state(),random=__import__('random').getstate())
        before=rng_record(rng);self.assertEqual(before,rng_record(rng))
        self.assertTrue(torch.equal(rng['cpu'],torch.get_rng_state()))
    def test_loaded_manager_compares_actual_tensors_and_preserves_rng(self):
        from .worker import verify_loaded_manager
        from types import SimpleNamespace
        optimizer=self.fixture();model=torch.nn.Linear(2,2)
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.arange(16,dtype=torch.uint8),
                 numpy=np.random.get_state(),random=__import__('random').getstate())
        scheduler=dict(last_epoch=140,_step_count=141,_last_lr=[1e-6])
        proof=dict(size='3b',native=dict(actual_updates=140,
            model_shards=[{k:tensor_record(v) for k,v in model.state_dict().items()}]*8,
            workers=[dict(rng=rng_record(rng),scheduler=scheduler,
                          optimizer=optimizer_record(optimizer.state_dict(),140,37))]*8))
        manager=SimpleNamespace(world_size=8,model=model,optimizer=optimizer,
            get_rng_state=lambda:rng,lr_scheduler=SimpleNamespace(state_dict=lambda:scheduler))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'proof.json';path.write_text('fixture')
            with patch('torch.distributed.get_rank',return_value=0),patch('torch.distributed.get_world_size',return_value=8),\
                 patch('torch.distributed.barrier'),patch('verl.utils.fsdp_utils.get_fsdp_state_ctx',return_value=contextlib.nullcontext()),\
                 patch.dict('os.environ',{'SLURM_JOB_ID':'cpu-fixture','BRANPO150_NATIVE_PROOF':str(path)}):
                verify_loaded_manager(manager,proof,tmp)
            self.assertTrue((Path(tmp)/'cold-restore-rank-00.json').exists())
        self.assertFalse(torch.cuda.is_initialized())
    def test_loaded_manager_wrong_adam_rejected(self):
        # The shared comparison itself is exact, not allclose or counter-only.
        x=self.fixture();a=optimizer_record(x.state_dict(),140,37)
        x.state[next(iter(x.state))]['exp_avg'][0]+=0.01
        self.assertNotEqual(a,optimizer_record(x.state_dict(),140,37))


if __name__=='__main__':unittest.main()
