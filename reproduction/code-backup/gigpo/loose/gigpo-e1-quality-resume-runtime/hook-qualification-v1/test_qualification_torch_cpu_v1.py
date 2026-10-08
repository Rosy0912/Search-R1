"""Real CPU Torch observation tests, not FSDP/official-model/GPU qualification."""
import types
import unittest
import torch
from qualification_torch_backend_v1 import ExistingForwardTensorBackend
from existing_forward_recorder_v1 import record_existing_actor_forwards


class ActualCPUActor:
    def __init__(self,module):self.module=module;self.calls=0;self.output=None
    def _forward_micro_batch(self,micro_batch,temperature):
        self.calls+=1;value=self.module(micro_batch['input'])
        self.output=value*0,value;return self.output
    def _policy_safety_guard(self,*args):raise AssertionError('No CPU monitor required by this narrow observation test')


class RealTorchObservation(unittest.TestCase):
    def worker(self):
        module=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(module.parameters(),lr=1e-6)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        return types.SimpleNamespace(actor_module_fsdp=module,actor_optimizer=optimizer,
                actor_lr_scheduler=scheduler,rollout_sharding_manager=types.SimpleNamespace())

    def test_snapshot_does_not_change_actual_CPU_rng_tensor_input_or_forward_return(self):
        worker=self.worker();backend=ExistingForwardTensorBackend(worker);actor=ActualCPUActor(worker.actor_module_fsdp)
        micro={'input':torch.tensor([[1.,2.],[3.,4.]]),'step_ids':torch.tensor([[0,0],[1,1]])}
        before=backend.capture_observer_read_state();input_copy={key:value.clone() for key,value in micro.items()}
        with record_existing_actor_forwards(actor,backend) as rows:
            actual=actor._forward_micro_batch(micro,1)
            self.assertIs(actual,actor.output)
        self.assertEqual(actor.calls,1);self.assertEqual(before,backend.capture_observer_read_state())
        self.assertTrue(all(torch.equal(value,input_copy[key]) for key,value in micro.items()))
        self.assertTrue(torch.equal(rows[0]['actual_output']['current_log_prob'],actual[1].detach()))
        self.assertFalse(rows[0]['actual_output']['current_log_prob'].requires_grad)
        self.assertIsNot(rows[0]['actual_micro']['input'],micro['input'])

    def test_rng_mutating_snapshot_refuses_before_original_forward(self):
        worker=self.worker();backend=ExistingForwardTensorBackend(worker);actor=ActualCPUActor(worker.actor_module_fsdp)
        original=backend.snapshot_micro_tensors
        def corrupt(micro):torch.rand(1);return original(micro)
        backend.snapshot_micro_tensors=corrupt
        with self.assertRaisesRegex(ValueError,'before original forward'):
            with record_existing_actor_forwards(actor,backend):actor._forward_micro_batch({'input':torch.ones(1,2)},1)
        self.assertEqual(actor.calls,0)

    def test_actual_tensor_version_mutating_probability_snapshot_is_refused(self):
        worker=self.worker();backend=ExistingForwardTensorBackend(worker);actor=ActualCPUActor(worker.actor_module_fsdp)
        original=backend.snapshot_probability_tensors
        def corrupt(entropy,current):
            with torch.no_grad():worker.actor_module_fsdp.weight.add_(1.)
            return original(entropy,current)
        backend.snapshot_probability_tensors=corrupt
        with self.assertRaisesRegex(ValueError,'altered actual model'):
            with record_existing_actor_forwards(actor,backend):actor._forward_micro_batch({'input':torch.ones(1,2)},1)
        self.assertEqual(actor.calls,1)


if __name__=='__main__':unittest.main(verbosity=2)
