"""Loss reporting must preserve the mini objective across micro partitions."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from search_r1.baselines import actor as actor_module


def test_loss_reporting_and_gradients_are_microbatch_invariant(monkeypatch):
    def differentiable_loss(model, teacher, tensors, tokenizer, targets, settings, denominator):
        # Unequal supervision counts exercise the real actor's token denominator.
        return (model.weight.reshape(()) * tensors['probe_features'] * tensors['loss_mask']).sum() / denominator

    monkeypatch.setattr(actor_module, 'opsd_loss', differentiable_loss)
    data = DataProto(batch=TensorDict({
        'responses': torch.zeros(6, 4, dtype=torch.long),
        'loss_mask': torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0],
                                   [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]]),
        'probe_features': torch.arange(2., 14., 2.)[:, None].expand(6, 4),
    }, batch_size=[6]), non_tensor_batch={
        'reward_model': np.array([{'ground_truth': {'target': ['answer']}} for _ in range(6)], dtype=object),
    })
    runs = []
    for micro in (1, 2, 4):
        actor = actor_module.DistillationActor.__new__(actor_module.DistillationActor)
        actor.actor_module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            actor.actor_module.weight.fill_(.5)
        actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.01)
        actor.config = SimpleNamespace(distillation=SimpleNamespace(), ppo_epochs=1,
                                       ppo_mini_batch_size=4, ppo_micro_batch_size=micro)
        actor.tokenizer = None
        actor.teacher = SimpleNamespace(on_device=nullcontext, parameter_count=1, scored_tokens=0,
                                        cost_metrics=lambda: {}, audit_metrics=lambda optimizer: {})
        gradients = []

        def step():
            grad = actor.actor_module.weight.grad.detach().clone()
            gradients.append(grad)
            actor.actor_optimizer.step()
            return grad.norm()

        actor._optimizer_step = step
        metrics = actor.update_policy(data)
        assert metrics['actor/optimizer_steps'] == 2
        assert len(metrics['actor/opsd_loss']) == 2
        # First complete mini and a smaller final mini have separate objectives.
        torch.testing.assert_close(torch.tensor(metrics['actor/opsd_loss']), torch.tensor([3., .44 * 34 / 3]))
        expected_mean = (3. + .44 * 34 / 3) / 2
        assert abs(metrics['actor/opsd_loss_mean/rank_0'] - expected_mean) < 1e-6
        runs.append((metrics, gradients, actor.actor_module.weight.detach().clone()))
    for metrics, gradients, weight in runs[1:]:
        torch.testing.assert_close(torch.tensor(metrics['actor/opsd_loss']), torch.tensor(runs[0][0]['actor/opsd_loss']))
        for grad, reference in zip(gradients, runs[0][1]):
            torch.testing.assert_close(grad, reference)
        torch.testing.assert_close(weight, runs[0][2])
