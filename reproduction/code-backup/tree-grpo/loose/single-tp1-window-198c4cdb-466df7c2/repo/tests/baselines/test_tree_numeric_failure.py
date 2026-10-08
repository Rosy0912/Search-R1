"""Real CPU actor optimization and failure capture, without rollout mocks."""
from types import MethodType
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.gradient_norm import clip_grad_norm64


def run_actor(monkeypatch, root, enabled, corrupt=False):
    monkeypatch.setenv('SEARCH_R1_TREE_RANK_AUDIT_DIR', str(root) if enabled else '')
    monkeypatch.setenv('SEARCH_R1_BASELINE_COMMIT', 'candidate-test')
    monkeypatch.setattr(TensorDict, 'cuda', lambda self, *a, **k: self)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(.01)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.actor_optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    actor.config = OmegaConf.create(dict(ppo_mini_batch_size=4, ppo_micro_batch_size=2,
        ppo_epochs=1, state_masking=True, use_kl_loss=False, use_dynamic_bsz=False,
        baseline_method='tree_grpo', entropy_coeff=0., clip_ratio=.2, grad_clip=1.))
    def forward(self, micro_batch, temperature):
        logp = self.actor_module(torch.ones(len(micro_batch), 1)).expand(-1, 3)
        return logp * 0, logp
    actor._forward_micro_batch = MethodType(forward, actor)
    if corrupt:
        model.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
    batch = TensorDict(dict(responses=torch.ones(4, 3, dtype=torch.long),
        input_ids=torch.ones(4, 5, dtype=torch.long), attention_mask=torch.ones(4, 5),
        position_ids=torch.arange(5).expand(4, -1), loss_mask=torch.ones(4, 3),
        old_log_probs=torch.zeros(4, 3), advantages=torch.ones(4, 3),
        policy_weights=torch.full((4, 3), 1/12)), batch_size=[4])
    data = DataProto(batch=batch, meta_info={'temperature': 1., 'baseline_outer_step': 7})
    before = model.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    if corrupt:
        with pytest.raises(ValueError, match='Nonfinite'):
            actor.update_policy(data)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    else:
        actor.update_policy(data)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    return model.weight.detach().clone()


def test_actual_tree_success_capture_on_off_is_equivalent(monkeypatch, tmp_path):
    a = run_actor(monkeypatch, tmp_path/'off', False)
    b = run_actor(monkeypatch, tmp_path/'on', True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not (tmp_path/'on'/'failures').exists()
    assert len(list((tmp_path/'on').glob('step-*.json'))) == 1


def test_actual_tree_failure_keeps_all_accumulation_microbatches(monkeypatch, tmp_path):
    run_actor(monkeypatch, tmp_path, True, corrupt=True)
    saved = list((tmp_path/'failures').glob('*.pt'))
    assert len(saved) == 1
    record = torch.load(saved[0], weights_only=True)
    assert record['outer_step'] == 7
    assert record['successful_optimizer_calls'] == 0
    assert record['code_commit'] == 'candidate-test'
    assert len(record['microbatches']) == 2
    for micro in record['microbatches']:
        assert micro['tensors']['input_ids'].shape == (2, 5)
        assert micro['tensors']['policy_weights'].shape == (2, 3)
        assert not micro['tensors']['log_prob'].requires_grad
        assert torch.isfinite(micro['tensors']['loss'])
    gradient = record['local_gradient_evidence'][0]
    assert gradient['finite_count'] == 0
    assert gradient['nonfinite_sample_indices'] == [0]
    assert gradient['nonfinite_sample_values'] == [float('inf')]


def test_generic_diagnostic_io_failure_preserves_optimizer_refusal(monkeypatch, tmp_path):
    file = tmp_path/'file';file.write_text('not a directory')
    monkeypatch.setenv('SEARCH_R1_BASELINE_AUDIT_DIR', str(file))
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    with pytest.raises(ValueError, match='Nonfinite actor gradient norm'):
        clip_grad_norm64(model, 1.)
