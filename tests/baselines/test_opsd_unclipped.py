"""Standard forward KL and a regression for the positive-component cap."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from search_r1.baselines.distillation import generalized_jsd


def test_default_is_forward_kl_with_analytic_gradient_and_frozen_teacher():
    student = torch.tensor([[.1, .9], [.2, .8]], requires_grad=True)
    teacher = torch.tensor([[.9, .1], [.6, .4]], requires_grad=True)
    temperature = 1.1
    loss = generalized_jsd(student, teacher, temperature=temperature)
    expected = F.kl_div(F.log_softmax(student / temperature, -1),
                        F.log_softmax(teacher.detach() / temperature, -1),
                        log_target=True, reduction='none').sum(-1)
    torch.testing.assert_close(loss, expected)
    loss.sum().backward()
    torch.testing.assert_close(student.grad,
        (student.detach().div(temperature).softmax(-1)
         - teacher.detach().div(temperature).softmax(-1)) / temperature)
    assert teacher.grad is None


def test_unclipped_counterexample_moves_toward_teacher_legacy_cap_does_not():
    teacher = torch.tensor([.9, .1]).log()
    deltas = []
    for clip in (None, .05):
        student = torch.tensor([.1, .9]).log().requires_grad_()
        loss = generalized_jsd(student, teacher, component_clip=clip)
        gradient, = torch.autograd.grad(loss, student)
        updated = student.detach() - .01 * gradient / gradient.norm()
        deltas.append(float(generalized_jsd(updated, teacher)
                            - generalized_jsd(student.detach(), teacher)))
    assert deltas[0] < 0 < deltas[1]


def test_explicit_clip_preserves_historical_component_objective():
    student = torch.tensor([.1, .9]).log()
    teacher = torch.tensor([.9, .1]).log()
    parts = F.kl_div(student.log_softmax(-1), teacher.log_softmax(-1),
                     log_target=True, reduction='none')
    torch.testing.assert_close(generalized_jsd(student, teacher, component_clip=.05),
                               parts.clamp(max=.05).sum())
    assert generalized_jsd(student, teacher) > 0


def test_candidate_config_disables_component_cap_preserves_other_protections():
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_baselines')
    assert config.baselines.distillation.component_clip is None
    assert config.baselines.distillation.beta == 0
    assert config.baselines.distillation.teacher_prompt_mode == 'user_context'
    assert config.actor_rollout_ref.actor.grad_clip == 1.0
    assert config.actor_rollout_ref.actor.clip_ratio == .2
