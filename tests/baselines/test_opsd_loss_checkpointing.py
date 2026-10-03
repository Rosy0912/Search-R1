"""Non-reentrant divergence recomputation preserves complete OPSD mathematics."""
import math
from unittest.mock import patch
import pytest
import torch
from transformers import Qwen2ForCausalLM
from tests.baselines import test_models_and_driver as fixtures
from tests.baselines.test_opsd_zero_initialization import tensors_and_settings
from search_r1.baselines import distillation
from search_r1.baselines.controller import validate_config


@pytest.mark.parametrize('chunk', [1, 3])
@pytest.mark.parametrize('beta,temperature,clip', [(0.,1.,.05),(0.,2.,None),(.4,.5,.05),(.4,1.,None),(1.,2.,.05),(1.,1.,None)])
def test_checkpoint_loss_and_every_student_gradient_exact_and_teacher_not_recomputed(chunk, beta, temperature, clip):
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = distillation.FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, chunk)
        settings.beta, settings.temperature, settings.component_clip = beta, temperature, clip
        losses, gradients = [], []
        expected_chunks = math.ceil(tensors['responses'].shape[1] / chunk)
        calls = []
        hook = teacher.model.get_output_embeddings().register_forward_hook(lambda *args: calls.append(1))
        try:
            for enabled in (False, True):
                calls.clear()
                settings.loss_checkpointing = enabled
                student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
                loss = distillation.opsd_loss(student, teacher, tensors, tok, [['Paris'], ['Rome']],
                                             settings, tensors['loss_mask'].sum())
                assert len(calls) == expected_chunks
                loss.backward()
                assert len(calls) == expected_chunks  # Teacher projections are not checkpointed.
                losses.append(loss.detach())
                gradients.append({name: param.grad.clone() for name, param in student.named_parameters()})
        finally:
            hook.remove()
        torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=0, rtol=0)
        assert all(param.grad is None and not param.requires_grad for param in teacher.model.parameters())
    finally:
        fixtures.TinyModels.tearDownClass()


@pytest.mark.parametrize('all_empty', [False, True])
def test_masked_nonfinite_chunks_and_zero_mask_stay_finite_with_checkpoint(all_empty):
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = distillation.FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, 3)
        if all_empty:
            tensors['loss_mask'].zero_()
        values = torch.randn((*tensors['input_ids'].shape, len(tok)))
        offset = tensors['input_ids'].shape[1] - tensors['responses'].shape[1] - 1
        values[1, offset, 0] = float('nan')
        values[1, offset, -1] = float('inf')
        losses, gradients = [], []
        class DirectLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(values.clone())
            def forward(self, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(logits=self.logits)
        for enabled in (False, True):
            settings.loss_checkpointing = enabled
            student = DirectLogits()
            loss = distillation.opsd_loss(student, teacher, tensors, tok, [['Paris'], ['Rome']], settings,
                                         tensors['loss_mask'].sum().clamp_min(1))
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(student.logits.grad).all()
            losses.append(loss.detach())
            gradients.append(student.logits.grad.clone())
        torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
        torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
        if all_empty:
            assert losses[1] == 0 and torch.count_nonzero(gradients[1]) == 0
    finally:
        fixtures.TinyModels.tearDownClass()


def test_nonreentrant_recompute_occurs_and_preserves_outer_autocast_state():
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = distillation.FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, 3)
        loss_function = distillation.generalized_jsd
        records, losses, gradients = [], [], []
        def recorded(*args):
            records.append(dict(cpu_autocast=torch.is_autocast_enabled('cpu'),
                                student_dtype=args[0].dtype, teacher_dtype=args[1].dtype))
            return loss_function(*args)
        for enabled in (False, True):
            settings.loss_checkpointing = enabled
            student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
            records.clear()
            with patch.object(distillation, 'generalized_jsd', side_effect=recorded):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    assert torch.is_autocast_enabled('cpu')
                    loss = distillation.opsd_loss(student, teacher, tensors, tok, [['Paris'], ['Rome']],
                                                 settings, tensors['loss_mask'].sum())
                    assert torch.is_autocast_enabled('cpu')
                count_before_backward = len(records)
                assert not torch.is_autocast_enabled('cpu')
                loss.backward()
                assert not torch.is_autocast_enabled('cpu')
                if enabled:
                    assert len(records) > count_before_backward
                else:
                    assert len(records) == count_before_backward
                assert all(record['cpu_autocast'] and record['student_dtype'] == torch.bfloat16
                           and record['teacher_dtype'] == torch.float32 for record in records)
            losses.append(loss.detach())
            gradients.append({name: param.grad.clone() for name, param in student.named_parameters()})
        torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=0, rtol=0)
    finally:
        fixtures.TinyModels.tearDownClass()


@pytest.mark.parametrize('value', ['true', 1, None])
def test_config_rejects_nonboolean_checkpoint_flag(value):
    config = fixtures.config('opsd')
    config.actor_rollout_ref.actor.use_kl_loss = False
    config.actor_rollout_ref.actor.entropy_coeff = 0
    config.algorithm.kl_ctrl.kl_coef = 0
    config.baselines.distillation.teacher_path = '/teacher'
    config.baselines.distillation.privileged_answers = True
    config.baselines.distillation.loss_checkpointing = value
    with pytest.raises(ValueError, match='loss_checkpointing must be a boolean'):
        validate_config(config)


def test_config_checkpoint_flag_defaults_false_and_accepts_true():
    config = fixtures.config('opsd')
    assert config.baselines.distillation.loss_checkpointing is False
    config.actor_rollout_ref.actor.use_kl_loss = False
    config.actor_rollout_ref.actor.entropy_coeff = 0
    config.algorithm.kl_ctrl.kl_coef = 0
    config.baselines.distillation.teacher_path = '/teacher'
    config.baselines.distillation.privileged_answers = True
    config.baselines.distillation.loss_checkpointing = True
    validate_config(config)


@pytest.mark.parametrize('beta', [0., .4, 1.])
def test_component_clip_is_actually_active_and_checkpoint_preserves_its_gradient(beta):
    from torch.utils.checkpoint import checkpoint
    teacher = torch.tensor([[20., -20., -20.], [-20., 20., -20.]])
    initial = torch.tensor([[-20., 20., -20.], [20., -20., -20.]])
    losses, gradients = [], []
    for enabled in (False, True):
        student = initial.clone().requires_grad_()
        if enabled:
            divergence = checkpoint(distillation.generalized_jsd, student, teacher, beta, 1., .05,
                                    use_reentrant=False)
        else:
            divergence = distillation.generalized_jsd(student, teacher, beta, 1., .05)
        assert torch.all(divergence < distillation.generalized_jsd(student, teacher, beta, 1., None))
        divergence.sum().backward()
        losses.append(divergence.detach())
        gradients.append(student.grad.clone())
    torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)


@pytest.mark.parametrize('column', [0, -1])
def test_checkpoint_still_rejects_supervised_nan_before_divergence(column):
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = distillation.FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, 3)
        settings.loss_checkpointing = True
        values = torch.randn((*tensors['input_ids'].shape, len(tok)), requires_grad=True)
        offset = tensors['input_ids'].shape[1] - tensors['responses'].shape[1] - 1
        with torch.no_grad():
            values[0, offset, column] = float('nan')
        class DirectLogits:
            def __call__(self, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(logits=values)
        with patch.object(distillation, 'checkpoint', side_effect=AssertionError('Checkpoint should not run')):
            with pytest.raises(ValueError, match='Nonfinite OPSD logits'):
                distillation.opsd_loss(DirectLogits(), teacher, tensors, tok, [['Paris'], ['Rome']], settings,
                                      tensors['loss_mask'].sum())
    finally:
        fixtures.TinyModels.tearDownClass()
