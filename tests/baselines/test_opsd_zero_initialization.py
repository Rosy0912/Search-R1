"""Compare scalar-zero memory fix against the exact previous full-vocabulary zero."""
import inspect
from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen2ForCausalLM
from tests.baselines import test_models_and_driver as fixtures
from search_r1.baselines import distillation


def previous_loss():
    source = inspect.getsource(distillation.opsd_loss)
    old = '    total = torch.where(torch.isfinite(student), student, torch.zeros_like(student)).sum() * 0'
    new = '    seed = student[..., :1]\n    total = torch.where(torch.isfinite(seed), seed, torch.zeros_like(seed)).sum() * 0'
    assert source.count(new) == 1
    namespace = dict(distillation.__dict__)
    exec(source.replace(new, old), namespace)
    return namespace['opsd_loss']


def tensors_and_settings(tokenizer, chunk):
    prompt = torch.tensor([tokenizer.encode('Question capital?')]).repeat(2, 1)
    response = torch.tensor([tokenizer.encode('<answer>Paris</answer>')]).repeat(2, 1)
    ids = torch.cat((prompt, response), 1)
    mask = torch.ones_like(response)
    mask[1, ::2] = 0
    tensors = dict(prompts=prompt, responses=response, input_ids=ids, attention_mask=torch.ones_like(ids),
                   position_ids=torch.arange(ids.shape[1])[None].repeat(2, 1), loss_mask=mask)
    return tensors, SimpleNamespace(chunk_size=chunk, beta=0., temperature=1., component_clip=.05)


@pytest.mark.parametrize('chunk', [1, 3])
def test_full_loss_and_every_real_student_gradient_remain_bitwise_equal(chunk):
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = distillation.FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, chunk)
        losses, gradients = [], []
        for loss_function in (previous_loss(), distillation.opsd_loss):
            student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
            loss = loss_function(student, teacher, tensors, tok, [['Paris'], ['Rome']], settings,
                                 tensors['loss_mask'].sum())
            loss.backward()
            losses.append(loss.detach())
            gradients.append({name: param.grad.clone() for name, param in student.named_parameters()})
        torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
        assert gradients[0].keys() == gradients[1].keys()
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=0, rtol=0)
        assert sum(float(grad.abs().sum()) for grad in gradients[1].values()) > 0
        assert all(param.grad is None for param in teacher.model.parameters())
    finally:
        fixtures.TinyModels.tearDownClass()


@pytest.mark.parametrize('empty', [False, True])
def test_nonfinite_unsupervised_logits_remain_safe_and_gradient_equal(empty):
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = distillation.FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, 3)
        if empty:
            tensors['loss_mask'].zero_()
        shape = (*tensors['input_ids'].shape, len(tok))
        values = torch.randn(shape)
        width = tensors['responses'].shape[1]
        # One NaN is in the zero seed column, another in an unselected column.
        response_start = tensors['input_ids'].shape[1] - width - 1
        values[1, response_start, 0] = float('nan')
        values[1, response_start, -1] = float('inf')
        class DirectLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(values.clone())
            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.logits)
        losses, gradients = [], []
        for loss_function in (previous_loss(), distillation.opsd_loss):
            student = DirectLogits()
            loss = loss_function(student, teacher, tensors, tok, [['Paris'], ['Rome']], settings,
                                 tensors['loss_mask'].sum().clamp_min(1))
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(student.logits.grad).all()
            losses.append(loss.detach())
            gradients.append(student.logits.grad.clone())
        torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
        torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
        if empty:
            assert losses[1] == 0 and torch.count_nonzero(gradients[1]) == 0
    finally:
        fixtures.TinyModels.tearDownClass()


@pytest.mark.parametrize('column', [0, -1])
def test_supervised_nonfinite_logits_still_rejected(column):
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = distillation.FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, 3)
        values = torch.randn((*tensors['input_ids'].shape, len(tok)), requires_grad=True)
        with torch.no_grad():
            start = tensors['input_ids'].shape[1] - tensors['responses'].shape[1] - 1
            values[0, start, column] = float('nan')
        class DirectLogits:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=values)
        for loss_function in (previous_loss(), distillation.opsd_loss):
            with pytest.raises(ValueError, match='Nonfinite OPSD logits'):
                loss_function(DirectLogits(), teacher, tensors, tok, [['Paris'], ['Rome']], settings,
                              tensors['loss_mask'].sum())
    finally:
        fixtures.TinyModels.tearDownClass()


def test_empty_response_width_retains_graph_zero_and_every_student_gradient():
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = distillation.FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, 3)
        tensors['input_ids'] = tensors['prompts']
        tensors['attention_mask'] = torch.ones_like(tensors['prompts'])
        tensors['position_ids'] = torch.arange(tensors['prompts'].shape[1])[None].repeat(2, 1)
        tensors['responses'] = tensors['responses'][:, :0]
        tensors['loss_mask'] = tensors['loss_mask'][:, :0]
        gradients = []
        for loss_function in (previous_loss(), distillation.opsd_loss):
            student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
            loss = loss_function(student, teacher, tensors, tok, [['Paris'], ['Rome']], settings, 1)
            assert loss == 0 and loss.requires_grad
            loss.backward()
            grads = {name: param.grad.clone() for name, param in student.named_parameters()}
            assert all(torch.count_nonzero(grad) == 0 for grad in grads.values())
            gradients.append(grads)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=0, rtol=0)
    finally:
        fixtures.TinyModels.tearDownClass()
