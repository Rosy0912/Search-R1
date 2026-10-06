"""Real Qwen full-vocabulary loss/gradient equivalence, including masked history."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2ForCausalLM

from search_r1.baselines.distillation import FrozenTeacher, opsd_loss
from tests.baselines import test_models_and_driver as fixtures
from tests.baselines.test_opsd_zero_initialization import tensors_and_settings


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('chunk', [1, 3])
@pytest.mark.parametrize('teacher_micro', [None, 1])
def test_native_head_slice_preserves_loss_and_all_parameter_gradients(checkpoint, chunk, teacher_micro):
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, chunk)
        settings.component_clip = None
        settings.loss_checkpointing = checkpoint
        # Differing right padding and observation masks exercise absolute
        # response alignment, including its first prediction predecessor.
        tensors['responses'][1, -1] = tok.pad_token_id
        tensors['input_ids'][1, -1] = tok.pad_token_id
        tensors['attention_mask'][1, -1] = 0
        tensors['loss_mask'][1, -1] = 0
        outputs = []
        for only in (False, True):
            settings.response_only_logits = only
            settings.teacher_forward_micro_batch = teacher_micro if only else None
            student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
            head_shapes = []
            handle = student.lm_head.register_forward_pre_hook(lambda _, a: head_shapes.append(tuple(a[0].shape)))
            loss = opsd_loss(student, teacher, tensors, tok, [['Paris'], ['Rome']], settings,
                             tensors['loss_mask'].sum())
            loss.backward()
            handle.remove()
            assert torch.isfinite(loss) and loss >= -1e-6
            assert head_shapes[0][1] == (tensors['responses'].shape[1]+1 if only else tensors['input_ids'].shape[1])
            outputs.append((loss.detach(), {n: p.grad.clone() for n,p in student.named_parameters()}))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1e-7, rtol=1e-6)
        assert outputs[0][1].keys() == outputs[1][1].keys()
        for name in outputs[0][1]:
            torch.testing.assert_close(outputs[0][1][name], outputs[1][1][name], atol=1e-7, rtol=1e-5)
        assert all(p.grad is None for p in teacher.model.parameters())
    finally:
        fixtures.TinyModels.tearDownClass()


def test_unsupported_head_slice_cannot_silently_keep_wrong_logits():
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, 3)
        settings.response_only_logits = True
        model = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
        class IgnoresSlice:
            config = model.config
            def __call__(self, **kwargs):
                kwargs.pop('num_logits_to_keep')
                return model(**kwargs)
        with pytest.raises(ValueError, match='exact native response-only'):
            opsd_loss(IgnoresSlice(), teacher, tensors, tok, [['Paris'], ['Rome']], settings,
                      tensors['loss_mask'].sum())
    finally:
        fixtures.TinyModels.tearDownClass()
