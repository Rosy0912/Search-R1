"""Independent tiny CPU math test; not native GPU or official-weight acceptance."""
import pytest
import torch
from transformers import Qwen2ForCausalLM
from search_r1.baselines.distillation import FrozenTeacher, opsd_loss
from tests.baselines import test_models_and_driver as fixtures
from tests.baselines.test_opsd_zero_initialization import tensors_and_settings


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('chunk', [1, 3])
def test_full_native_head_teacher4_vs2_all_gradients(checkpoint, chunk):
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        tensors, settings = tensors_and_settings(tok, chunk)
        tensors = {k: v.repeat(2, 1) for k, v in tensors.items()}
        # Differing context/response padding and interspersed observation masks.
        tensors['responses'][3, -1] = tok.pad_token_id
        tensors['input_ids'][3, -1] = tok.pad_token_id
        tensors['attention_mask'][3, -1] = 0
        tensors['loss_mask'][3, -1] = 0
        settings.component_clip = None
        settings.response_only_logits = False
        settings.loss_checkpointing = checkpoint
        outputs = []
        for micro in (4, 2):
            settings.teacher_forward_micro_batch = micro
            student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
            forward_kwargs, head_shapes = [], []
            a = student.register_forward_pre_hook(lambda _, args, kwargs: forward_kwargs.append(dict(kwargs)), with_kwargs=True)
            b = student.lm_head.register_forward_pre_hook(lambda _, args: head_shapes.append(tuple(args[0].shape)))
            teacher_shapes = []
            c = teacher.model.base_model.register_forward_pre_hook(lambda _, args, kwargs: teacher_shapes.append(tuple(kwargs['input_ids'].shape)), with_kwargs=True)
            loss = opsd_loss(student, teacher, tensors, tok, [['Paris'], ['Rome']] * 2,
                             settings, tensors['loss_mask'].sum())
            loss.backward()
            for handle in (a, b, c):
                handle.remove()
            assert len(forward_kwargs) == 1 and 'num_logits_to_keep' not in forward_kwargs[0]
            assert head_shapes == [(4, tensors['input_ids'].shape[1], student.config.hidden_size)]
            assert len(teacher_shapes) == 4 // micro and all(shape[0] == micro for shape in teacher_shapes)
            assert len({shape[1] for shape in teacher_shapes}) == 1
            assert torch.isfinite(loss) and loss >= -1e-6
            gradients = {name: p.grad.clone() for name, p in student.named_parameters()}
            assert all(torch.isfinite(x).all() for x in gradients.values())
            assert sum(float((x.double() ** 2).sum()) for x in gradients.values()) > 0
            outputs.append((loss.detach(), gradients, teacher_shapes[0][1]))
        assert outputs[0][2] == outputs[1][2]
        torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1e-7, rtol=1e-6)
        assert outputs[0][1].keys() == outputs[1][1].keys()
        for name in outputs[0][1]:
            torch.testing.assert_close(outputs[0][1][name], outputs[1][1][name], atol=1e-7, rtol=1e-5)
        assert all(p.grad is None for p in teacher.model.parameters())
        assert not torch.cuda.is_initialized()
    finally:
        fixtures.TinyModels.tearDownClass()
