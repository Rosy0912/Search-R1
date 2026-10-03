"""Frozen teacher placement/precision and bounded timing without a CUDA runtime."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import Qwen2ForCausalLM
from tests.baselines import test_models_and_driver as fixtures
from search_r1.baselines.distillation import FrozenTeacher, opsd_loss


def test_cpu_teacher_outer_autocast_does_not_change_fp32_distribution():
    fixtures.TinyModels.setUpClass()
    try:
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, fixtures.TinyModels.tokenizer)
        prompt = torch.tensor([fixtures.TinyModels.tokenizer.encode('Question capital?')])
        response = torch.tensor([fixtures.TinyModels.tokenizer.encode('<answer>Paris</answer>')])
        attention = torch.ones_like(response)
        expected = teacher.log_probs(prompt, response, attention, targets=[['Paris']], chunk=3)
        with teacher.on_device(), torch.autocast('cpu', dtype=torch.bfloat16):
            actual = teacher.log_probs(prompt, response, attention, targets=[['Paris']], chunk=3)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert teacher.device.type == 'cpu'
        assert all(p.dtype == torch.float32 and not p.requires_grad and p.grad is None
                   for p in teacher.model.parameters())
        assert teacher.cost_metrics()['teacher/cuda_stream_seconds_cumulative_per_rank'] == 0
    finally:
        fixtures.TinyModels.tearDownClass()


def test_cpu_context_preserves_loss_and_every_student_gradient():
    fixtures.TinyModels.setUpClass()
    try:
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, fixtures.TinyModels.tokenizer)
        prompt = torch.tensor([fixtures.TinyModels.tokenizer.encode('Question capital?')])
        response = torch.tensor([fixtures.TinyModels.tokenizer.encode('<answer>Paris</answer>')])
        ids = torch.cat((prompt, response), 1)
        tensors = dict(prompts=prompt, responses=response, input_ids=ids, attention_mask=torch.ones_like(ids),
                       position_ids=torch.arange(ids.shape[1])[None], loss_mask=torch.ones_like(response))
        settings = SimpleNamespace(chunk_size=3, beta=0., temperature=1., component_clip=.05)
        losses, gradients = [], []
        for context in (nullcontext(), teacher.on_device()):
            student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
            with context:
                loss = opsd_loss(student, teacher, tensors, fixtures.TinyModels.tokenizer,
                                 [['Paris']], settings, response.numel())
                loss.backward()
            losses.append(loss.detach())
            gradients.append({name: p.grad.clone() for name, p in student.named_parameters()})
        torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=0, rtol=0)
    finally:
        fixtures.TinyModels.tearDownClass()


def test_cuda_context_offloads_on_exception_and_events_are_bounded():
    teacher = FrozenTeacher.__new__(FrozenTeacher)
    teacher.device, teacher.execution_device = torch.device('cpu'), torch.device('cuda:0')
    moves = []
    teacher.model = SimpleNamespace(to=lambda device: moves.append(str(device)))
    teacher._validate_execution = lambda device: None
    teacher.cuda_update_entries = 0
    teacher.transfer_wall_seconds = teacher.cuda_seconds = teacher.wall_seconds = teacher.cpu_seconds = 0.
    teacher._cuda_events = []
    class Event:
        def record(self, stream):
            assert stream == 'owned-stream'
        def synchronize(self):
            pass
        def elapsed_time(self, end):
            return 2.
    with patch('torch.cuda.synchronize') as synchronize, \
         patch('torch.cuda.device', return_value=nullcontext()), \
         patch('torch.cuda.current_stream', return_value='owned-stream'), \
         patch('torch.cuda.Event', side_effect=lambda **kw: Event()):
        try:
            with teacher.on_device():
                assert teacher.device == teacher.execution_device
                for _ in range(1025):
                    with teacher.measure():
                        pass
                assert len(teacher._cuda_events) == 1
                assert abs(teacher.cuda_seconds - 2.048) < 1e-12
                raise RuntimeError('update failed')
        except RuntimeError as error:
            assert str(error) == 'update failed'
        assert synchronize.call_count == 2
    assert moves == ['cuda:0', 'cpu']
    assert teacher.device.type == 'cpu'
    assert not teacher._cuda_events
    assert abs(teacher.cuda_seconds - 2.05) < 1e-12
    assert teacher.transfer_wall_seconds > 0


def test_fixed_microbatch_accumulation_preserves_loss_and_all_gradients():
    fixtures.TinyModels.setUpClass()
    try:
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, fixtures.TinyModels.tokenizer)
        prompt = torch.tensor([fixtures.TinyModels.tokenizer.encode('Question capital?')]).repeat(2, 1)
        response = torch.tensor([fixtures.TinyModels.tokenizer.encode('<answer>Paris</answer>')]).repeat(2, 1)
        ids = torch.cat((prompt, response), 1)
        loss_mask = torch.ones_like(response)
        loss_mask[1, ::2] = 0  # Observations stay in the context but get no loss.
        tensors = dict(prompts=prompt, responses=response, input_ids=ids, attention_mask=torch.ones_like(ids),
                       position_ids=torch.arange(ids.shape[1])[None].repeat(2, 1), loss_mask=loss_mask)
        settings = SimpleNamespace(chunk_size=3, beta=0., temperature=1., component_clip=.05)
        losses, gradients = [], []
        for micro_size in (1, 2):
            student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
            total = torch.zeros(())
            for start in range(0, 2, micro_size):
                micro = {key: value[start:start + micro_size] for key, value in tensors.items()}
                loss = opsd_loss(student, teacher, micro, fixtures.TinyModels.tokenizer,
                                 [['Paris']] * micro_size, settings, loss_mask.sum())
                loss.backward()
                total += loss.detach()
            losses.append(total)
            gradients.append({name: p.grad.clone() for name, p in student.named_parameters()})
        torch.testing.assert_close(losses[0], losses[1], atol=1e-7, rtol=1e-5)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=1e-7, rtol=1e-5)
        assert all(p.grad is None for p in teacher.model.parameters())
    finally:
        fixtures.TinyModels.tearDownClass()


def test_cuda_timing_failure_still_attempts_offload_without_swallowing_error():
    teacher = FrozenTeacher.__new__(FrozenTeacher)
    teacher.device, teacher.execution_device = torch.device('cpu'), torch.device('cuda:0')
    moves = []
    teacher.model = SimpleNamespace(to=lambda device: moves.append(str(device)))
    teacher._validate_execution = lambda device: None
    teacher.cuda_update_entries = 0
    teacher.transfer_wall_seconds = 0.
    with patch('torch.cuda.synchronize'), \
         patch.object(teacher, '_flush_cuda_events', side_effect=RuntimeError('event timing failed')):
        try:
            with teacher.on_device():
                pass
        except RuntimeError as error:
            assert str(error) == 'event timing failed'
        else:
            raise AssertionError('timing failure was swallowed')
    assert moves == ['cuda:0', 'cpu']
    assert teacher.device.type == 'cpu'
    assert teacher.transfer_wall_seconds > 0


def test_actual_teacher_precision_and_attention_audit_rejects_drift():
    fixtures.TinyModels.setUpClass()
    try:
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, fixtures.TinyModels.tokenizer)
        student = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(student.parameters())
        metrics = teacher.cost_metrics()
        assert metrics['teacher/storage_is_cpu_per_rank'] == 1
        assert metrics['teacher/all_parameter_dtype_is_fp32_per_rank'] == 1
        assert metrics['teacher/attention_is_eager_per_rank'] == 1
        assert metrics['teacher/cuda_update_entries_cumulative_per_rank'] == 0
        teacher.model.to(torch.bfloat16)
        assert teacher.cost_metrics()['teacher/all_parameter_dtype_is_fp32_per_rank'] == 0
        try:
            teacher.audit_metrics(optimizer)
        except RuntimeError as error:
            assert 'FP32' in str(error)
        else:
            raise AssertionError('BF16 teacher was accepted')
        teacher.model.float()
        teacher.model.config._attn_implementation = 'sdpa'
        assert teacher.cost_metrics()['teacher/attention_is_eager_per_rank'] == 0
        try:
            teacher.audit_metrics(optimizer)
        except RuntimeError as error:
            assert 'eager' in str(error)
        else:
            raise AssertionError('SDPA teacher was accepted')
    finally:
        fixtures.TinyModels.tearDownClass()


def test_actual_teacher_device_audit_rejects_false_cpu_storage():
    fixtures.TinyModels.setUpClass()
    try:
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, fixtures.TinyModels.tokenizer)
        teacher.model.to('meta')  # Real parameter device mismatch without CUDA.
        assert teacher.cost_metrics()['teacher/storage_is_cpu_per_rank'] == 0
        try:
            with teacher.on_device():
                raise AssertionError('CPU storage claim was accepted')
        except RuntimeError as error:
            assert 'actual device' in str(error)
    finally:
        fixtures.TinyModels.tearDownClass()


def test_cuda_context_rejects_unsuccessful_move_before_counting_entry():
    fixtures.TinyModels.setUpClass()
    try:
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, fixtures.TinyModels.tokenizer)
        teacher.execution_device = torch.device('cuda:0')
        # Simulate a no-op transfer while leaving real FP32 parameters on CPU.
        with patch.object(teacher.model, 'to', return_value=teacher.model) as move, \
             patch('torch.cuda.synchronize'):
            try:
                with teacher.on_device():
                    raise AssertionError('No-op CUDA transfer was accepted')
            except RuntimeError as error:
                assert 'actual device' in str(error)
            assert move.call_count == 2
        assert teacher.cuda_update_entries == 0
        assert teacher.device.type == 'cpu'
        assert teacher.cost_metrics()['teacher/storage_is_cpu_per_rank'] == 1
    finally:
        fixtures.TinyModels.tearDownClass()
