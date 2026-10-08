"""Real CPU parameter/buffer/RNG restoration and corrupt-snapshot counterexamples."""
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('paired_local', Path(__file__).with_name('paired_gate.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.5))
    model.register_buffer('nonpersistent_state', torch.arange(4.), persistent=False)
    model.train(); model[1].eval()
    teacher_model = torch.nn.Linear(3, 2).eval().requires_grad_(False)
    teacher_model.register_buffer('rotary_state', torch.arange(7.), persistent=False)
    teacher = SimpleNamespace(model=teacher_model, device=torch.device('cpu'), _cuda_events=[])
    for key in gate.TEACHER_COUNTERS: setattr(teacher, key, 0)
    named = list(model.named_parameters())
    initial = {name: param.detach().clone() for name, param in named}
    for index, (name, param) in enumerate(named):
        gate.snapshot('baseline', tmp_path, f'initial-{index}', dict(name=name, master=param))
    values = dict(named=named, directory=tmp_path, model=model,
                  student_buffers=gate.saved_buffers(model), teacher=teacher,
                  teacher_buffers=gate.saved_buffers(teacher_model),
                  teacher_counters={key: 0 for key in gate.TEACHER_COUNTERS},
                  teacher_digest=gate.teacher_parameter_digest(teacher),
                  rng=gate.capture_rng(torch.device('cpu')), device=torch.device('cpu'),
                  decoder_counts={'0':4}, model_modes={name: child.training for name, child in model.named_modules()})
    return values, initial


def test_full_restore_after_actual_two_adam_updates_buffers_rng_modes_and_counters(tmp_path):
    values, initial = fixture(tmp_path)
    model, teacher = values['model'], values['teacher']
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).square().sum().backward()
        optimizer.step()
    assert optimizer.state and all(p.grad is not None for p in model.parameters())
    assert any(not torch.equal(initial[name], p) for name, p in model.named_parameters())
    model.nonpersistent_state.add_(3)
    teacher.model.rotary_state.add_(2)
    for key in gate.TEACHER_COUNTERS: setattr(teacher, key, 99)
    model.eval(); teacher.model.train()
    torch.manual_seed(987); random.seed(765); np.random.seed(543)
    optimizer.zero_grad(set_to_none=True); optimizer.state.clear(); del optimizer
    restored = gate.restore_initial(**values)
    assert restored['master_elements_including_padding'] == sum(p.numel() for p in model.parameters())
    for name, param in model.named_parameters():
        assert torch.equal(param, initial[name]) and param.grad is None
    assert torch.equal(model.nonpersistent_state, values['student_buffers']['nonpersistent_state'])
    assert torch.equal(teacher.model.rotary_state, values['teacher_buffers']['rotary_state'])
    assert restored['rng_restored']['python'] and restored['rng_restored']['numpy']
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
    assert not optimizer.state
    assert all(count == 0 for count in values['decoder_counts'].values())


def test_missing_later_parameter_snapshot_cannot_silently_restore_first_only(tmp_path):
    values, _ = fixture(tmp_path)
    (tmp_path / 'initial-1.pt').unlink()
    with pytest.raises(FileNotFoundError): gate.restore_initial(**values)


def test_corrupt_later_snapshot_shape_rejected(tmp_path):
    values, _ = fixture(tmp_path)
    torch.save(dict(name=values['named'][1][0], master=torch.ones(123)), tmp_path / 'initial-1.pt')
    with pytest.raises(AssertionError): gate.restore_initial(**values)


def test_missing_nonpersistent_buffer_rejected(tmp_path):
    values, _ = fixture(tmp_path)
    del values['model']._buffers['nonpersistent_state']
    with pytest.raises(AssertionError): gate.restore_initial(**values)


def test_frozen_teacher_weight_change_rejected_before_restore(tmp_path):
    values, _ = fixture(tmp_path)
    with torch.no_grad(): next(values['teacher'].model.parameters()).flatten()[-1].add_(1)
    with pytest.raises(AssertionError): gate.restore_initial(**values)


def test_clip_observer_passes_original_set_once_without_reordering_or_changing_gradients():
    import importlib
    from unittest.mock import patch
    module = importlib.import_module('torch.distributed.fsdp.fully_sharded_data_parallel')
    named = [('first', torch.nn.Parameter(torch.zeros(4))), ('last', torch.nn.Parameter(torch.zeros(7)))]
    params = {p for _, p in named}
    initial_grads = [torch.arange(p.numel(), dtype=torch.float32) + 2 for _, p in named]
    for (_, p), grad in zip(named, initial_grads): p.grad = grad.clone()
    original_calls = []
    def original(parameters, norm_type):
        assert parameters is params
        original_calls.append([id(p) for p in parameters])
        return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad, norm_type) for p in parameters]), norm_type)
    class NativeClip:
        def clip_grad_norm_(self, maximum):
            result = module._get_grad_norm(params, 2.)
            coefficient = torch.clamp(maximum / (result + 1e-6), max=1.)
            for p in params: p.grad.mul_(coefficient)
            return result
    with patch.object(module, '_get_grad_norm', original):
        baseline = NativeClip().clip_grad_norm_(1.)
        baseline_grads = [p.grad.clone() for _, p in named]
        for (_, p), grad in zip(named, initial_grads): p.grad = grad.clone()
        records = []
        candidate = gate.observed_clip(NativeClip(), named, records)
        assert module._get_grad_norm is original
    assert len(original_calls) == 2 and original_calls[0] == original_calls[1]
    assert torch.equal(baseline, candidate)
    assert all(torch.equal(saved, p.grad) for saved, (_, p) in zip(baseline_grads, named))
    assert records[0]['parameter_object_ids'] == original_calls[1]
    assert records[0]['original_local_norm_fp32'] == float(candidate)


def test_clip_observer_restores_original_function_even_on_error():
    import importlib
    from unittest.mock import patch
    module = importlib.import_module('torch.distributed.fsdp.fully_sharded_data_parallel')
    param = torch.nn.Parameter(torch.zeros(2)); param.grad = torch.ones(2)
    def original(*_): raise RuntimeError('injected original clip failure')
    class NativeClip:
        def clip_grad_norm_(self, *_): return module._get_grad_norm({param}, 2.)
    with patch.object(module, '_get_grad_norm', original):
        with pytest.raises(RuntimeError, match='injected original clip failure'):
            gate.observed_clip(NativeClip(), [('all', param)], [])
        assert module._get_grad_norm is original
