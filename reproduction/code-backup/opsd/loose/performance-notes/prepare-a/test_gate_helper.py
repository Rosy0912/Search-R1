"""CPU tests of gate comparators only, not a substitute for actual FSDP gate."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location('external_gate_helper', Path(__file__).with_name('proposal_a_fsdp_gate.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_exact_comparator_checks_interior_elements_beyond_one_chunk():
    baseline = torch.arange(1_048_581, dtype=torch.float32)
    candidate = baseline.clone()
    result = gate.compare_tensor(baseline, candidate)
    assert result == dict(elements=baseline.numel(), max_abs=0., l2=0.)
    candidate[1_048_579] += 1
    with pytest.raises(AssertionError, match='Exact tensor comparison failed'):
        gate.compare_tensor(baseline, candidate)


def test_snapshot_detects_optimizer_state_only_change(tmp_path):
    initial = dict(name='all-local-shard', master=torch.arange(8, dtype=torch.float32),
                   step=torch.tensor(2.), exp_avg=torch.ones(8), exp_avg_sq=torch.full((8,), .1))
    gate.snapshot('baseline', tmp_path, 'step2', initial)
    exact = gate.snapshot('proposal', tmp_path, 'step2', initial)
    assert exact['elements'] == 25 and exact['max_abs'] == exact['l2'] == 0
    initial['exp_avg_sq'][3] += .0001
    with pytest.raises(AssertionError, match='Exact tensor comparison failed'):
        gate.snapshot('proposal', tmp_path, 'step2', initial)


def test_comparator_rejects_nonfinite_and_missing_gradient():
    with pytest.raises(AssertionError):
        gate.compare_tensor(torch.ones(2), torch.tensor([1., float('nan')]))
    with pytest.raises(AssertionError):
        gate.compare_tensor(torch.ones(2), None)
    assert gate.compare_tensor(None, None)['elements'] == 0
