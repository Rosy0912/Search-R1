"""CPU control checks only; real constructor-to-restore needs eight CUDA ranks."""
import pytest
import torch
from verl.utils.fsdp_utils import initialize_fsdp_before_full_restore


def test_non_fsdp_restore_rejected():
    with pytest.raises(TypeError, match='actual FSDP root'):
        initialize_fsdp_before_full_restore(torch.nn.Linear(2, 2))


def test_initialized_root_roundtrip_is_noop(monkeypatch):
    class Root:
        _is_root = True
    monkeypatch.setattr(torch.distributed.fsdp, 'FullyShardedDataParallel', Root)
    root = Root()
    before = dict(root.__dict__)
    assert initialize_fsdp_before_full_restore(root) == dict(initialized=False, reason='already_initialized')
    assert root.__dict__ == before


def test_nested_fsdp_restore_refuses_reinitialization(monkeypatch):
    class Child:
        _is_root = False
    monkeypatch.setattr(torch.distributed.fsdp, 'FullyShardedDataParallel', Child)
    with pytest.raises(ValueError, match='not a child'):
        initialize_fsdp_before_full_restore(Child())
