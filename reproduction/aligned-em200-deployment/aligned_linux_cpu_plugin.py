"""Two explicit CPU-fixture adaptations; never imported by production training."""
from pathlib import Path
import sys,types
import pytest
REPO=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/repo')
def pytest_configure(config):
    # Bind the test-only namespace to actual reviewed files, bypassing an installed
    # unrelated regular 'tests' package. No production module is replaced.
    for name,path in [('tests',REPO/'tests'),('tests.baselines',REPO/'tests/baselines')]:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
@pytest.fixture(autouse=True)
def explicit_cpu_oracle_backend(request,monkeypatch):
    if request.node.nodeid=='tests/baselines/test_raw_sampler_diagnostic.py::test_real_actor_forward_recomputes_raw_selected_token_probabilities':
        import torch
        import verl.utils.torch_functional as functional
        assert not torch.cuda.is_available(),'This fixture adaptation is CPU-only'
        monkeypatch.setattr(functional,'FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE',False)
