"""Fresh0 review pause must stay unchanged; no old25 restoration test repeated."""
import copy
import pytest
import torch
from calibration_state_v1 import validate_pause_workers, assert_complete_pause_unchanged
from test_calibration_contract_v1 import fixture_workers


def test_zero_scheduler_pause_accepts_and_old25_refuses():
    workers=fixture_workers(); validate_pause_workers(workers)
    workers[2]['scheduler_step']=25
    with pytest.raises(ValueError): validate_pause_workers(workers)


def test_after_real_sampling_driver_cursor512_is_bound_without_rewind():
    workers=fixture_workers()
    driver=dict(pid=42,global_steps=1,actual_outer_updates=0,sampler=dict(cursor=512),rng=torch.arange(8))
    assert_complete_pause_unchanged(workers,copy.deepcopy(workers),driver,copy.deepcopy(driver))
    changed=copy.deepcopy(driver);changed['sampler']['cursor']=0
    with pytest.raises(ValueError): assert_complete_pause_unchanged(workers,workers,driver,changed)
