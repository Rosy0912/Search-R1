"""UNEXECUTED CPU fixtures for fixed matched phases and unknown failure handling."""
from types import SimpleNamespace
import pytest
import torch
from calibration_boundary_v2 import matched_stage, MatchedCalibrationRefusal


class MatchedRanks:
    ReduceOp = SimpleNamespace(MAX='MAX')
    def __init__(self, failed):
        self.failed, self.calls = failed, []
    def all_reduce(self, flag, op):
        self.calls.append(op)
        flag.fill_(int(self.failed))
    def get_world_size(self):
        return 2
    def all_gather(self, rows, payload):
        self.calls.append('GATHER')
        for row in rows:
            row.copy_(payload)


def test_one_rank_local_refusal_is_seen_by_each_rank_before_next_forward(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    for local_errors in ([], ['injected observer copy failure']):
        distributed = MatchedRanks(True)
        with pytest.raises(MatchedCalibrationRefusal, match='generate/after_original_before_repeat'):
            matched_stage('generate/after_original_before_repeat', local_errors, distributed=distributed)
        assert distributed.calls == ['MAX', 'GATHER']


def test_unknown_stage_collective_failure_starts_no_second_collective(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    failure = RuntimeError('unknown collective failure')
    distributed = MatchedRanks(False)
    def fail(flag, op):
        distributed.calls.append(op)
        raise failure
    distributed.all_reduce = fail
    with pytest.raises(RuntimeError) as raised:
        matched_stage('known-local-stage', ['known local problem'], distributed=distributed)
    assert raised.value is failure and distributed.calls == ['MAX']
