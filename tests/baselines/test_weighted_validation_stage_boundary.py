"""Actual NumPy validation aggregates cross the strict stage commit boundary."""
import numpy as np
import pytest

from search_r1.baselines.controller import weighted_validation
from search_r1.baselines.aligned_outer_commit import stage_evidence


def test_numpy_mean_weighted_score_commits_complete_stage():
    nq = np.mean([1.0] * 12 + [0.0] * 20)
    hotpot = np.mean([1.0] * 4 + [0.0] * 28)
    score = weighted_validation({
        'val/test_score/nq': nq, 'val/evaluated_count/nq': 32,
        'val/test_score/hotpot': hotpot, 'val/evaluated_count/hotpot': 32,
    })
    assert type(score) is float
    assert score == 16 / 64
    evidence = stage_evidence(1, 1, {
        'bapo/stage_validation': score, 'bapo/stage_evaluated_count': 64,
    })
    assert evidence['score'] == score and evidence['evaluated_count'] == 64


@pytest.mark.parametrize('values,count', [([float('nan')], 64), ([float('inf')], 64), ([0.5], 14)])
def test_native_conversion_does_not_relax_stage_checks(values, count):
    score = weighted_validation({
        'val/test_score/nq': np.mean(values), 'val/evaluated_count/nq': count,
    })
    assert type(score) is float
    with pytest.raises(ValueError, match='finite complete64'):
        stage_evidence(1, 1, {
            'bapo/stage_validation': score, 'bapo/stage_evaluated_count': count,
        })
