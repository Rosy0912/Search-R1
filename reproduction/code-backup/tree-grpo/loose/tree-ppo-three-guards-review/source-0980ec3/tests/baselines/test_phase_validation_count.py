"""64 phase questions are fixed; complete teacher selection holdout remains intact."""
import pandas as pd
import pytest
from search_r1.baselines.controller import phase_validation_frame


def test_phase_subset_does_not_reduce_teacher_holdout():
    full = pd.DataFrame({'question': range(3393)})
    before = full.copy(deep=True)
    selected = phase_validation_frame(full, 64, 42)
    pd.testing.assert_frame_equal(selected, phase_validation_frame(full, 64, 42))
    pd.testing.assert_frame_equal(full, before)
    assert len(selected) == selected.question.nunique() == 64
    assert set(selected.question) <= set(full.question)
    assert not selected.equals(phase_validation_frame(full, 64, 43))
    assert len(phase_validation_frame(full.head(2), 64, 42)) == 2
    with pytest.raises(ValueError):
        phase_validation_frame(full, 0, 42)
