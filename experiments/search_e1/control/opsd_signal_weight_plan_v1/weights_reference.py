"""CPU reference for the proposed four 25-outer weighting ablations.

This module does not load models, sample trajectories, or launch training.
None means missing answer payload; it never means a zero log probability.
Production scoring and the four-rank controller are separate planned work.
"""
import math

VERSION = 'opsd-five-sibling-centered-signal-weights-v1'
AMPLITUDE = 0.9
NUMERICAL_FLOOR = 1e-6


def weights(scores, direction, *, neutral_reason=None):
    """Return five detached scalar weights; each direction uses the same scores.

    Inputs are current-preupdate HF replay scalars, never generation scores.
    A group-level encoding failure must supply neutral_reason for ALL siblings.
    Nonfinite inputs always raise, even when neutral fallback was requested.
    """
    scores = list(scores)
    if len(scores) != 5 or type(direction) is not int or direction not in (-1, 1):
        raise ValueError('Exactly five sibling scores and integer direction +/-1 required')
    if neutral_reason is not None and (not isinstance(neutral_reason, str) or not neutral_reason):
        raise ValueError('Neutral fallback requires an explicit nonempty reason')
    for x in scores:
        if x is not None and (type(x) not in (float, int) or not math.isfinite(x)):
            raise ValueError('A score must be finite numeric or explicit None; never hide NaN')
    valid = [j for j, x in enumerate(scores) if x is not None]
    centered = [0.0] * 5
    scale = None
    mean = None
    raw_maxabs = None
    if neutral_reason is None and len(valid) >= 2:
        # Center differences from an anchor to avoid a large common offset
        # amplifying floating-point error in the group mean.
        anchor = scores[valid[0]]
        delta = [scores[j] - anchor for j in valid]
        if not all(math.isfinite(x) for x in delta):
            raise ValueError('Finite scores overflowed during subtraction')
        mean_delta = math.fsum(delta) / len(valid)
        mean = anchor + mean_delta
        for j, x in zip(valid, delta):
            centered[j] = x - mean_delta
        raw_maxabs = max(abs(x) for x in centered)
        scale = max(NUMERICAL_FLOOR, raw_maxabs)
    elif neutral_reason is None:
        neutral_reason = 'fewer_than_two_available_scores'
    result = [1.0 + direction * AMPLITUDE * x / scale if scale is not None else 1.0
              for x in centered]
    if (not all(math.isfinite(x) and 0.1 - 1e-12 <= x <= 1.9 + 1e-12 for x in result)
            or abs(math.fsum(result) - 5.0) > 1e-10):
        raise ArithmeticError('Weight range/group-mean invariant failed')
    return dict(version=VERSION, direction=direction, amplitude=AMPLITUDE,
                numerical_floor=NUMERICAL_FLOOR, scores=scores,
                available_siblings=valid, centered_scores=centered,
                available_score_mean=mean, raw_maxabs=raw_maxabs,
                normalization_scale=scale, weights=result,
                neutral_reason=neutral_reason, group_mean=math.fsum(result) / 5.0)


def selected_batch_summary(selected_weights):
    """Describe actual selected-D scale; do not normalize the optimizer loss."""
    values = list(selected_weights)
    if not values or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in values):
        raise ValueError('A nonempty batch of finite positive weights is required')
    total = math.fsum(values)
    squares = math.fsum(w * w for w in values)
    return dict(actual_pairs=len(values), selected_weight_sum=total,
                selected_weight_sumsq=squares,
                selected_weight_mean=total / len(values),
                selected_weight_min=min(values), selected_weight_max=max(values),
                effective_sample_size=total * total / squares)
