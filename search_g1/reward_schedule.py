"""Optional, manifest-bound reward-only ablation using completed logical steps."""

SCHEDULE = 'g1_through50_then_binary'


def binary_phase(schedule, logical_update):
    if schedule is None:
        return False
    if schedule != SCHEDULE:
        raise ValueError('Unsupported reward schedule')
    if type(logical_update) is not int or not 1 <= logical_update <= 200:
        raise ValueError('Reward schedule requires a formal logical update in 1..200')
    return logical_update > 50


def scheduled_score(score, correct, schedule, logical_update):
    return float(bool(correct)) if binary_phase(schedule, logical_update) else score
