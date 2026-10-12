"""Optional, manifest-bound reward-only ablation using completed logical steps."""

SCHEDULE = 'g1_through50_then_binary'
ACTOR_PROBE_SCHEDULE = 'g1_through100_then_binary'
SCHEDULES = {SCHEDULE: 50, ACTOR_PROBE_SCHEDULE: 100}


def reward_boundary(schedule):
    if schedule not in SCHEDULES:
        raise ValueError('Unsupported reward schedule')
    return SCHEDULES[schedule]


def binary_phase(schedule, logical_update):
    if schedule is None:
        return False
    boundary = reward_boundary(schedule)
    if type(logical_update) is not int or not 1 <= logical_update <= 200:
        raise ValueError('Reward schedule requires a formal logical update in 1..200')
    return logical_update > boundary


def scheduled_score(score, correct, schedule, logical_update):
    return float(bool(correct)) if binary_phase(schedule, logical_update) else score
