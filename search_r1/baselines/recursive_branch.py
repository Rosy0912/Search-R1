"""Document algorithm: uniform eligible parents, fixed-boundary backoff.

Executor callbacks must provide real pre-action snapshots and persist events.
This module neither rebuilds text nor reinitializes interaction budgets.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Record:
    trajectory_id: str
    parent_id: str | None
    root_id: str
    generation: int
    reward: int
    state: object
    snapshots: list
    steps: list
    source: str


def common_prefix(a, b):
    """Steps include the model action AND its corresponding observation."""
    n = 0
    for x, y in zip(a.steps, b.steps):
        if x != y:
            break
        n += 1
    return n


def legal_positions(parent, history):
    earlier = [r for r in history if r.generation < parent.generation]
    lcps = {r.trajectory_id: common_prefix(parent, r) for r in earlier}
    boundary = max(lcps.values(), default=0)
    return list(range(boundary, len(parent.snapshots))), boundary, lcps


def search(initial, continue_from, rng, emit, target=5, budget=10):
    """continue_from(parent, action_index, attempt) returns a complete Record.

    An exception consumes one attempt and terminates search before pool fill.
    No failed/incomplete execution becomes a fabricated reward-zero record.
    """
    if target < 1 or budget < 0 or initial.generation != 0:
        raise ValueError('Invalid root or search budget')
    accepted, pool = [initial], []
    attempts = 0
    reason = 'target_reached'
    emit('question_start', root_id=initial.root_id, target=target, budget=budget,
         initial_reward=initial.reward)
    while len(accepted) < target and attempts < budget:
        history = accepted + pool
        eligible = [(r, legal_positions(r, history)) for r in accepted]
        eligible = [(r, p) for r, p in eligible if p[0]]
        if not eligible:
            reason = 'no_expandable_record'
            break
        parent, (positions, boundary, lcps) = rng.choice(eligible)
        start = rng.choice(positions)
        emit('search_start', eligible=[dict(id=r.trajectory_id,
             generation=r.generation, probability=1/len(eligible)) for r, _ in eligible],
             parent_id=parent.trajectory_id, lcps=lcps, boundary=boundary,
             legal_positions=positions, start=start)
        error = False
        for index in reversed([p for p in positions if p <= start]):
            if attempts == budget:
                reason = 'budget_exhausted'
                break
            attempts += 1
            emit('branch_start', attempt=attempts, parent_id=parent.trajectory_id,
                 action_index=index, boundary=boundary, backoff=index != start)
            try:
                child = continue_from(parent, index, attempts)
                if (child.parent_id != parent.trajectory_id or
                    child.root_id != parent.root_id or
                    child.generation != parent.generation + 1 or
                    child.reward not in (0, 1) or
                    child.steps[:index] != parent.steps[:index] or
                    child.trajectory_id in {r.trajectory_id for r in accepted + pool}):
                    raise ValueError('Invalid continuation identity/prefix/reward')
            except Exception as exc:
                emit('branch_end', attempt=attempts, status='error', error=repr(exc))
                reason, error = 'execution_error', True
                break
            different = child.reward != parent.reward
            child.source = 'contrast_accept' if different else 'same_reward'
            (accepted if different else pool).append(child)
            emit('branch_end', attempt=attempts, status='complete',
                 id=child.trajectory_id, parent_reward=parent.reward,
                 reward=child.reward, generation=child.generation, source=child.source)
            if different:
                break
        emit('search_end', accepted=len(accepted), pool=len(pool), attempts=attempts)
        if error:
            break
    if len(accepted) < target and attempts == budget and reason != 'execution_error':
        reason = 'budget_exhausted'
    fill = rng.sample(pool, min(target-len(accepted), len(pool)))
    for r in fill:
        r.source = 'failure_fill'
    final = accepted + fill
    emit('fill_end', pool_ids=[r.trajectory_id for r in pool],
         selected_ids=[r.trajectory_id for r in fill], fill_count=len(fill))
    emit('question_end', final_ids=[r.trajectory_id for r in final],
         attempts=attempts, accepted_branches=len(accepted)-1,
         same_reward_candidates=len(pool), fill_count=len(fill),
         complete_group=len(final)==target, stop_reason=reason)
    return final, attempts, reason
