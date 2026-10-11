"""Generation-bounded recursive search, including document experiment B.

The caller owns real executor snapshots, strict rewards, cost and durable events.
No text reconstruction or new rollout is performed by failure-pool filling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import hashlib
import time


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
    cost: dict = field(default_factory=dict)


def common_prefix(a, b):
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
    positions = [i for i in range(boundary, len(parent.snapshots))
                 if parent.snapshots[i] is not None
                 and not getattr(parent.snapshots[i], 'done', False)
                 and getattr(parent.snapshots[i], 'restorable', True)]
    return positions, boundary, lcps


def parent_probabilities(eligible, variant):
    if variant not in ('uniform', 'inverse_generation'):
        raise ValueError('Unknown recursive parent selection')
    if not eligible:
        raise ValueError('Cannot normalize an empty eligible set')
    if any(type(r.generation) is not int or r.generation < 0 for r, _ in eligible):
        raise ValueError('Generation must be a nonnegative integer')
    weights = [1.0 if variant == 'uniform' else 1.0 / (r.generation + 1)
               for r, _ in eligible]
    total = math.fsum(weights)
    return weights, [w / total for w in weights]


class Planner:
    """One question; requests from many planners can be generated together."""
    def __init__(self, initial, rng, emit, target=5, budget=10,
                 variant='inverse_generation'):
        if (type(target) is not int or target < 1 or type(budget) is not int
                or budget < 0 or initial.generation != 0 or initial.reward not in (0, 1)):
            raise ValueError('Invalid initial completed root or search budget')
        if variant not in ('uniform', 'inverse_generation'):
            raise ValueError('Unknown recursive parent selection')
        self.accepted, self.pool = [initial], []
        self.rng, self.emit, self.target, self.budget = rng, emit, target, budget
        self.variant = variant
        self.attempts, self.pending, self.backoff = 0, None, []
        self.reason, self.active_search, self.finished = None, None, None
        self.search_count, self.errors = 0, 0
        emit('question_start', root_id=initial.root_id, initial_reward=initial.reward,
             target=target, budget=budget, variant=variant, cost=initial.cost)

    def _close_search(self, reason):
        if self.active_search is not None:
            self.emit('search_end', search_id=self.active_search,
                      attempts=self.attempts-self.search_started_attempt,
                      total_attempts=self.attempts, accepted=len(self.accepted),
                      pool=len(self.pool), reason=reason,
                      same_count=len(self.pool)-self.search_started_pool,
                      accepted_id=self.accepted[-1].trajectory_id if len(self.accepted)>self.search_started_accepted else None)
            self.active_search = None
        self.backoff = []

    def end_search(self, reason):
        self._close_search(reason)

    def next_request(self):
        if self.pending is not None:
            raise RuntimeError('Previous request has not completed')
        if self.reason or len(self.accepted) == self.target or self.attempts == self.budget:
            self.reason = self.reason or ('target_reached' if len(self.accepted)==self.target
                                         else 'budget_exhausted')
            self._close_search(self.reason)
            return None
        if not self.backoff:
            history = self.accepted + self.pool
            eligible = [(r, legal_positions(r, history)) for r in self.accepted]
            eligible = [(r, p) for r, p in eligible if p[0]]
            if not eligible:
                self.reason = 'no_expandable_record'
                return None
            weights, probabilities = parent_probabilities(eligible, self.variant)
            selected = (self.rng.choice(eligible) if self.variant == 'uniform' else
                        self.rng.choices(eligible, weights=weights, k=1)[0])
            self.parent, (positions, self.boundary, lcps) = selected
            self.start = self.rng.choice(positions)
            self.backoff = list(reversed([p for p in positions if p <= self.start]))
            self.search_count += 1
            self.active_search = f'{self.parent.root_id}:search:{self.search_count}'
            self.search_started_attempt = self.attempts
            self.search_started_pool = len(self.pool)
            self.search_started_accepted = len(self.accepted)
            self.backoff_ordinal = 0
            self.emit('search_start', search_id=self.active_search,
                      eligible=[dict(id=r.trajectory_id, generation=r.generation,
                                     weight=w, probability=p)
                                for (r, _), w, p in zip(eligible, weights, probabilities)],
                      parent_id=self.parent.trajectory_id, lower_generation_ids=list(lcps), lcps=lcps, boundary=self.boundary,
                      action_count=len(self.parent.steps),
                      legal_positions=positions, start=self.start,
                      accepted=len(self.accepted), pool=len(self.pool), used_budget=self.attempts)
        index = self.backoff.pop(0)
        self.attempts += 1
        self.pending = self.parent, index, self.attempts
        self.backoff_ordinal += 1
        self.emit('branch_start', search_id=self.active_search, attempt=self.attempts,
                  attempt_id=f'{self.parent.root_id}:attempt:{self.attempts}',
                  started_at=time.time(), backoff_ordinal=self.backoff_ordinal,
                  parent_id=self.parent.trajectory_id, action_index=index,
                  boundary=self.boundary, backoff=index!=self.start,
                  used_budget_before=self.attempts-1)
        return self.pending

    def accept_result(self, child=None, error=None):
        if self.pending is None:
            raise RuntimeError('No pending branch')
        parent, index, attempt = self.pending
        self.pending = None
        if error is None and (child is None or child.parent_id != parent.trajectory_id
                or child.root_id != parent.root_id or child.generation != parent.generation+1
                or child.reward not in (0, 1) or child.steps[:index] != parent.steps[:index]
                or child.trajectory_id in {r.trajectory_id for r in self.accepted+self.pool}):
            error = ValueError('Continuation violates identity, generation, prefix or reward')
        if error is not None:
            self.errors += 1
            self.reason = 'execution_error'
            self.emit('branch_end', search_id=self.active_search, attempt=attempt,
                      status='error', parent_id=parent.trajectory_id, parent_generation=parent.generation,
                      ended_at=time.time(), error=repr(error))
            self._close_search('execution_error')
            return
        different = child.reward != parent.reward
        child.source = 'contrast_accept' if different else 'same_reward'
        (self.accepted if different else self.pool).append(child)
        self.emit('branch_end', search_id=self.active_search, attempt=attempt,
                  status='complete', id=child.trajectory_id, parent_id=parent.trajectory_id,
                  root_id=child.root_id, parent_reward=parent.reward, reward=child.reward,
                  generation=child.generation, parent_generation=parent.generation,
                  source=child.source, destination='L' if different else 'F',
                  ended_at=time.time(), cost=child.cost)
        if different:
            self._close_search('contrast_accept')
        elif self.attempts == self.budget:
            self.reason = 'budget_exhausted'
            self._close_search('budget_exhausted')
        elif not self.backoff:
            self._close_search('boundary_reached')

    def finish(self):
        if self.finished is not None:
            return self.finished
        if self.pending is not None or self.reason is None:
            raise RuntimeError('Search not terminal')
        rng_sha256=hashlib.sha256(repr(self.rng.getstate()).encode()).hexdigest()
        before=reward_statistics(self.accepted)
        fill = self.rng.sample(self.pool, min(self.target-len(self.accepted), len(self.pool)))
        for r in fill:
            r.source = 'failure_fill'
        result = self.accepted + fill
        self.emit('fill_end', pool_ids=[r.trajectory_id for r in self.pool],
                  selected_ids=[r.trajectory_id for r in fill], fill_count=len(fill),
                  accepted_before=len(self.accepted), shortfall=self.target-len(result),
                  rng_state_sha256=rng_sha256,
                  unselected_ids=[r.trajectory_id for r in self.pool if r not in fill])
        self.emit('question_end', final_ids=[r.trajectory_id for r in result],
                  rewards=[r.reward for r in result], generations=[r.generation for r in result],
                  sources=[r.source for r in result], attempts=self.attempts,
                  accepted_branches=len(self.accepted)-1, same_reward_candidates=len(self.pool),
                  error_attempts=self.errors, fill_count=len(fill),
                  complete_group=len(result)==self.target, stop_reason=self.reason,
                  before_fill_reward=before, after_fill_reward=reward_statistics(result),
                  fill_fraction=len(fill)/len(result),
                  same_fraction=len(self.pool)/(self.attempts-self.errors) if self.attempts>self.errors else None,
                  pool_use=len(fill)/len(self.pool) if self.pool else None,
                  total_cost={key:sum(r.cost.get(key,0) for r in self.accepted+self.pool)
                              for key in ("new_generated_tokens","retrieval_calls","model_calls")},
                  cost_incomplete=self.errors>0)
        assert self.attempts == len(self.accepted)-1+len(self.pool)+self.errors
        self.finished = result
        return result


def reward_statistics(records):
    rewards=[r.reward for r in records]
    p=sum(rewards)/len(rewards)
    entropy=-sum(x*math.log2(x) for x in (p,1-p) if x>0)
    return dict(rewards=rewards, count=len(rewards), correct=sum(rewards), mean=p,
                mixed=0<p<1, population_variance=p*(1-p), entropy_bits=entropy,
                zero_advantage=not 0<p<1)


def search(initial, continue_from, rng, emit, target=5, budget=10,
           variant='inverse_generation'):
    planner = Planner(initial, rng, emit, target, budget, variant)
    while (request := planner.next_request()) is not None:
        try:
            child = continue_from(*request)
        except Exception as exc:
            planner.accept_result(error=exc)
        else:
            planner.accept_result(child)
    return planner.finish(), planner.attempts, planner.reason
