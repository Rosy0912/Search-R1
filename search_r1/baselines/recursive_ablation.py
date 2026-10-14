"""Isolated recursive-A planner ablations; executor and PPO stay unchanged.

The baseline path delegates to the frozen planner verbatim.  ``install`` is a
process-local bootstrap hook; it does not edit the published A source files.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import random

from .recursive_branch import Planner as OriginalPlanner, legal_positions


@dataclass(frozen=True)
class Options:
    variant: str = 'baseline'
    position: str = 'uniform'
    max_generation: int | None = None
    selection_seed: int = 42

    def __post_init__(self):
        if self.variant not in ('baseline', 'root_only', 'random_final'):
            raise ValueError('Unknown recursive ablation variant')
        if self.position not in ('uniform', 'early', 'late'):
            raise ValueError('Unknown branch position distribution')
        if (self.max_generation is not None and
                (type(self.max_generation) is not int or self.max_generation not in (1, 2, 4))):
            raise ValueError('max_generation must be null, 1, 2 or 4')
        if type(self.selection_seed) is not int:
            raise ValueError('selection_seed must be an integer')

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        return cls(**dict(value))


def selection_rng(seed, question_id, outer):
    """Separate question/outer stream, unaffected by UUIDs or generation RNG."""
    if question_id is None or type(outer) is not int or outer < 0:
        raise ValueError('Random-final selection requires a question ID and outer')
    payload = json.dumps(['recursive-final-v1', seed, str(question_id), outer],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), 'big'))


def position_distribution(positions, mode):
    """Triangular legal-position rank weights, frozen before any experiment.

    Early gives ordered positions weights N,N-1,...,1; late gives 1,...,N.
    For uniform, the caller retains the original ``rng.choice`` call exactly.
    """
    if not positions:
        raise ValueError('Cannot choose from no legal branch positions')
    if mode == 'uniform':
        weights = [1] * len(positions)
    elif mode == 'early':
        weights = list(range(len(positions), 0, -1))
    elif mode == 'late':
        weights = list(range(1, len(positions) + 1))
    else:
        raise ValueError('Unknown branch position distribution')
    total = sum(weights)
    return weights, [weight / total for weight in weights]


class Planner(OriginalPlanner):
    """Drop-in constructor for the action-pipelined recursive executor."""
    def __init__(self, initial, rng, emit, target=5, budget=10, *, options=None,
                 question_id=None, outer=None):
        self.options = Options.parse(options)
        if initial.generation != 0 or initial.parent_id is not None:
            raise ValueError('Ablation requires the actual original root')
        if target < 1 or budget < 0:
            raise ValueError('Invalid target or continuation budget')
        self.initial = initial
        self.final_rng = (selection_rng(self.options.selection_seed, question_id, outer)
                          if self.options.variant == 'random_final' else None)
        self.original_final_ids = None
        self._random_final_result = None
        super().__init__(initial, rng, emit, target=target, budget=budget)

    def next_request(self):
        # Crucial: C2 and matching baseline controls use exactly the original
        # candidate tree, original RNG calls, legal boundary and backoff.
        if (self.options.variant != 'root_only' and
                self.options.max_generation is None and self.options.position == 'uniform'):
            return super().next_request()
        if self.pending is not None:
            raise RuntimeError('Previous request has not completed')
        if self.reason or len(self.accepted) == self.target or self.attempts == self.budget:
            self.reason = self.reason or ('target_reached' if len(self.accepted) == self.target
                                         else 'budget_exhausted')
            self.end_search(self.reason)
            return None
        if not self.backoff:
            history = self.accepted + self.pool
            parents = ([self.initial] if self.options.variant == 'root_only'
                       else self.accepted)
            if self.options.max_generation is not None:
                parents = [r for r in parents if r.generation < self.options.max_generation]
            eligible = [(r, legal_positions(r, history)) for r in parents]
            eligible = [(r, positions) for r, positions in eligible if positions[0]]
            if not eligible:
                self.reason = 'no_expandable_record'
                return None
            self.parent, (positions, self.boundary, lcps) = self.rng.choice(eligible)
            weights, probabilities = position_distribution(positions, self.options.position)
            self.start = (self.rng.choice(positions) if self.options.position == 'uniform'
                          else self.rng.choices(positions, weights=weights, k=1)[0])
            self.backoff = list(reversed([p for p in positions if p <= self.start]))
            self.search_open = True
            self.emit('search_start', eligible=[dict(id=r.trajectory_id, generation=r.generation,
                      weight=1, probability=1 / len(eligible)) for r, _ in eligible],
                      parent_id=self.parent.trajectory_id, lcps=lcps, boundary=self.boundary,
                      legal_positions=positions, start=self.start,
                      position_distribution=self.options.position,
                      position_probabilities=probabilities,
                      max_generation=self.options.max_generation,
                      ablation_variant=self.options.variant)
        index = self.backoff.pop(0)
        self.attempts += 1
        self.pending = self.parent, index, self.attempts
        self.emit('branch_start', attempt=self.attempts, parent_id=self.parent.trajectory_id,
                  action_index=index, boundary=self.boundary, backoff=index != self.start)
        return self.pending

    def finish(self):
        if self.options.variant != 'random_final':
            return super().finish()
        if self._random_final_result is not None:
            return self._random_final_result
        emit = self.emit

        def original_emit(event, **fields):
            # Preserve original fill behavior and RNG consumption, while the
            # final question_end describes the group actually sent to PPO.
            emit('original_selection_end' if event == 'question_end' else event, **fields)

        self.emit = original_emit
        try:
            original = super().finish()
        finally:
            self.emit = emit
        self.original_final_ids = [r.trajectory_id for r in original]
        candidates = [r for r in self.accepted + self.pool if r is not self.initial]
        ids = [r.trajectory_id for r in candidates]
        if len(set(ids)) != len(ids) or self.initial.trajectory_id in ids:
            raise ValueError('Final candidate trajectory identities are not unique')
        count = min(self.target - 1, len(candidates))
        # Selecting all when short still uses uniform without-replacement
        # ordering; no model sampling, fabricated ID or extra attempt is added.
        chosen = self.final_rng.sample(candidates, count)
        result = [self.initial] + chosen
        emit('final_selection', selection='uniform_without_replacement',
             original_final_ids=self.original_final_ids, candidate_ids=ids,
             selected_nonroot_ids=[r.trajectory_id for r in chosen],
             marginal_probability=(count / len(candidates) if candidates else 0.),
             selection_rng_protocol='sha256_seed_question_outer_v1')
        emit('question_end', final_ids=[r.trajectory_id for r in result], attempts=self.attempts,
             accepted_branches=len(self.accepted) - 1, same_reward_candidates=len(self.pool),
             fill_count=sum(r.source == 'failure_fill' for r in result),
             complete_group=len(result) == self.target, stop_reason=self.reason,
             ablation_variant='random_final')
        self._random_final_result = result
        return result


_rollout_context = ContextVar('recursive_ablation_question_outer', default=None)
_installed_options = None


def install(options):
    """Install in driver and Ray workers before SearchRollout execution.

    ``tree.continuation_budget_per_question`` remains the budget authority.
    Only planner creation gets explicit original question/outer identity; the
    production batching loop, snapshots, state transitions and PPO are reused.
    """
    global _installed_options
    options = Options.parse(options)
    if _installed_options is not None:
        if _installed_options != options:
            raise RuntimeError('Cannot change ablation settings in a live process')
        return
    from . import recursive_branch, rollout
    original_method = rollout.SearchRollout._recursive_branches

    class ConfiguredPlanner(Planner):
        def __init__(self, initial, rng, emit, target=5, budget=10):
            context = _rollout_context.get()
            if context is None:
                raise RuntimeError('Ablation planner created outside identified rollout context')
            question_ids, outer = context
            source = initial.state.source
            if type(source) is not int or not 0 <= source < len(question_ids):
                raise ValueError('Invalid question source index')
            super().__init__(initial, rng, emit, target, budget, options=options,
                             question_id=question_ids[source], outer=outer)

    @wraps(original_method)
    def identified_rollout(self, roots, snapshots, tree, meta):
        if len(self.question_uids) != len(roots):
            raise ValueError('Ablation question IDs do not cover actual root rows')
        token = _rollout_context.set((self.question_uids, int(self.rollout_outer)))
        try:
            return original_method(self, roots, snapshots, tree, meta)
        finally:
            _rollout_context.reset(token)

    recursive_branch.Planner = ConfiguredPlanner
    rollout.SearchRollout._recursive_branches = identified_rollout
    _installed_options = options
