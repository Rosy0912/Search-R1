"""Reward-selected fixed-data token PPO surrogate; no selection-unbiased claim.

Only sampling/provenance changes. Advantages, per-leaf 1/T and global reduction
remain the original Tree-GRPO objective. Failed candidates only consume cost.
"""
import copy
import time
import uuid
import numpy as np
from .tree_fresh200 import tree_option, leaf_quotas
from .tree_audit import fingerprint, state_signature

PROTOCOL = 'random_nested_backtrack_v1'
OBJECTIVE = 'reward_selected_nested_tree_grpo_surrogate_v1'


def validate_nested_options(tree):
    if tree.roots != 2 or leaf_quotas(tree) != (2, 3):
        raise ValueError('Nested protocol requires two roots with 2/3 leaves')
    required = dict(action_positions='valid_action_start', exclude_initial_from_backtrack=True,
                    fallback='initial_fresh_once_unconditional', remaining_budget='inherit_parent',
                    reward_comparison='absolute_tolerance', sampling_objective=OBJECTIVE)
    for key, value in required.items():
        if tree_option(tree, key) != value:
            raise ValueError('Nested protocol option differs: '+key)
    tolerance = tree_option(tree, 'reward_difference_tolerance')
    if type(tolerance) not in (float, int) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Nested reward tolerance must be explicitly finite and nonnegative')
    return float(tolerance)


def valid_depths(parent, path):
    if len(parent.records) != len(path):
        raise ValueError('Complete snapshot path and action records differ')
    if any('valid_action' not in record for record in parent.records):
        raise ValueError('Nested protocol requires independent per-action validity')
    return [depth for depth, record in enumerate(parent.records) if record['valid_action']]


def checked_scores(engine, states):
    started = time.perf_counter()
    scores = engine.reward_callback(states)
    elapsed = time.perf_counter()-started
    if len(scores) != len(states) or not np.isfinite(scores).all():
        raise ValueError('Nonfinite or misaligned nested candidate reward')
    engine.cost['scoring_calls'] = engine.cost.get('scoring_calls', 0)+len(states)
    engine.cost['scoring_wall_seconds'] = engine.cost.get('scoring_wall_seconds', 0.)+elapsed
    return list(map(float, scores)), elapsed


def nested_branches(engine, roots, paths, quotas, tree, meta):
    tolerance = validate_nested_options(tree)
    if quotas != (2, 3) or len(roots) % 2:
        raise ValueError('Nested protocol requires complete paired root groups')
    if engine.reward_callback is None:
        raise ValueError('Nested branching requires actual training reward callback')
    if any(not path for path in paths):
        raise ValueError('Nested root must have an initial action-start snapshot')
    root_scores, _ = checked_scores(engine, roots)
    scores = {root.trajectory: score for root, score in zip(roots, root_scores)}
    ancestry = {root.trajectory: [] for root in roots}
    leaves = [[root] for root in roots]
    engine.tree_candidate_attempts = []
    engine.tree_candidate_rounds = []
    attempt_ordinals = {}
    engine.tree_candidate_counts = [len(valid_depths(root, path)) for root, path in zip(roots, paths)]
    for index, root in enumerate(roots):
        root.branch_metadata = dict(protocol=PROTOCOL, objective=OBJECTIVE, root_slot=index % 2,
            continuation=0, sampled_reward=scores[root.trajectory], stop_reason='original_root',
            ancestor_trajectory_ids=[], selection_unbiased=False)

    def sample_round(jobs, stage):
        # jobs: (root index, event parent leaf, complete parent path, depth, fallback).
        children, starts, cloned = [], [], []
        for index, parent, path, depth, fallback in jobs:
            start = path[depth]
            child = start.branch()
            if state_signature(start) != state_signature(child):
                raise ValueError('Nested branch changed initial prefix state')
            children.append(child); starts.append(start); cloned.append(copy.deepcopy(child))
        before = engine.cost.copy()
        started = time.perf_counter()
        suffixes = engine.complete(children, meta)
        generation_seconds = time.perf_counter()-started
        rewards, scoring_seconds = checked_scores(engine, children)
        delta = {key: engine.cost[key]-before.get(key, 0) for key in engine.cost}
        # Batch totals include padding/discarded native generation, with no false
        # per-candidate attribution of physical padding overhead.
        round_id = uuid.uuid4().hex
        engine.tree_candidate_rounds.append(dict(round_id=round_id, stage=stage,
            candidate_count=len(jobs), cost_delta=delta, generation_wall_seconds=generation_seconds,
            scoring_wall_seconds=scoring_seconds))
        for key in ('generated_tokens', 'prefill_tokens', 'retrieval_calls', 'raw_emitted_tokens'):
            if key in before:
                engine.cost['branch_'+key] = engine.cost.get('branch_'+key, 0)+delta[key]
        result = []
        for job, child, start, clone, suffix, reward in zip(jobs, children, starts, cloned, suffixes, rewards):
            index, parent, path, depth, fallback = job
            prefix_length = len(start.response)
            if (child.response[:prefix_length] != start.response or
                    child.model_mask[:prefix_length] != start.model_mask or
                    child.steps[:prefix_length] != start.steps or
                    fingerprint(child.raw_ledger[:len(start.raw_ledger)]) != fingerprint(start.raw_ledger)):
                raise ValueError('Nested completed candidate altered inherited prefix/behavior ledger')
            # complete() includes the branching start itself; retain parent[:depth]
            # exactly once. This is the actual child path used by the next stage.
            full_path = path[:depth]+suffix
            valid_depths(child, full_path)
            different = abs(reward-scores[parent.trajectory]) > tolerance
            accepted = stage == 'A1' or fallback or different
            reason = ('random_valid_action_once' if stage == 'A1' and not fallback else
                      'no_valid_action_initial_fallback' if stage == 'A1' else
                      'initial_fresh_fallback_unconditional' if fallback else
                      'first_different_reward' if different else 'same_reward_rejected')
            parent_suffix_depth = parent.branch_metadata.get('action_depth', 0)
            attempt_key = (stage, parent.trajectory)
            attempt_ordinals[attempt_key] = attempt_ordinals.get(attempt_key, 0)+1
            event_id = uuid.uuid4().hex
            inherited_ancestors = ancestry[parent.trajectory]+[parent.trajectory]
            metadata = dict(protocol=PROTOCOL, objective=OBJECTIVE, selection_unbiased=False,
                branch_event_id=event_id, root_slot=index % 2, continuation=1 if stage != 'B2' else 2,
                stage=stage, attempt_ordinal=attempt_ordinals[attempt_key],
                initial_state_attempts=int(fallback or (stage == 'A1' and depth == 0)),
                parent_trajectory_id=parent.trajectory, child_trajectory_id=child.trajectory,
                ancestor_trajectory_ids=inherited_ancestors, actual_prefix_trajectory_id=start.trajectory,
                action_start_id=start.node, action_depth=depth, root_action_states=len(path),
                candidate_valid_action=bool(parent.records[depth]['valid_action']),
                backoff_actions=len(path)-1-depth, backoff_clamped=False, fallback=bool(fallback),
                b2_prefix_origin=('inherited_parent_prefix' if depth < parent_suffix_depth else 'parent_new_suffix')
                    if stage == 'B2' else None,
                remaining_turns=max(0, engine.config.max_turns-start.turns),
                final_answer_available=bool(engine.config.force_final_answer and not start.final_used),
                inherited_response_tokens=prefix_length,
                inherited_observation_tokens=sum(1-m for m in start.model_mask),
                inherited_prefix_sha256=fingerprint(start.rolling),
                inherited_raw_ledger_sha256=fingerprint(start.raw_ledger),
                inherited_raw_ledger_actions=len(start.raw_ledger), inherited_prefix_verified=True,
                finite_quota=quotas[index % 2], sampled_reward=reward,
                parent_reward=scores[parent.trajectory], original_root_reward=root_scores[index],
                same_reward_as_parent=not different, same_reward_as_original=bool(reward == root_scores[index]),
                reason=reason, accepted=accepted, training_leaf=accepted,
                stop_reason='kept_training_leaf' if accepted else 'discarded_not_training',
                reward_difference_tolerance=tolerance, round_id=round_id,
                new_generated_tokens=sum(record['generated_tokens'] for record in child.records[depth:]),
                new_retained_model_tokens=sum(child.model_mask[prefix_length:]),
                new_prefill_tokens=sum(len(snapshot.rolling) for snapshot in suffix),
                retrieval_calls=child.valid_searches-start.valid_searches, scoring_calls=1,
                generation_wall_seconds_amortized=generation_seconds/len(jobs),
                scoring_wall_seconds_amortized=scoring_seconds/len(jobs),
                timing_scope='batched round shared elapsed; amortized candidate share')
            engine.tree_candidate_attempts.append(metadata.copy())
            engine.cost['candidate_attempts'] = engine.cost.get('candidate_attempts', 0)+1
            kind = 'accepted' if accepted else 'discarded'
            engine.cost['candidate_'+kind] = engine.cost.get('candidate_'+kind, 0)+1
            engine.cost['candidate_fallbacks'] = engine.cost.get('candidate_fallbacks', 0)+int(fallback)
            if accepted:
                child.branch_metadata = metadata
                scores[child.trajectory] = reward
                ancestry[child.trajectory] = inherited_ancestors
                if engine.audit is not None:
                    # Only kept leaves enter branch inheritance / PPO audit.
                    engine.audit.branch(start, clone, event_parent=parent, metadata=metadata)
            result.append((index, child, full_path, accepted))
        return result

    a_jobs = []
    for index in range(0, len(roots), 2):
        depths = valid_depths(roots[index], paths[index])
        depth = engine.rng.choice(depths) if depths else 0
        a_jobs.append((index, roots[index], paths[index], depth, not depths))
    for index, child, _, _ in sample_round(a_jobs, 'A1'):
        leaves[index].append(child)

    parents = {i: (roots[i], paths[i]) for i in range(1, len(roots), 2)}
    for stage in ('B1', 'B2'):
        queues = {index: list(reversed([depth for depth in valid_depths(parent, path) if depth != 0]))
                  for index, (parent, path) in parents.items()}
        pending = dict(parents)
        accepted_parents = {}
        while pending:
            jobs = []
            for index, (parent, path) in pending.items():
                fallback = not queues[index]
                depth = queues[index].pop(0) if not fallback else 0
                jobs.append((index, parent, path, depth, fallback))
            for index, child, path, accepted in sample_round(jobs, stage):
                if accepted:
                    leaves[index].append(child)
                    accepted_parents[index] = (child, path)
                    del pending[index]
        parents = accepted_parents
    states = [leaf for root_leaves in leaves for leaf in root_leaves]
    if len({s.trajectory for s in states}) != len(states) or any(len(row) != quotas[i % 2] for i, row in enumerate(leaves)):
        raise ValueError('Nested protocol lost unique five-leaf groups')
    if engine.audit is not None:
        engine.audit.report.update(candidate_attempts=copy.deepcopy(engine.tree_candidate_attempts),
            candidate_rounds=copy.deepcopy(engine.tree_candidate_rounds), objective=OBJECTIVE,
            selection_unbiased=False)
    return states
