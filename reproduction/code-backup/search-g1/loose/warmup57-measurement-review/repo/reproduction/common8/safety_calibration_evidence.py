"""Pure structural verification of common8 calibration worker records.

No files, workers, thresholds, or training state are changed. The result is not
production admission: Qwen identity, fresh initialization, probability parity,
parameter/Adam finite proofs and health rules require separate bound evidence.
"""

import copy
import math
import re
from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 1
CALIBRATION_ROLES = frozenset((
    'policy_safety_calibration_discovery',
    'policy_safety_calibration_confirmation',
))


def _enforcement(role):
    return dict(negative_tail='observe_only' if role == 'policy_safety_calibration_discovery' else 'enforced',
                old_policy_kl='enforced', numerical_finite='enforced', dual_clip='enforced')
OBJECTIVE_WEIGHT_DEFINITION = 'equal two-row micro masked_mean; weighted global numerator/denominator'
MONITORING_SCOPE = 'fixed four trajectories per rank across length strata; not full rollout'
QUANTILE_METHOD = 'exact global nearest-order statistic'
# IEEE-754 binary32 has 24 significant bits: epsilon is 2**-23 and
# round-to-nearest unit roundoff is epsilon/2. These are numerical tolerances,
# independent of policy KL/tail thresholds or any future health criterion.
FLOAT32_EPSILON = 2. ** -23
GRADIENT_CLIP_LIMIT = 1.
GRADIENT_CLIP_FLOAT32_TOLERANCE = FLOAT32_EPSILON
GLOBAL_FIELDS = (
    'allowed', 'reasons', 'sampled_old_policy_kl', 'kl_estimate_complete',
    'effective_weight_mass', 'weighted_k3_numerator',
    'negative_effective_weight_mass', 'tail_effective_weight_mass',
    'effective_token_count', 'negative_token_count', 'negative_tail_count',
    'negative_tail_count_fraction', 'negative_tail_weight_fraction',
    'dual_clip_saturated_objective_weight_fraction', 'invalid_count',
    'overflow_count', 'quantile_method', 'negative_log_ratio_p95',
    'negative_log_ratio_p99', 'negative_log_ratio_max',
)
ADJACENT_FLOAT_FIELDS = frozenset((
    'weighted_k3_numerator', 'sampled_old_policy_kl',
    'negative_tail_count_fraction', 'negative_tail_weight_fraction',
    'dual_clip_saturated_objective_weight_fraction',
    'negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max',
))
MONITOR_TENSORS = frozenset((
    'responses', 'input_ids', 'attention_mask', 'position_ids',
    'old_log_probs', 'advantages', 'loss_mask', 'ref_log_prob',
))
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')


class EvidenceError(ValueError):
    """Input evidence cannot prove a complete common8 calibration segment."""


def _require(ok, message):
    if not ok:
        raise EvidenceError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, 'Invalid integer: ' + name)
    return value


def _finite(value, name, nonnegative=True):
    _require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite number: ' + name)
    _require(not nonnegative or value >= 0, 'Negative number: ' + name)
    return value


def _same_arithmetic(actual, expected, name):
    # This tolerance only checks FP64 report arithmetic, never health thresholds.
    _require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
             'Inconsistent report arithmetic: ' + name)


def _hash(value, name):
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             'Invalid SHA256: ' + name)


def _adjacent(previous, current):
    # Identical weights/inputs between post(j) and pre(j+1) mean the same
    # observation. Counts, decisions and all masses must remain exact. Only
    # forward-derived floats permit one binary32 epsilon (relative and absolute
    # at unit scale) for FP32 recomputation; this is not a model-error bound.
    for name in GLOBAL_FIELDS:
        left, right = previous[name], current[name]
        if name in ADJACENT_FLOAT_FIELDS and left is not None and right is not None:
            _require(math.isclose(left, right, rel_tol=FLOAT32_EPSILON, abs_tol=FLOAT32_EPSILON),
                     'Adjacent unchanged-policy float differs: ' + name)
        else:
            _require(left == right, 'Adjacent unchanged-policy exact field differs: ' + name)


def _identity(value):
    _require(isinstance(value, Mapping), 'Missing local monitor identity')
    _require(set(value) == {'selection_indices', 'actor_token_lengths', 'tensor_sha256'},
             'Unexpected monitor identity schema')
    indices = value['selection_indices']
    lengths = value['actor_token_lengths']
    _require(isinstance(indices, list) and len(indices) == 4, 'Need four monitor indices')
    for index in indices:
        _integer(index, 'selection_index')
        _require(index < 320, 'Monitor index outside local rollout')
    _require(len(set(indices)) == 4, 'Duplicate monitor index')
    _require(isinstance(lengths, list) and len(lengths) == 4, 'Need four monitor lengths')
    for length in lengths:
        _integer(length, 'actor_token_length', minimum=1)
    _require(lengths == sorted(lengths), 'Monitor lengths are not stratified')
    hashes = value['tensor_sha256']
    _require(isinstance(hashes, Mapping) and set(hashes) == MONITOR_TENSORS,
             'Missing actual common8 monitor tensor hashes')
    for name, digest in hashes.items():
        _hash(digest, name)


def _statistics(row):
    _require(row['allowed'] is True and row['reasons'] == [], 'Rejected monitor')
    _require(row['kl_estimate_complete'] is True, 'Incomplete KL is diagnostic only')
    for name in ('effective_token_count', 'negative_token_count', 'negative_tail_count',
                 'invalid_count', 'overflow_count'):
        _integer(row[name], name)
    _require(row['invalid_count'] == row['overflow_count'] == 0, 'Invalid or over-bound monitor')
    count, negative, tail = (row[name] for name in
                            ('effective_token_count', 'negative_token_count', 'negative_tail_count'))
    _require(count > 0 and 0 <= tail <= negative <= count, 'Inconsistent token counts')
    mass, numerator, negative_mass, tail_mass = (
        _finite(row[name], name) for name in
        ('effective_weight_mass', 'weighted_k3_numerator',
         'negative_effective_weight_mass', 'tail_effective_weight_mass'))
    _require(mass > 0, 'Empty effective weight mass')
    _require(negative_mass <= mass or math.isclose(negative_mass, mass, rel_tol=1e-12, abs_tol=1e-12),
             'Negative mass exceeds total mass')
    _require(tail_mass <= negative_mass or math.isclose(tail_mass, negative_mass, rel_tol=1e-12, abs_tol=1e-12),
             'Tail mass exceeds negative mass')
    _require((negative == 0) == (negative_mass == 0), 'Negative count/mass disagree')
    _require((tail == 0) == (tail_mass == 0), 'Tail count/mass disagree')
    expected = {
        'sampled_old_policy_kl': numerator / mass,
        'negative_tail_count_fraction': tail / negative if negative else 0.,
        'negative_tail_weight_fraction': tail_mass / negative_mass if negative_mass else 0.,
        'dual_clip_saturated_objective_weight_fraction': tail_mass / mass,
    }
    for name, ratio in expected.items():
        _same_arithmetic(_finite(row[name], name), ratio, name)
    _require(row['quantile_method'] == QUANTILE_METHOD, 'Unexpected quantile method')
    quantiles = [row[name] for name in
                 ('negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max')]
    if negative:
        for value in quantiles:
            _finite(value, 'negative log-ratio quantile', nonnegative=False)
        _require(quantiles == sorted(quantiles), 'Inconsistent negative quantiles')
    else:
        _require(quantiles == [None, None, None], 'Quantiles without negative tokens')
    return {name: copy.deepcopy(row[name]) for name in GLOBAL_FIELDS}


def _row(row, rank, binding, kind):
    _require(isinstance(row, Mapping), 'Malformed worker record')
    _require(row['kind'] == kind and row['run_id'] == binding['run_id'], 'Wrong record kind or run')
    _require(_integer(row['rank'], 'rank') == rank, 'Wrong rank in worker log')


def _context(row, outer, mini, prior):
    context = row['context']
    expected = dict(logical_update=outer, minibatch_index=mini, prior_optimizer_writes=prior)
    gate_key = 'behavior_probability_gate_sha256'
    _require(isinstance(context, Mapping) and set(context) in
             (set(expected), set(expected) | {gate_key}), 'Unexpected context schema')
    if gate_key in context:
        _hash(context[gate_key], 'actual behavior gate context')
    for name, value in expected.items():
        _require(_integer(context[name], name) == value, 'Wrong mini/write context: ' + name)


def verify_segment(rows_by_rank, binding):
    """Validate complete fresh-counter logs and coalesce rank replicas once.

    ``binding`` carries schema_version=1, a calibration discovery/confirmation
    role, run_id, calibration_sha256, outer_count and initial_optimizer_writes=0.
    ``rows_by_rank`` contains the actual ordered worker records for ranks 0..7.
    Explicit synthetic/component roles are rejected. This pure check cannot
    authenticate physical execution from JSON; it never grants production or
    healthy admission, even when all supplied records are structurally valid.
    """
    try:
        return _verify_segment(rows_by_rank, binding)
    except EvidenceError:
        raise
    except (KeyError, TypeError, OverflowError) as failure:
        raise EvidenceError('Malformed calibration evidence: ' + str(failure)) from failure


def _verify_segment(rows_by_rank, binding):
    _require(isinstance(binding, Mapping), 'Missing evidence binding')
    _require(type(binding['schema_version']) is int and binding['schema_version'] == SCHEMA_VERSION,
             'Unsupported evidence schema')
    _require(binding['role'] in CALIBRATION_ROLES, 'Not an actual-policy calibration role')
    _require(isinstance(binding['run_id'], str) and bool(binding['run_id']), 'Missing run identity')
    _hash(binding['calibration_sha256'], 'calibration binding')
    outers = _integer(binding['outer_count'], 'outer_count', minimum=1)
    _require(_integer(binding['initial_optimizer_writes'], 'initial_optimizer_writes') == 0,
             'Calibration counters must start at zero')
    _require(isinstance(rows_by_rank, Mapping) and
             all(type(rank) is int for rank in rows_by_rank) and set(rows_by_rank) == set(range(8)),
             'Need exactly eight distinct rank logs')
    observations = {}
    behavior_gates = {}
    identities = {outer: {} for outer in range(1, outers + 1)}
    for rank in range(8):
        rows = rows_by_rank[rank]
        _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and len(rows) == outers * 61,
                 'Incomplete or duplicate write/check/budget schedule on rank ' + str(rank))
        cursor = 0
        for outer in range(1, outers + 1):
            identity = None
            frozen_statistics = None
            for mini in range(20):
                prior = (outer - 1) * 20 + mini
                before, write, after = rows[cursor:cursor + 3]
                cursor += 3
                for row, stage, writes in ((before, 'before_minibatch', prior),
                                          (after, 'after_minibatch', prior + 1)):
                    _row(row, rank, binding, 'POLICY_MONITOR')
                    _context(row, outer, mini, prior)
                    _require(row['stage'] == stage, 'Wrong pre/post monitor order')
                    _require(_integer(row['optimizer_writes'], 'optimizer_writes') == writes,
                             'Wrong monitor write count')
                    _require(row['calibration_sha256'] == binding['calibration_sha256'], 'Changed calibration binding')
                    _require(row['execution_role'] == binding['role'] and
                             row['enforcement'] == _enforcement(binding['role']),
                             'Calibration role or enforced brakes differ')
                    _require(row['objective_weight_definition'] == OBJECTIVE_WEIGHT_DEFINITION and
                             row['monitoring_scope'] == MONITORING_SCOPE, 'Unexpected monitoring definition')
                    _require(_integer(row['rejection_count'], 'rejection_count') == 0, 'Prior policy rejection')
                    _identity(row['monitor_identity'])
                    if identity is None:
                        identity = copy.deepcopy(row['monitor_identity'])
                        identities[outer][rank] = identity
                    _require(row['monitor_identity'] == identity, 'Local monitor identity changed within outer')
                    statistics = _statistics(row)
                    fixed = tuple(statistics[name] for name in
                                  ('effective_weight_mass', 'negative_effective_weight_mass',
                                   'effective_token_count', 'negative_token_count'))
                    if frozen_statistics is None:
                        frozen_statistics = fixed
                    _require(fixed == frozen_statistics, 'Frozen mask/advantage statistics changed within outer')
                    key = (binding['run_id'], outer, mini, stage)
                    if key not in observations:
                        observations[key] = dict(run_id=binding['run_id'], context=copy.deepcopy(row['context']),
                                                 stage=stage, optimizer_writes=writes,
                                                 calibration_sha256=binding['calibration_sha256'],
                                                 execution_role=binding['role'],
                                                 enforcement=copy.deepcopy(row['enforcement']), **statistics)
                    else:
                        _require(all(observations[key][name] == statistics[name] for name in GLOBAL_FIELDS),
                                 'Global monitor fields disagree across ranks')
                _row(write, rank, binding, 'OPTIMIZER_WRITE')
                _context(write, outer, mini, prior)
                # Old immutable logs have no behavior receipt. New logs bind
                # every check/write, across all ranks, to one gate per outer.
                for event in (before, write, after):
                    gate = event['context'].get('behavior_probability_gate_sha256')
                    if outer not in behavior_gates:
                        behavior_gates[outer] = gate
                    _require(gate == behavior_gates[outer],
                             'Behavior probability gate changed within outer or across ranks')
                _require(_integer(write['write_count'], 'write_count') == prior + 1, 'Wrong actual write count')
                rates = write['learning_rates']
                _require(isinstance(rates, list) and bool(rates), 'Missing actual learning rates')
                for rate in rates:
                    _require(_finite(rate, 'learning_rate') == 1e-6, 'Actual learning rate differs')
                for name in ('gradient_norm_before', 'gradient_norm_after'):
                    _finite(write[name], name)
                # FP32 coordinate rounding has unit roundoff epsilon/2; one
                # epsilon also leaves room for FP64 norm arithmetic/serialization.
                _require(write['gradient_norm_after'] <= GRADIENT_CLIP_LIMIT + GRADIENT_CLIP_FLOAT32_TOLERANCE,
                         'Actual post-clip norm exceeds clip1 numerical bound')
            budget = rows[cursor]
            cursor += 1
            _row(budget, rank, binding, 'OUTER_BUDGET')
            for name, expected in dict(logical_update=outer, local_trajectories=320,
                                       local_mini=16, local_micro=2, optimizer_writes=20,
                                       cumulative_writes=outer * 20).items():
                _require(_integer(budget[name], name) == expected, 'Wrong actual outer budget: ' + name)
    for observation in observations.values():
        outer = observation['context']['logical_update']
        count = sum(sum(identity['actor_token_lengths']) for identity in identities[outer].values())
        _require(observation['effective_token_count'] == count, 'Global token count differs from frozen monitors')
        _same_arithmetic(observation['effective_weight_mass'], 16., 'eight-rank two-row micro weight mass')
    for outer in range(1, outers + 1):
        for mini in range(1, 20):
            previous = observations[(binding['run_id'], outer, mini - 1, 'after_minibatch')]
            current = observations[(binding['run_id'], outer, mini, 'before_minibatch')]
            _require(previous['optimizer_writes'] == current['optimizer_writes'],
                     'Unexpected write between adjacent monitors')
            _adjacent(previous, current)
    return dict(schema_version=SCHEMA_VERSION, role='policy_safety_calibration_structural_verification',
                execution_role=binding['role'], run_id=binding['run_id'],
                calibration_sha256=binding['calibration_sha256'], outer_count=outers,
                optimizer_writes_per_rank=outers * 20, global_monitors=list(observations.values()),
                monitor_identities=identities,
                enforcement=_enforcement(binding['role']), production_admission=False)
