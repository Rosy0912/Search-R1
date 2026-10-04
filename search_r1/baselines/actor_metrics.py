"""All-rank reporting for the fixed BAPO sequence-mean actor objective.

Losses are local numerators divided by (global mini rows / DP world size),
so averaging ranks reconstructs the global objective, even for uneven rows.
This is reporting only: it does not modify gradients, optimizer, or schedules.
"""
import math
import torch.distributed as dist


_MEAN = {'actor/pg_loss', 'actor/kl_loss', 'actor/entropy_loss',
         'actor/ppo_kl', 'actor/pg_clipfrac', 'mfu/actor'}
_SUM = {'bapo/old_logprob_nonfinite', 'bapo/new_logprob_nonfinite', 'bapo/loss_tokens'}
_REPLICATED = {'actor/grad_norm', 'bapo/grad_norm_min', 'bapo/grad_norm_max',
               'actor/optimizer_steps', 'actor/skipped_empty_minibatches',
               'actor/lr', 'actor/lr_before_outer', 'actor/scheduler_outer_steps',
               'actor/kl_coef'}


def validate_actor_metric_topology(sequence_parallel_size=1, rollout_tensor_parallel_size=1):
    # This implementation is deliberately scoped to the approved TP1/SP1 run.
    # Ulysses duplicates row measures and requires its DP group, not WORLD.
    if sequence_parallel_size != 1 or rollout_tensor_parallel_size != 1:
        raise ValueError('BAPO all-rank sequence-mean metrics require TP1 and Ulysses SP1')


def _values(value):
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
        raise ValueError('Actor metric values must be finite numeric scalars/lists')
    return values


def merge_actor_metric_records(records):
    """Pure reducer; each record is already normalized by the global denominator.

    Do NOT weight losses by local row counts a second time. Replica metrics
    (global gradient norm, clock) are checked and retained once, never summed.
    """
    if not records:
        raise ValueError('Missing actor metric records')
    keys = set(records[0])
    if any(set(r) != keys for r in records):
        raise ValueError('Actor metric keys disagree across ranks')
    unknown = keys - _MEAN - _SUM - _REPLICATED
    if unknown:
        raise ValueError('Unclassified all-rank actor metrics: ' + ', '.join(sorted(unknown)))
    result = {}
    for key in sorted(keys):
        vectors = [_values(r[key]) for r in records]
        list_shape = isinstance(records[0][key], (list, tuple))
        if any(isinstance(r[key], (list, tuple)) != list_shape for r in records):
            raise ValueError('Actor metric scalar/list shape mismatch: ' + key)
        if any(len(v) != len(vectors[0]) for v in vectors):
            raise ValueError('Actor mini counts disagree across ranks: ' + key)
        if key in _REPLICATED:
            if any(any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-15)
                       for a, b in zip(vectors[0], v)) for v in vectors[1:]):
                raise ValueError('Replicated actor metric disagrees across ranks: ' + key)
            combined = vectors[0]
        else:
            combined = [math.fsum(items) for items in zip(*vectors)]
            if key in _MEAN:
                combined = [x / len(records) for x in combined]
        result[key] = combined if list_shape else combined[0]
    return result


def aggregate_sequence_mean_actor_metrics(metrics, sequence_parallel_size=1,
                                          rollout_tensor_parallel_size=1):
    validate_actor_metric_topology(sequence_parallel_size, rollout_tensor_parallel_size)
    if dist.is_initialized():
        records = [None] * dist.get_world_size()
        # Metadata is small CPU JSON-like data; NCCL uses current CUDA device
        # already selected by the worker. All ranks execute the same collective.
        dist.all_gather_object(records, metrics)
    else:
        records = [metrics]
    return merge_actor_metric_records(records)
