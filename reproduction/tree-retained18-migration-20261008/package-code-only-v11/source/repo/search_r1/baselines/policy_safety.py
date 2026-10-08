"""Bounded token Dual-Clip and globally mergeable sampled-policy diagnostics.

Unreduced loss preserves the caller's original leaf weights and denominator.
This module never writes an optimizer or claims rollback.
"""
import math
import torch
import torch.distributed as dist


def dual_clip_terms(old, current, advantages, mask, epsilon=0.2, dual_clip=3.0):
    if not 0 < epsilon < 1 or not math.isfinite(dual_clip) or dual_clip <= 1 + epsilon:
        raise ValueError('Require epsilon in (0,1) and dual_clip > 1+epsilon')
    valid = mask.bool()
    for x in (old, current, advantages):
        if not torch.isfinite(x[valid]).all():
            raise ValueError('Nonfinite valid policy input')
    # Mask each input BEFORE subtraction. Invalid padding must not participate.
    delta = torch.where(valid, current, 0.).double() - torch.where(valid, old, 0.).double()
    adv = torch.where(valid, advantages, 0.).double()
    if not torch.isfinite(delta[valid]).all():
        raise ValueError('Nonfinite valid policy difference')
    bounded = torch.where(adv > 0, delta.clamp_max(math.log1p(epsilon)),
                          torch.where(adv < 0, delta.clamp(math.log1p(-epsilon),
                                                           math.log(dual_clip)),
                                      torch.zeros_like(delta)))
    terms = -adv * bounded.exp()
    if not torch.isfinite(terms[valid]).all():
        raise ValueError('Nonfinite bounded policy loss')
    return terms


@torch.no_grad()
def update_statistics(old, current, advantages, mask, weights, dual_clip=3., kl_log_bound=30.):
    if not math.isfinite(dual_clip) or dual_clip <= 1 or not math.isfinite(kl_log_bound) or kl_log_bound <= 0:
        raise ValueError('Invalid explicit numeric bounds')
    valid = mask.bool()
    old = torch.where(valid, old, 0.).double()
    current = torch.where(valid, current, 0.).double()
    adv = torch.where(valid, advantages, 0.).double()
    weight = torch.where(valid, weights, 0.).double()
    delta = current - old
    bad = valid & (~torch.isfinite(old) | ~torch.isfinite(current) |
                   ~torch.isfinite(delta) | ~torch.isfinite(adv) |
                   ~torch.isfinite(weight) | (weight < 0))
    oversized = valid & ~bad & (delta > kl_log_bound)
    good = valid & ~bad
    weight = torch.where(good, weight, 0.)
    safe = torch.where(good & ~oversized, delta, 0.)
    k3 = torch.expm1(safe) - safe
    negative = good & (adv < 0)
    extreme = negative & (delta > math.log(dual_clip))
    kl_sum = (k3 * weight).sum()
    mass = weight.sum()
    stats = dict(mass=mass, kl_sum=kl_sum,
                 negative_mass=torch.where(negative, weight, 0.).sum(),
                 negative_tail_mass=torch.where(extreme, weight, 0.).sum(),
                 count=good.sum(), negative_count=negative.sum(),
                 negative_tail_count=extreme.sum(), saturation_mass=torch.where(extreme, weight, 0.).sum(),
                 invalid_count=bad.sum(), overflow_count=oversized.sum())
    # Any overflowing reduction is also a refusal, never a NaN collective/log.
    for key in ('mass', 'kl_sum', 'negative_mass', 'negative_tail_mass', 'saturation_mass'):
        finite = torch.isfinite(stats[key])
        stats['invalid_count'] = stats['invalid_count'] + (~finite).long()
        stats[key] = torch.where(finite, stats[key], 0.)
    return stats, delta[negative].contiguous()


@torch.no_grad()
def admission(stats, target_kl, negative_tail_limit, group=None):
    if not math.isfinite(target_kl) or target_kl <= 0 or not 0 <= negative_tail_limit <= 1:
        raise ValueError('Explicit calibrated policy thresholds required')
    keys = tuple(stats)
    values = torch.stack([stats[k].double() for k in keys])
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=group)
    nonfinite = ~torch.isfinite(values)
    values = torch.where(nonfinite, 0., values)
    values[keys.index('invalid_count')] += nonfinite.sum()
    result = dict(zip(keys, values.tolist()))
    mass, negative_mass = result['mass'], result['negative_mass']
    kl_valid = result['invalid_count'] == result['overflow_count'] == 0
    kl = result['kl_sum'] / mass if mass > 0 else 0.
    tail = result['negative_tail_mass'] / negative_mass if negative_mass > 0 else 0.
    reasons = []
    if mass <= 0: reasons.append('empty_effective_mass')
    if result['invalid_count']: reasons.append('nonfinite_or_invalid_input_or_reduction')
    if result['overflow_count']: reasons.append('delta_exceeds_kl_log_bound_before_exp')
    if kl_valid and kl > target_kl: reasons.append('sampled_old_policy_kl')
    if tail > negative_tail_limit: reasons.append('negative_tail_fraction')
    result.update(allowed=not reasons, reasons=reasons, kl_computation_valid=kl_valid,
                  nonfinite_global_reductions=int(nonfinite.sum()),
                  sampled_old_policy_kl=kl if kl_valid else None,
                  negative_tail_fraction=tail,
                  negative_tail_count_fraction=(result['negative_tail_count'] / result['negative_count']
                                                if result['negative_count'] else 0.),
                  dual_clip_saturated_weight_fraction=result['saturation_mass'] / mass if mass else 0.)
    return result


@torch.no_grad()
def global_negative_quantiles(samples, group=None):
    """Exact global token quantiles; not a mean of local percentiles."""
    if dist.is_initialized():
        size = torch.tensor([samples.numel()], device=samples.device, dtype=torch.long)
        sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size(group))]
        dist.all_gather(sizes, size, group=group)
        lengths = [int(s.item()) for s in sizes]
        width = max(max(lengths), 1)
        padded = torch.zeros(width, device=samples.device, dtype=torch.float64)
        padded[:samples.numel()] = samples
        values = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(values, padded, group=group)
        samples = torch.cat([value[:length] for value, length in zip(values, lengths)])
    if not samples.numel():
        return dict(negative_log_ratio_p95=None, negative_log_ratio_p99=None,
                    negative_log_ratio_max=None, quantile_sample_count=0)
    q = torch.quantile(samples.double(), torch.tensor([.95, .99, 1.], device=samples.device,
                                                     dtype=torch.float64)).tolist()
    return dict(negative_log_ratio_p95=q[0], negative_log_ratio_p99=q[1],
                negative_log_ratio_max=q[2], quantile_sample_count=samples.numel())
