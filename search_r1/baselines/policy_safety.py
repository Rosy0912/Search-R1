"""Opt-in Dual-Clip and synchronized old-policy admission; no rollback or skipped outer."""
import math
import re
import torch
import torch.distributed as dist


def validate_config(config, require_thresholds=True):
    if not config.get('enabled', False):
        return
    c, bound = config['dual_clip'], config['kl_log_bound']
    if not math.isfinite(c) or c <= 1.2 or not math.isfinite(bound) or not 0 < bound <= 30:
        raise ValueError('Invalid Dual-Clip or exponential rejection boundary')
    if config.get('monitor_first_global_rows') != 256:
        raise ValueError('This continuation monitors the fixed first global256 rows')
    if require_thresholds:
        kl, tail = config.get('target_old_policy_kl'), config.get('negative_tail_limit')
        if (kl is None or tail is None or not math.isfinite(kl) or kl <= 0
                or not math.isfinite(tail) or not 0 <= tail <= 1
                or re.fullmatch(r'[0-9a-f]{64}', str(config.get('calibration_evidence_sha256'))) is None):
            raise ValueError('Production policy safety requires explicit GPU-calibrated thresholds and evidence')
        import hashlib, json, os
        from pathlib import Path
        path = config.get('calibration_evidence_path')
        if not path or not Path(path).is_file():
            raise ValueError('Missing independent GPU calibration acceptance artifact')
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != config['calibration_evidence_sha256']:
            raise ValueError('Calibration acceptance SHA mismatch')
        proof = json.loads(raw)
        if (proof.get('status') != 'PASS' or proof.get('scope') != 'GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION'
                or proof.get('rank_count') != 8
                or proof.get('restore_manifest_sha256') != '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
                or proof.get('target_old_policy_kl') != kl or proof.get('negative_tail_limit') != tail
                or proof.get('dual_clip') != c or proof.get('kl_log_bound') != bound):
            raise ValueError('Calibration acceptance scope/threshold identity mismatch')
        actual_source = os.environ.get('SEARCH_R1_BASELINE_COMMIT')
        if actual_source and proof.get('source_commit') != actual_source:
            raise ValueError('Calibration source differs from actual deployed actor')


def dual_clip_terms(delta, advantages, mask, epsilon, c):
    if not 0 < epsilon < 1 or not math.isfinite(c) or c <= 1 + epsilon:
        raise ValueError('Invalid Dual-Clip coefficient')
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError('Invalid effective policy mask')
    valid = mask.bool()
    if not torch.isfinite(delta[valid]).all() or not torch.isfinite(advantages[valid]).all():
        raise ValueError('Nonfinite effective policy input')
    delta = torch.where(valid, delta, 0.).double()
    adv = torch.where(valid, advantages, 0.).double()
    bounded = torch.where(adv > 0, delta.clamp_max(math.log1p(epsilon)),
                          torch.where(adv < 0, delta.clamp(math.log1p(-epsilon), math.log(c)),
                                      torch.zeros_like(delta)))
    return -adv * bounded.exp()


def joint_log_ratios(old, current, advantages, mask, step_ids):
    """Promote before subtraction AND sum; preserve complete textual actions."""
    valid = mask.bool()
    if step_ids.shape != mask.shape or step_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('Invalid action metadata')
    token = torch.where(valid, current.double(), 0.) - torch.where(valid, old.double(), 0.)
    rows = []
    for i in range(len(token)):
        row = token[i] * 0
        for step in torch.unique(step_ids[i][valid[i]]):
            selected = valid[i] & (step_ids[i] == step)
            adv = advantages[i][selected]
            if step < 0 or not torch.allclose(adv, adv[:1].expand_as(adv)):
                raise ValueError('Invalid joint action advantage/identity')
            row = row + selected * token[i][selected].sum()
        rows.append(row)
    return token, torch.stack(rows)


@torch.no_grad()
def local_statistics(old, current, advantages, mask, weights, step_ids, c, bound):
    """Finite mergeable sums; action counts are independent actions, not broadcasts."""
    valid = mask.bool()
    if step_ids.shape != mask.shape or step_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        sums = torch.zeros(11, device=old.device, dtype=torch.float64); sums[3] = 1; sums[9] = 1
        empty = old.new_empty(0, dtype=torch.float64)
        return sums, empty, empty, old.new_zeros((), dtype=torch.float64)
    token = torch.where(valid, current.double(), 0.) - torch.where(valid, old.double(), 0.)
    corrupt_mask = ~torch.isfinite(mask) | ((mask != 0) & (mask != 1))
    bad = valid & (~torch.isfinite(old) | ~torch.isfinite(current) | ~torch.isfinite(token)
                   | ~torch.isfinite(advantages) | ~torch.isfinite(weights) | (weights < 0))
    safe_weights = torch.where(valid & ~bad, weights.double(), 0.)
    oversized = valid & (token > bound)
    safe_delta = torch.where(bad | oversized, 0., token)
    k3 = torch.expm1(safe_delta) - safe_delta
    # Refuse an overflowing diagnostic sum rather than report a truncated KL.
    kl_sum = (k3 * safe_weights).sum()
    extra_bad = ~torch.isfinite(kl_sum)
    kl_sum = torch.where(extra_bad, 0., kl_sum)
    sums = torch.zeros(11, device=old.device, dtype=torch.float64)
    sums[0:4] = torch.stack((safe_weights.sum(), kl_sum, valid.sum().double(),
                             (bad.sum() + oversized.sum() + corrupt_mask.sum() + extra_bad).double()))
    samples = []
    lengths = []
    for i in range(len(token)):
        for step in torch.unique(step_ids[i][valid[i]]):
            selected = valid[i] & (step_ids[i] == step)
            adv = advantages[i][selected]
            value = token[i][selected].sum()
            if (step < 0 or not torch.allclose(adv, adv[:1].expand_as(adv)) or not torch.isfinite(value)):
                sums[3] += 1
                continue
            mass = safe_weights[i][selected].sum()
            lengths.append(selected.sum().double())
            sums[4] += 1  # all independently counted actions
            if bool((adv < 0).all()):
                sums[5] += 1
                sums[6] += mass
                saturated = value > math.log(c)
                sums[7] += saturated.double()
                sums[8] += mass * saturated
                samples.append(value)
    sums[9] = (bad.sum() + corrupt_mask.sum()).double()
    sums[10] = oversized.sum().double()
    maximum = torch.where(valid & ~bad, token.abs(), 0.).max() if token.numel() else torch.zeros((), device=old.device, dtype=torch.float64)
    return sums, torch.stack(samples) if samples else token.new_empty(0), torch.stack(lengths) if lengths else token.new_empty(0), maximum


@torch.no_grad()
def global_statistics(parts, group=None):
    sums = torch.stack([p[0] for p in parts]).sum(0)
    negatives = torch.cat([p[1] for p in parts])
    lengths = torch.cat([p[2] for p in parts])
    maximum = torch.stack([p[3] for p in parts]).max()
    if dist.is_initialized():
        dist.all_reduce(sums, group=group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    def gather(values):
        if not dist.is_initialized():
            return values
        count = torch.tensor(values.numel(), device=sums.device, dtype=torch.int64)
        counts = [torch.zeros_like(count) for _ in range(dist.get_world_size(group))]
        dist.all_gather(counts, count, group=group)
        capacity = max(1, max(int(n) for n in counts))
        padded = torch.zeros(capacity, device=sums.device, dtype=torch.float64)
        padded[:values.numel()] = values
        gathered = [torch.empty_like(padded) for _ in counts]
        dist.all_gather(gathered, padded, group=group)
        return torch.cat([v[:int(n)] for v, n in zip(gathered, counts)])
    negatives, lengths = gather(negatives), gather(lengths)
    values = sums.tolist()
    mass, kl_sum, tokens, rejected, actions, negative_count, negative_mass, tail_count, tail_mass, invalid, over = values
    out = dict(weight_mass=mass, valid_token_count=int(tokens), independent_action_count=int(actions),
               sampled_old_policy_kl=None if rejected or mass <= 0 else kl_sum / mass,
               invalid_count=int(invalid), kl_log_bound_exceeded_count=int(over), rejected_numeric_count=int(rejected),
               negative_action_count=int(negative_count), negative_action_weight_mass=negative_mass,
               negative_tail_fraction_count=tail_count / negative_count if negative_count else 0.,
               negative_tail_fraction_weight=tail_mass / negative_mass if negative_mass else 0.,
               dual_clip_saturated_action_fraction=tail_count / actions if actions else 0.,
               dual_clip_saturated_objective_mass_fraction=tail_mass / mass if mass else 0.,
               max_abs_token_delta=float(maximum),
               negative_action_logratio_p95=float(torch.quantile(negatives, .95)) if negatives.numel() else None,
               negative_action_logratio_p99=float(torch.quantile(negatives, .99)) if negatives.numel() else None,
               negative_action_logratio_max=float(negatives.max()) if negatives.numel() else None,
               action_token_length_p95=float(torch.quantile(lengths, .95)) if lengths.numel() else None,
               action_token_length_max=float(lengths.max()) if lengths.numel() else None,
               coverage='fixed_first_global256_trajectories_complete_actions', quantiles='exact_global_samples')
    return out


def rejection_reason(stats, config):
    if stats['rejected_numeric_count'] or stats['weight_mass'] <= 0:
        return 'NONFINITE_OR_EXP_RANGE_OR_EMPTY_EFFECTIVE_MONITOR'
    if stats['sampled_old_policy_kl'] > config['target_old_policy_kl']:
        return 'OLD_POLICY_TOKEN_KL'
    if stats['negative_tail_fraction_count'] > config['negative_tail_limit']:
        return 'NEGATIVE_JOINT_ACTION_TAIL_COUNT'
    return None
