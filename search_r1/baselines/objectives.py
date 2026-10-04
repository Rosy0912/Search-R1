"""Paper objective measures, computed on complete rollout before DP sharding.

Weights survive row permutations. Mini batches estimate the complete rollout
objective by uniformly sampling rows; no local-rank question re-normalization.
"""
from collections import defaultdict
import torch
from verl.utils import torch_functional as F


@torch.no_grad()
def attach_policy_weights(data, method):
    width = data.batch['responses'].shape[1]
    mask = (data.batch['attention_mask'][:, -width:] * data.batch['info_mask'][:, -width:]).float()
    weights = torch.zeros_like(mask)
    if method == 'topd':
        groups = defaultdict(list)
        for i, uid in enumerate(data.non_tensor_batch['uid']):
            groups[uid].append(i)
        # Empty questions remain in the outer expectation with zero contribution.
        scale = len(data) / max(1, len(groups))
        for rows in groups.values():
            count = mask[rows].sum()
            if count > 0:
                weights[rows] = mask[rows] * scale / count
    elif method == 'gigpo':
        for i in range(len(data)):
            steps = data.batch['step_ids'][i]
            actions = torch.unique(steps[mask[i].bool()])
            if (actions < 0).any():
                raise ValueError('A supervised GiGPO token has no action ID')
            for step in actions:
                selected = (steps == step) & mask[i].bool()
                weights[i, selected] = 1. / (len(actions) * selected.sum())
    elif method in ('tree_grpo', 'cosearch'):
        weights = mask / mask.sum(-1, keepdim=True).clamp_min(1)
    else:
        raise ValueError(f'No paper policy measure for {method}')
    data.batch['policy_weights'] = weights
    return data


def row_denominator(rows, device):
    from .distillation import global_sum_count
    count, world = global_sum_count(torch.tensor(float(rows), device=device))
    return count.clamp_min(1) / world


def weighted_policy_loss(old, current, advantages, mask, weights, cliprange, denominator, step_ids=None, log_ratio_clip=None):
    """Token PPO, or GiGPO's joint textual-action ratio (Eq.9), with row measure."""
    valid = mask.bool()
    for name, value in [('old', old), ('current', current), ('advantage', advantages), ('weight', weights)]:
        if not torch.isfinite(value[valid]).all():
            raise ValueError(f'Nonfinite policy {name} on a supervised token')
    old = torch.where(valid, old, 0.)
    current = torch.where(valid, current, 0.)
    advantages = torch.where(valid, advantages, 0.)
    weights = torch.where(valid, weights, 0.)
    delta = current - old
    if step_ids is not None:
        action_deltas = []
        for i in range(len(delta)):
            row = delta[i] * 0
            for step in torch.unique(step_ids[i][valid[i]]):
                selected = (step_ids[i] == step) & valid[i]
                if step < 0:
                    raise ValueError('Missing GiGPO action metadata')
                action_advantages = advantages[i][selected]
                if not torch.allclose(action_advantages, action_advantages[:1].expand_as(action_advantages)):
                    raise ValueError('GiGPO advantage must be constant within one textual action')
                row = row + selected * delta[i][selected].sum()
            action_deltas.append(row)
        delta = torch.stack(action_deltas)
    if log_ratio_clip is not None:
        if log_ratio_clip <= 0 or not __import__("math").isfinite(log_ratio_clip):
            raise ValueError("log_ratio_clip must be finite and positive")
        # Author BAPO token-PPO stability guard, before exponentiation.
        delta = delta.clamp(-log_ratio_clip, log_ratio_clip)
    ratio = delta.exp()
    if not torch.isfinite(ratio[valid]).all():
        raise ValueError('Policy ratio overflow on a supervised action/token')
    loss1 = -advantages * ratio
    loss2 = -advantages * ratio.clamp(1 - cliprange, 1 + cliprange)
    loss = F.masked_sum(torch.maximum(loss1, loss2), weights) / denominator
    mass = weights.sum()
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    clipfrac = F.masked_sum((loss2 > loss1).float(), weights) / safe_mass
    kl = F.masked_sum(-delta, weights) / safe_mass
    return loss, clipfrac, kl
