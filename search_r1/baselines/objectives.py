"""Paper objective measures, computed on complete rollout before DP sharding.

Weights survive row permutations. Mini batches estimate the complete rollout
objective by uniformly sampling rows; no local-rank question re-normalization.
"""
from collections import defaultdict
import math
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


def reduce_paper_metrics(metrics, device):
    """Make detached objective logs identical before DataProto keeps rank zero.

    Each rank's contributions already use the global row/mass denominator
    divided by world size, so averaging them recovers the full minibatch value.
    This collective only touches logs and does not rescale loss or gradients.
    """
    if not torch.distributed.is_initialized():
        return
    keys = [key for key in ('actor/pg_loss', 'actor/entropy_loss', 'actor/kl_loss',
                            'actor/pg_clipfrac', 'actor/ppo_kl') if key in metrics]
    if not keys:
        return
    sizes = [len(metrics[key]) for key in keys]
    values = torch.tensor([value for key in keys for value in metrics[key]],
                          dtype=torch.float64, device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    values /= torch.distributed.get_world_size()
    offset = 0
    for key, size in zip(keys, sizes):
        metrics[key] = values[offset:offset + size].tolist()
        offset += size


def row_denominator(rows, device):
    count = torch.tensor(float(rows), device=device)
    world = 1
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        world = torch.distributed.get_world_size()
    return count.clamp_min(1) / world


def weighted_policy_loss(old, current, advantages, mask, weights, cliprange, denominator, step_ids=None, dual_clip=None):
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
    if dual_clip is not None:
        if step_ids is not None:
            raise ValueError("Token Dual-Clip integration only; GiGPO requires separate action review")
        from .policy_safety import dual_clip_terms
        terms = dual_clip_terms(old, current, advantages, valid, cliprange, dual_clip)
        loss = F.masked_sum(terms, weights) / denominator
        safe_mass = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
        clipped = ((advantages > 0) & (delta > math.log1p(cliprange))) | ((advantages < 0) & (delta < math.log1p(-cliprange)))
        clipfrac = F.masked_sum(clipped.float(), weights) / safe_mass
        kl = F.masked_sum(-delta, weights) / safe_mass
        return loss, clipfrac, kl
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
