"""Driver-side estimators. Observations condition the policy but never receive loss."""
from collections import defaultdict
import math
import torch


def grouped_center(values, keys, normalize=True, epsilon=1e-6):
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    result = torch.zeros_like(values)
    for indices in groups.values():
        v = values[indices]
        if len(indices) > 1:
            centered = v - v.mean()
            result[indices] = centered / (v.std(correction=1) + epsilon) if normalize else centered
    return result


@torch.no_grad()
def gigpo(rewards, mask, step_ids, anchors, uids, gamma=.95, omega=1., normalize=False):
    """One episode per row; discounted action returns, exact within-question anchors.

    This implements the paper's episode weighting (each trajectory once), rather
    than the official optional cross-step length-weighted episode statistics.
    """
    if not 0 <= gamma <= 1:
        raise ValueError('GiGPO gamma must be in [0, 1]')
    episode = grouped_center(rewards.sum(-1), list(uids), normalize)
    positions, keys, returns = [], [], []
    for i in range(len(rewards)):
        steps = sorted(set(step_ids[i][mask[i].bool()].tolist()))
        if any(s < 0 or s >= len(anchors[i]) for s in steps):
            raise ValueError('Missing action metadata')
        # Count zero-token actions in the discount horizon too.
        timeline = range(len(anchors[i]))
        per_step = [rewards[i][step_ids[i] == s].sum() for s in timeline]
        q = rewards.new_tensor(0.)
        future = {}
        for s, r in reversed(list(zip(timeline, per_step))):
            q = r + gamma * q
            future[s] = q
        # A zero-token action has no loss but still contributes to the anchor baseline.
        for s in timeline:
            positions.append((i, s))
            keys.append((uids[i], anchors[i][s]))
            returns.append(future[s])
    advantage = episode[:, None].expand_as(rewards).clone()
    if returns:
        local = grouped_center(torch.stack(returns), keys, normalize)
        for j, (i, s) in enumerate(positions):
            advantage[i, step_ids[i] == s] += omega * local[j]
    return advantage * mask


@torch.no_grad()
def tree_grpo(rewards, mask, uids, trees):
    score = rewards.sum(-1)
    inter = grouped_center(score, list(uids))
    intra = grouped_center(score, list(zip(uids, trees)))
    return (inter + intra)[:, None] * mask


@torch.no_grad()
def topd(teacher_logp, old_logp, mask, uids, alpha=.1):
    """Proximal dense rewards, exclusive future-token mean, pooled group whitening."""
    if not 0 < alpha <= 1:
        raise ValueError('TOP-D requires 0 < alpha <= 1')
    effective = mask.bool()
    # Zero masked inputs before subtraction: masked infinities must not create NaNs.
    d = torch.where(effective, teacher_logp - old_logp, 0.).float()
    reward = d if alpha == 1 else torch.logaddexp(d + math.log(alpha), d.new_full((), math.log1p(-alpha)))
    reward = torch.where(effective, reward, 0.)
    suffix = torch.flip(torch.cumsum(torch.flip(reward, [-1]), -1), [-1]) - reward
    counts = torch.flip(torch.cumsum(torch.flip(mask.float(), [-1]), -1), [-1]) - mask
    returns = (reward + suffix / counts.clamp_min(1)) * mask
    advantage = torch.zeros_like(returns)
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    for rows in groups.values():
        valid = effective[rows]
        v = returns[rows][valid]
        if v.numel() > 1:
            block = (returns[rows] - v.mean()) / (v.std(correction=1) + 1e-6)
            advantage[rows] = torch.where(valid, block, 0.)
    return advantage, returns, reward


def token_mean_ppo(old_logp, logp, advantage, mask, cliprange, denominator):
    """Local numerator / (global token count / DP size); safe for empty microbatches."""
    from verl.trainer.ppo.core_algos import compute_policy_loss
    # Shared numerical guards, with the caller's global normalization.
    loss, clipfrac, kl = compute_policy_loss(old_logp, logp, advantage, mask, cliprange)
    local_count = mask.sum()
    return loss * local_count / denominator, clipfrac, kl
