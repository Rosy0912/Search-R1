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
def tree_variance_metrics(score, uids, trees, branch_metadata=None, estimator_score=None):
    """Equal-question averages of population moments, using the unchanged estimator.

    A is the actual scalar Tree-GRPO advantage (not token-length weighted).
    Between-root variance uses root means of A, with both equal-tree and 3/2
    leaf-mass versions made explicit. These diagnostics make no improvement claim.
    """
    score = score.detach()
    estimator_score = score if estimator_score is None else estimator_score.detach()
    inter = grouped_center(estimator_score, list(uids))
    intra = grouped_center(estimator_score, list(zip(uids, trees)))
    # Preserve the actual training dtype and epsilon arithmetic before moments.
    actual = (inter + intra).double()
    inter, intra, score = inter.double(), intra.double(), score.double()
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    values = defaultdict(list)
    for rows in groups.values():
        local_trees = defaultdict(list)
        for i in rows:
            local_trees[trees[i]].append(i)
        if len(rows) != 5 or sorted(map(len, local_trees.values())) != [2, 3]:
            # Historical rectangular runs remain supported by this diagnostic.
            if branch_metadata is not None and any(branch_metadata[i].get('protocol') in ('branpo_tail_v1', 'random_nested_backtrack_v1') for i in rows):
                raise ValueError('Tail metrics lost the five-leaf 3/2 group')
        r = score[rows]
        a = actual[rows]
        values['group_raw_reward_population_variance'].append(r.var(correction=0).item())
        values['group_actual_adv_population_variance'].append(a.var(correction=0).item())
        values['group_inter_normalized_component_population_variance'].append(inter[rows].var(correction=0).item())
        values['same_reward_question_fraction'].append(float(r.max() == r.min()))
        values['mixed_reward_question_fraction'].append(float(r.max() != r.min()))
        values['exact_reward_one_question_count'].append(float(torch.all(r == 1)))
        values['exact_reward_zero_question_count'].append(float(torch.all(r == 0)))
        values['mixed_question_count'].append(float(r.max() != r.min()))
        means, reward_means, masses, within, intra_within = [], [], [], [], []
        for indices in local_trees.values():
            tr, ta = score[indices], actual[indices]
            means.append(ta.mean()); reward_means.append(tr.mean()); masses.append(len(indices)/len(rows))
            v = ta.var(correction=0).item(); within.append(v)
            iv = intra[indices].var(correction=0).item(); intra_within.append(iv)
            values['within_root_raw_reward_population_variance'].append(tr.var(correction=0).item())
            values['within_root_actual_adv_population_variance'].append(v)
            values['within_root_raw_centered_adv_population_variance'].append(tr.var(correction=0).item())
            values['within_root_normalized_component_population_variance'].append(intra[indices].var(correction=0).item())
            values['intra_adv_population_variance'].append(iv)
            values['intra_adv_sample_variance'].append(intra[indices].var(correction=1).item() if len(indices)>1 else 0.)
            values['intra_adv_zero_variance_tree_fraction'].append(float(iv == 0))
            values['same_reward_tree_fraction'].append(float(tr.max() == tr.min()))
            values['mixed_reward_tree_fraction'].append(float(tr.max() != tr.min()))
            values['exact_reward_one_tree_count'].append(float(torch.all(tr == 1)))
            values['exact_reward_zero_tree_count'].append(float(torch.all(tr == 0)))
            values['mixed_tree_count'].append(float(tr.max() != tr.min()))
            if branch_metadata is not None and 'root_slot' in branch_metadata[indices[0]]:
                slot = branch_metadata[indices[0]]['root_slot']
                values[f'root{slot}_raw_reward_population_variance'].append(tr.var(correction=0).item())
                values[f'root{slot}_actual_adv_population_variance'].append(v)
                values[f'root{slot}_intra_adv_population_variance'].append(iv)
                values[f'root{slot}_intra_adv_zero_variance_fraction'].append(float(iv == 0))
                values[f'root{slot}_mixed_reward_fraction'].append(float(tr.max() != tr.min()))
                values[f'root{slot}_same_reward_fraction'].append(float(tr.max() == tr.min()))
                for label, condition in [('exact_reward_one',torch.all(tr == 1)), ('exact_reward_zero',torch.all(tr == 0)), ('mixed',tr.max() != tr.min())]:
                    values[f'root{slot}_{label}_count'].append(float(condition))
        reward_means = torch.stack(reward_means)
        values['root_raw_reward_mean_gap'].append((reward_means.max()-reward_means.min()).item())
        means = torch.stack(means)
        weights = score.new_tensor(masses)
        weighted_mean = (means*weights).sum()
        between = (weights*(means-weighted_mean).square()).sum().item()
        weighted_within = sum(m*v for m,v in zip(masses,within))
        values['between_root_actual_adv_population_variance_equal_tree'].append(means.var(correction=0).item())
        values['between_root_actual_adv_population_variance_leaf_mass'].append(between)
        values['within_root_actual_adv_population_variance_leaf_mass'].append(weighted_within)
        values['intra_adv_population_variance_leaf_mass'].append(sum(m*v for m,v in zip(masses,intra_within)))
        values['adv_variance_decomposition_abs_error'].append(abs(a.var(correction=0).item()-weighted_within-between))
    result = {'tree_variance/'+key: (sum(items) if key.endswith('_count') else sum(items)/len(items))
              for key, items in values.items()}
    result['tree_variance/question_count'] = len(groups)
    result['tree_variance/tree_count'] = len(set(zip(uids, trees)))
    return result


@torch.no_grad()
def topd(teacher_logp, old_logp, mask, uids, alpha=.1):
    """Proximal dense rewards, exclusive future-token mean, pooled group whitening."""
    if not 0 < alpha < 1:
        raise ValueError('TOP-D requires 0 < alpha < 1')
    effective = mask.bool()
    # Zero masked inputs before subtraction: masked infinities must not create NaNs.
    d = torch.where(effective, teacher_logp - old_logp, 0.).float()
    reward = torch.logaddexp(d + math.log(alpha), d.new_full((), math.log1p(-alpha)))
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
