"""Keep actual behavior probabilities distinct from frozen HF recomputation.

This module does not assert that two numerical backends are the same policy.
The original sampled token/context audit remains required independently.
"""
import torch

ROLE = 'actual_native_stochastic_sampling_policy'


def bind_actual_behavior(data):
    """Explicitly bind PPO's legacy old_log_probs key to captured behavior LP.

    Preserve the unchanged HF recomputation in its own tensor. Never modify
    sampling_log_probs or infer behavior probabilities from HF recomputation.
    Call once, after actual HF recomputation and before optimization/balancing.
    """
    from search_r1.llm_agent.sampling_tokens import audit_batch
    proof = audit_batch(data)
    if proof['probability_role'] != 'stochastic_sampling_policy':
        raise ValueError('Training behavior binding requires actual stochastic sampling')
    if 'frozen_hf_log_probs' in data.batch or 'behavior_policy_role' in data.meta_info:
        raise ValueError('Behavior binding is immutable and cannot be repeated')
    sampled = data.batch['sampling_log_probs']
    hf = data.batch['old_log_probs']
    mask = data.batch['sampling_mask'].bool()
    if sampled.shape != hf.shape or sampled.shape != mask.shape or not mask.any():
        raise ValueError('Behavior/HF probability shape or coverage differs')
    # Numeric finite-value admission is collective at the actor gate; capture
    # and transport nonfinite values faithfully instead of silently repairing.
    data.batch['frozen_hf_log_probs'] = hf.detach().clone()
    data.batch['old_log_probs'] = sampled.detach().clone()
    data.meta_info['behavior_policy_role'] = ROLE
    data.meta_info['frozen_hf_probability_role'] = 'unchanged_hf_recomputation_diagnostic'
    return proof


def validate_behavior_tensors(data):
    """Transport/role identity; finite guards remain separate from identity."""
    if data.meta_info.get('behavior_policy_role') != ROLE:
        raise ValueError('Actual behavior denominator role missing; no HF fallback')
    if data.meta_info.get('frozen_hf_probability_role') != 'unchanged_hf_recomputation_diagnostic':
        raise ValueError('Frozen HF diagnostic role missing')
    from search_r1.llm_agent.sampling_tokens import audit_batch
    audit = audit_batch(data)
    if audit['probability_role'] != 'stochastic_sampling_policy':
        raise ValueError('Actual stochastic context/journal proof required')
    t = data.batch
    if 'frozen_hf_log_probs' not in t:
        raise ValueError('Frozen HF diagnostic probability tensor missing')
    old, sampled, hf, mask = t['old_log_probs'], t['sampling_log_probs'], t['frozen_hf_log_probs'], t['sampling_mask'].bool()
    if not all(x.is_floating_point() for x in (old, sampled, hf)):
        raise ValueError('Behavior and HF diagnostic probabilities must remain floating point')
    if old.shape != sampled.shape or hf.shape != old.shape or mask.shape != old.shape:
        raise ValueError('Behavior/HF diagnostic probability shape differs')
    # Exact captured values, including the same nonfinite representation. This
    # is transport identity, not an approximate cross-backend comparison.
    finite = torch.isfinite(sampled)
    equal = (old == sampled) | (torch.isnan(old) & torch.isnan(sampled))
    if not bool(equal[mask].all()):
        raise ValueError('PPO denominator differs from actual captured behavior')
    if not bool(finite[mask].all()):
        raise ValueError('Actual native sampling LP nonfinite; capture identity rejected')
    if not bool(torch.isfinite(hf[mask]).all()):
        from .policy_safety import NonfiniteSamplingPolicy
        raise NonfiniteSamplingPolicy('Frozen HF diagnostic selected LP nonfinite')
    return True


def hf_drift_summary(hf0, behavior, current, mask, weights):
    """Rank-local diagnostic only; masked observations never enter arithmetic."""
    valid = mask.bool()
    result = dict(baseline_vs_behavior_max_abs=None,
                  current_vs_frozen_hf_sequence_mean_abs=None, finite=False,
                  scope='rank_local', role='unchanged_hf_recomputation_diagnostic_not_behavior_denominator')
    if not bool(valid.any()):
        return result
    base, sampled, latest, w = [x.double()[valid] for x in (hf0, behavior, current, weights)]
    if not all(bool(torch.isfinite(x).all()) for x in (base, sampled, latest, w)):
        return result
    mass = w.sum()
    gap = (base-sampled).abs().max()
    drift = ((latest-base).abs()*w).sum()/mass
    if not bool(mass>0 and torch.isfinite(gap) and torch.isfinite(drift)):
        return result
    result.update(baseline_vs_behavior_max_abs=float(gap),
                  current_vs_frozen_hf_sequence_mean_abs=float(drift), finite=True)
    return result
