"""BAPO sequence-mean safety: collective rejection, retained Adam writes.

Reference: Search-G1 policy_safety.py / _monitor_old_policy, read-only.
Unlike that reference's small monitor, this actor monitors the FULL frozen rollout
and uses each nonempty row's reciprocal supervised length as the token weight.
"""
import hashlib
import json
import math
import torch
import torch.distributed as dist


class NonfiniteSamplingPolicy(ValueError):
    """Actual effective selected LP is nonfinite; structural admission passed."""
    pass


class SkipOptimizerOpportunity(ValueError):
    """All ranks have rejected this mini; no live write has occurred."""
    pass


class PartialPolicyUpdate(RuntimeError):
    def __init__(self, event):
        self.event = event
        super().__init__('BAPO partial update: ' + json.dumps(event, sort_keys=True, allow_nan=False))


def distributed(group=None):
    return dist.is_available() and dist.is_initialized()


def collective_device(group=None):
    return torch.device('cuda', torch.cuda.current_device()) if distributed(group) and dist.get_backend(group) == 'nccl' else torch.device('cpu')


def consensus(ok, group=None):
    flag = torch.tensor(int(ok), device=collective_device(group), dtype=torch.int32)
    if distributed(group):
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def persist_rank_update_event(path, event, group=None):
    """IO failure is collective fatal feedback, never a rank-local escape."""
    from pathlib import Path
    import os
    error=None
    try:
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as handle:
            handle.write(json.dumps(event,indent=2,default=str));handle.flush();os.fsync(handle.fileno())
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    except Exception as caught:error=repr(caught)
    if consensus(error is None,group):return event,None
    event=dict(event)
    if event.get('status')=='INFRASTRUCTURE_FAULT':
        # A second IO failure must not erase original commit-copy truth.
        event['additional_infrastructure_faults']=list(event.get('additional_infrastructure_faults',[]))+[
            dict(reason='rank_update_ledger_io_fault',ledger_io_error=error)]
    else:
        event.update(status='INFRASTRUCTURE_FAULT',fatal=True,update_complete=False,
            attempt_complete=False,postabort=True,reason='rank_update_ledger_io_fault',
            confirmed_previous_optimizer_steps=event['optimizer_steps'],
            failed_commit_progress=dict(local_commit_complete=False,ledger_io_error=error),
            current_state_requires_native_save=True)
    events=[None]*dist.get_world_size(group) if distributed(group) else [event]
    if distributed(group):dist.all_gather_object(events,event,group=group)
    return event,events


def agree_document(document, ok=True, group=None):
    encoded = json.dumps(document, sort_keys=True, allow_nan=False).encode()
    digest = torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8, device=collective_device(group))
    if distributed(group):
        hashes = [torch.empty_like(digest) for _ in range(dist.get_world_size(group))]
        dist.all_gather(hashes, digest, group=group)
        ok = ok and all(torch.equal(x, digest) for x in hashes)
    return consensus(ok, group)


def finite_inputs(old, current, advantages, mask, weights, group=None):
    """Same collective on every rank, BEFORE subtraction or exponentiation."""
    valid = mask.bool()
    ok = torch.isfinite(mask).all().item() and bool((mask >= 0).all())
    for value in (old, current, advantages, weights):
        ok = ok and bool(torch.isfinite(value[valid]).all())
    ok = ok and bool((weights[valid] >= 0).all())
    if not consensus(ok, group):
        raise SkipOptimizerOpportunity('Nonfinite policy input on at least one rank')
    # Double subtraction avoids fp32 finite operands overflowing before checking.
    delta = torch.where(valid, current, 0.).double() - torch.where(valid, old, 0.).double()
    if not consensus(bool(torch.isfinite(delta[valid]).all()), group):
        raise SkipOptimizerOpportunity('Nonfinite policy subtraction on at least one rank')
    return delta


def dual_clip_surrogate(delta, advantages, mask, epsilon=.2, c=3., log_ratio_clip=20.):
    if not (0 < epsilon < 1 and math.isfinite(c) and c > 1 + epsilon and math.isfinite(log_ratio_clip) and 0 < log_ratio_clip <= 20):
        raise ValueError('Invalid Dual-Clip / bounded PG parameters')
    # Inputs must already pass caller's collective finite/subtraction guard.
    # Float64 also prevents half-precision exponent/c*A intermediates overflowing.
    a = torch.where(mask.bool(), advantages, 0.).double()
    d = torch.where(mask.bool(), delta, 0.).double().clamp_max(log_ratio_clip)
    ratio = d.exp()
    ppo = torch.maximum(-a * ratio, -a * ratio.clamp(1-epsilon, 1+epsilon))
    return torch.where(a < 0, torch.minimum(ppo, -c*a), ppo)


@torch.no_grad()
def monitor(old, current, advantages, mask, weights, *, c, kl_log_bound, target_kl, negative_tail_limit, group=None):
    """k3 under old sampled rows; tail denominator is ALL negative weight."""
    valid = mask.bool()
    inputs_ok = bool(torch.isfinite(mask).all()) and bool((mask >= 0).all())
    for x in (old, current, advantages, weights):
        inputs_ok = inputs_ok and bool(torch.isfinite(x[valid]).all())
    inputs_ok = inputs_ok and bool((weights[valid] >= 0).all())
    inputs_ok = consensus(inputs_ok, group)  # BEFORE any subtraction
    if not inputs_ok:
        return dict(allowed=False, reasons=['nonfinite_or_negative_effective_input'], kl_estimate_complete=False,
                    sampled_old_policy_kl=None, negative_tail_weight_fraction=None)
    d = torch.where(valid, current, 0.).double() - torch.where(valid, old, 0.).double()
    subtraction_ok = consensus(bool(torch.isfinite(d[valid]).all()), group)
    if not subtraction_ok:
        return dict(allowed=False, reasons=['nonfinite_subtraction'], kl_estimate_complete=False,
                    sampled_old_policy_kl=None, negative_tail_weight_fraction=None)
    w = torch.where(valid, weights, 0.).double()
    a = torch.where(valid, advantages, 0.).double()
    over = valid & (d > kl_log_bound)
    safe_d = torch.where(over, 0., d)
    negative = valid & (a < 0)
    tail = negative & (d > math.log(c))
    values = torch.stack([w.sum(), ((torch.expm1(safe_d)-safe_d)*w).sum(),
                          w[negative].sum(), w[tail].sum(), over.sum().double(), valid.sum().double()])
    if distributed(group):
        dist.all_reduce(values, group=group)
    mass, numerator, negative_mass, tail_mass, oversized, count = values.tolist()
    totals_ok = all(math.isfinite(x) for x in values.tolist())
    complete = totals_ok and oversized == 0 and mass > 0 and count > 0
    kl = numerator/mass if complete else None
    fraction = tail_mass/negative_mass if totals_ok and negative_mass > 0 else 0.
    reasons = []
    if not totals_ok: reasons.append('nonfinite_global_totals')
    if oversized: reasons.append('delta_above_kl_log_bound')
    if mass <= 0 or count <= 0: reasons.append('empty_effective_samples')
    if kl is not None and kl > target_kl: reasons.append('old_policy_kl')
    if negative_tail_limit is not None and fraction > negative_tail_limit: reasons.append('negative_advantage_tail')
    return dict(allowed=not reasons, reasons=reasons, kl_estimate_complete=complete,
                sampled_old_policy_kl=kl, negative_tail_weight_fraction=fraction,
                effective_weight_mass=mass if totals_ok else None,
                weighted_k3_numerator=numerator if totals_ok else None,
                negative_effective_weight_mass=negative_mass if totals_ok else None,
                tail_effective_weight_mass=tail_mass if totals_ok else None,
                overflow_count=int(oversized), effective_token_count=int(count))


def tensor_identity(batch):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in batch.items()}


def bind_config(config, layout, group=None):
    """Disabled is backward compatible; enabled admission is collective."""
    from omegaconf import OmegaConf
    try:
        document = OmegaConf.to_container(config, resolve=True)
        safety = document.get('policy_safety') or {'enabled': False}
        enabled = safety.get('enabled', False)
        ok = type(enabled) is bool
        if enabled:
            ok = ok and document.get('baseline_method') == 'bapo' and document.get('loss_reduction') == 'sequence_mean'
            ok = ok and document.get('state_masking') is True and not document.get('use_dynamic_bsz') and document.get('ppo_epochs') == 1
            ok = ok and safety['dual_clip'] == 3 and document['clip_ratio'] == .2
            ok = ok and 0 < float(safety['kl_log_bound']) <= 30 and 0 < float(safety['target_old_policy_kl'])
            ok = ok and all(math.isfinite(float(safety[k])) for k in ('kl_log_bound','target_old_policy_kl'))
        source_sha256 = {}
        if enabled:
            # One authoritative list binds production, diagnostics and every rank.
            from .fresh200_contract import critical_source_hashes
            source_sha256 = critical_source_hashes()
        binding = dict(config=document, layout=layout, source_sha256=source_sha256)
    except Exception:
        ok=False; enabled=False; binding={'invalid': True}
    if not agree_document(binding, ok, group):
        raise ValueError('Actual actor safety configuration / layout rejected collectively')
    return safety if enabled else None
