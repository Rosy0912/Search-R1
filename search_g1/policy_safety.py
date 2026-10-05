"""Unreduced Dual-Clip and collective old-policy monitoring; no rollback."""
import math
import torch
import torch.distributed as dist


def dual_clip_terms(delta, advantages, mask, epsilon, c):
    if not 0 < epsilon < 1 or not math.isfinite(c) or c <= 1 + epsilon:
        raise ValueError('Invalid Dual-Clip parameters')
    valid = mask.bool()
    d = torch.where(valid, delta, 0.).double()
    a = torch.where(valid, advantages, 0.).double()
    if not torch.isfinite(d).all() or not torch.isfinite(a).all():
        raise ValueError('Nonfinite effective Dual-Clip input')
    bounded = torch.where(a > 0, d.clamp_max(math.log1p(epsilon)),
                          torch.where(a < 0, d.clamp(math.log1p(-epsilon), math.log(c)),
                                      torch.zeros_like(d)))
    return -a * torch.exp(bounded)


@torch.no_grad()
def monitor(delta, advantages, mask, weights, *, c, kl_log_bound, target_kl,
            negative_tail_limit, group=None):
    """Global weighted k3 and exact negative-token quantiles, matching collectives.

    Invalid/over-bound samples generate finite diagnostics and a rejection flag;
    their substituted zero is NEVER reported as a true complete KL estimate.
    """
    if not (math.isfinite(target_kl) and target_kl > 0 and
            math.isfinite(negative_tail_limit) and 0 <= negative_tail_limit <= 1 and
            math.isfinite(kl_log_bound) and kl_log_bound > 0 and math.isfinite(c) and c > 1):
        raise ValueError('Explicit finite calibrated thresholds required')
    v = mask.bool();d = torch.where(v, delta, 0.).double()
    a = torch.where(v, advantages, 0.).double();w = torch.where(v, weights, 0.).double()
    bad = v & (~torch.isfinite(d) | ~torch.isfinite(a) | ~torch.isfinite(w) | (w < 0))
    over = v & (d > kl_log_bound)
    safe_d = torch.where(bad | over, torch.zeros_like(d), d)
    safe_w = torch.where(bad, torch.zeros_like(w), w)
    negative = v & ~bad & (a < 0);tail = negative & (d > math.log(c))
    k3 = torch.expm1(safe_d) - safe_d
    values = torch.stack([safe_w.sum(), (k3 * safe_w).sum(),
                          safe_w[negative].sum(), safe_w[tail].sum(),
                          bad.sum().double(), over.sum().double(),v.sum().double(),
                          negative.sum().double(),tail.sum().double()])
    distributed = dist.is_initialized()
    if distributed: dist.all_reduce(values, group=group)
    mass, numerator, nmass, tmass, invalid, oversized, count, ncount, tcount = values.tolist()
    finite_totals = all(math.isfinite(x) for x in values.tolist())
    samples = d[negative & torch.isfinite(d)].flatten()
    if distributed:
        size = torch.tensor([samples.numel()], dtype=torch.long, device=d.device)
        sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size(group))]
        dist.all_gather(sizes, size, group=group)
        width = max(1, max(int(x.item()) for x in sizes))
        padded = torch.zeros(width, dtype=torch.float64, device=d.device)
        padded[:samples.numel()] = samples
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded, group=group)
        samples = torch.cat([x[:int(n.item())] for x,n in zip(gathered,sizes)])
    quantiles = {k: None for k in ('negative_log_ratio_p95','negative_log_ratio_p99','negative_log_ratio_max')}
    if samples.numel():
        quantiles.update(negative_log_ratio_p95=float(torch.quantile(samples,.95,interpolation='nearest')),
                         negative_log_ratio_p99=float(torch.quantile(samples,.99,interpolation='nearest')),
                         negative_log_ratio_max=float(samples.max()))
    complete = finite_totals and invalid == 0 and oversized == 0 and mass > 0 and count > 0
    kl = numerator / mass if complete else None
    fraction = tmass / nmass if finite_totals and nmass > 0 else 0.
    reasons = []
    if not finite_totals: reasons.append('nonfinite_global_totals')
    if invalid: reasons.append('nonfinite_or_negative_effective_input')
    if oversized: reasons.append('delta_above_kl_log_bound')
    if mass <= 0 or count <= 0: reasons.append('empty_effective_samples')
    if kl is not None and kl > target_kl: reasons.append('old_policy_kl')
    if fraction > negative_tail_limit: reasons.append('negative_advantage_tail')
    return dict(allowed=not reasons,reasons=reasons,sampled_old_policy_kl=kl,
                kl_estimate_complete=complete,effective_weight_mass=mass if finite_totals else None,
                effective_token_count=int(count),negative_token_count=int(ncount),
                negative_tail_count=int(tcount),negative_tail_count_fraction=tcount/ncount if ncount else 0.,
                negative_tail_weight_fraction=fraction,dual_clip_saturated_objective_weight_fraction=tmass/mass if finite_totals and mass>0 else None,
                invalid_count=int(invalid),overflow_count=int(oversized),quantile_method='exact global nearest-order statistic',**quantiles)


def bind_thresholds(config, *, group=None):
    """Reject missing/unbound calibration collectively before any update."""
    import json,hashlib,os
    from pathlib import Path
    errors=[];artifact={}
    try:
        base=Path(os.environ['SEARCH_G1_BASE']).resolve()
        path=Path(config.calibration_manifest).resolve()
        if path != base/'logs/policy-safety-calibration.json':raise ValueError('Unbound calibration path')
        artifact=json.loads(path.read_text())
        run=json.loads((base/'logs/run-manifest.json').read_text())
        if artifact['accepted'] is not True or artifact['role']!='healthy_pretraining_policy_calibration':raise ValueError('Not healthy calibrated admission')
        if artifact['run_id']!=run['run_id'] or artifact['initial_model_sha256']!=run['initial_model_sha256'] or artifact['source_sha256']!=run['source_sha256']:raise ValueError('Calibration run/model/source differs')
        if artifact['input_sha256']!=run['input_sha256']:raise ValueError('Calibration data differs')
        if artifact['dual_clip']!=float(config.dual_clip) or artifact['target_old_policy_kl']!=float(config.target_old_policy_kl) or artifact['kl_log_bound']!=float(config.kl_log_bound):raise ValueError('Calibration thresholds differ')
        tail=float(artifact['negative_tail_limit'])
        if config.negative_tail_limit is not None and float(config.negative_tail_limit)!=tail:raise ValueError('Tail threshold differs')
        if not math.isfinite(tail) or not 0<=tail<=1:raise ValueError('Invalid calibrated tail limit')
        if artifact['actual_probability_gate_accepted'] is not True or artifact['actual_native_short_segment_accepted'] is not True:raise ValueError('Missing actual native calibration acceptance')
        def proof(relative,expected_hash):
            candidate=(base/relative).resolve()
            if base not in candidate.parents:raise ValueError('Proof outside bound run')
            content=candidate.read_bytes()
            if hashlib.sha256(content).hexdigest()!=expected_hash:raise ValueError('Proof hash mismatch')
            evidence=json.loads(content)
            if evidence['accepted'] is not True or evidence['run_id']!=run['run_id'] or evidence['source_sha256']!=run['source_sha256']:raise ValueError('Unbound actual proof')
            return evidence
        probability=proof(artifact['probability_proof'],artifact['probability_proof_sha256'])
        if probability['phase']!='initial' or probability['actual_optimizer_writes']!=0 or probability['ranks']!=list(range(8)):raise ValueError('Not complete initial probability proof')
        segment=proof(artifact['short_segment_proof'],artifact['short_segment_proof_sha256'])
        if segment['initial_model_sha256']!=run['initial_model_sha256'] or segment['actual_optimizer_writes']<1 or segment['all_parameters_and_adam_finite'] is not True:raise ValueError('No healthy actual segment proof')
        if segment['monitor_definition']!='four_length_stratified_trajectories_per_rank_two_row_micro_weighting' or segment['monitor_identity_before']!=segment['monitor_identity_after'] or segment['old_logprob_sha256_before']!=segment['old_logprob_sha256_after']:raise ValueError('Monitor or old policy changed')
        if segment['negative_tail_limit']!=tail or segment['target_old_policy_kl']!=float(config.target_old_policy_kl):raise ValueError('Threshold proof mismatch')
        serialized=json.dumps(artifact,sort_keys=True,allow_nan=False).encode()
    except (OSError,ValueError,KeyError,TypeError) as failure:
        errors.append(str(failure));serialized=b'REJECTED'
    device='cuda:'+str(torch.cuda.current_device()) if dist.is_initialized() and dist.get_backend(group)=='nccl' else 'cpu'
    ok=torch.tensor(int(not errors),device=device,dtype=torch.int32)
    digest=torch.tensor(list(hashlib.sha256(serialized).digest()),device=device,dtype=torch.uint8)
    if dist.is_initialized():
        dist.all_reduce(ok,op=dist.ReduceOp.MIN,group=group)
        hashes=[torch.empty_like(digest) for _ in range(dist.get_world_size(group))]
        dist.all_gather(hashes,digest,group=group)
        if any(not torch.equal(x,digest) for x in hashes):ok.zero_()
    if not ok.item():raise RuntimeError('Policy safety calibration admission rejected on all ranks: '+repr(errors))
    return tail,hashlib.sha256(serialized).hexdigest()


def bind_actual_config(actor_config, group=None):
    """Every actor rank participates, including disabled ranks, before branching."""
    import json,hashlib,os
    from pathlib import Path
    from omegaconf import OmegaConf
    ok=True
    try:
        document=OmegaConf.to_container(actor_config,resolve=True)
        serialized=json.dumps(document,sort_keys=True,allow_nan=False).encode()
        s=document.get('policy_safety')
        if s is not None and not isinstance(s.get('enabled'),bool):
            ok=False
        if os.environ.get('SEARCH_G1_PROFILE')=='common8':
            run=json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
            if run.get('policy_safety_required') is True and (s is None or s.get('enabled') is not True):
                ok=False
        if s and s['enabled']:
            epsilon=float(document['clip_ratio']);c=float(s['dual_clip'])
            ok=ok and (isinstance(s['enabled'],bool) and 0<epsilon<1 and math.isfinite(c) and c>1+epsilon
                and math.isfinite(float(s['target_old_policy_kl'])) and s['target_old_policy_kl']>0
                and math.isfinite(float(s['kl_log_bound'])) and s['kl_log_bound']>0
                and s['monitor_trajectories_per_rank']==4 and not document['use_dynamic_bsz'])
    except (ValueError,TypeError,KeyError,OSError):
        ok=False;serialized=b'INVALID'
    device='cuda:'+str(torch.cuda.current_device()) if dist.is_initialized() and dist.get_backend(group)=='nccl' else 'cpu'
    flag=torch.tensor(int(ok),device=device,dtype=torch.int32)
    digest=torch.tensor(list(hashlib.sha256(serialized).digest()),device=device,dtype=torch.uint8)
    if dist.is_initialized():
        dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=group)
        hashes=[torch.empty_like(digest) for _ in range(dist.get_world_size(group))]
        dist.all_gather(hashes,digest,group=group)
        if any(not torch.equal(digest,x) for x in hashes):flag.zero_()
    if not flag.item():raise RuntimeError('Actual actor/safety configuration differs or is invalid; all ranks rejected')
