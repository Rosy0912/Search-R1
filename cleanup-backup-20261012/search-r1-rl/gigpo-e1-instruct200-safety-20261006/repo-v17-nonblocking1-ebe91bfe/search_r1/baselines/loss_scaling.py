"""Detached, global-mini numeric representation of the strict GiGPO loss.

This bounds output derivatives, not the network Jacobian. Real gradients and
the original loss/norm must still pass the actor's finite checks.
"""
import math
import torch
import torch.distributed as dist


@torch.no_grad()
def describe_invalid_joint_probe(old, current, advantages, mask, weights, step_ids,
                                 cliprange, limit=32):
    """Bounded detached diagnostics; never used to decide validity or scale.

    The original probe remains authoritative. Preserve action/advantage
    correspondence so an inactive overflowing ratio is distinguishable from
    an active derivative. Nonfinite values use strings in diagnostic JSON.
    """
    def number(value):
        value = float(value)
        return value if math.isfinite(value) else str(value)

    result = dict(reasons={}, actions=[], omitted_actions=0)

    def reason(name):
        result['reasons'][name] = result['reasons'].get(name, 0) + 1

    valid = mask.bool()
    if (step_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or step_ids.shape != mask.shape):
        reason('step_ids_dtype_or_shape')
        return result
    for name, tensor in [('old', old), ('current', current),
                         ('advantages', advantages), ('weights', weights)]:
        if not bool(torch.isfinite(tensor[valid]).all()):
            reason('nonfinite_' + name)
    if bool((weights[valid] < 0).any()):
        reason('negative_weight')
    delta = torch.where(valid, current, 0.) - torch.where(valid, old, 0.)
    for row in range(len(delta)):
        for action in torch.unique(step_ids[row][valid[row]]):
            selected = valid[row] & (step_ids[row] == action)
            adv = advantages[row][selected]
            value = delta[row][selected].sum().double()
            causes = []
            if bool(action < 0):
                causes.append('negative_action_id')
            if not torch.allclose(adv, adv[:1].expand_as(adv)):
                causes.append('inconsistent_action_advantage')
            if not bool(torch.isfinite(value)):
                causes.append('nonfinite_joint_logratio')
            exact_zero = bool((adv == 0).all())
            if not exact_zero and not bool(torch.isfinite(value.exp())):
                causes.append('nonfinite_raw_joint_ratio')
            for cause in causes:
                reason(cause)
            if causes:
                if len(result['actions']) < limit:
                    a = adv[0]
                    active = bool((a < 0 and value >= math.log(1 - cliprange))
                                  or (a > 0 and value <= math.log(1 + cliprange)))
                    result['actions'].append(dict(row=row, action=int(action),
                        tokens=int(selected.sum()), logratio=number(value),
                        advantage_first=number(a), advantage_min=number(adv.min()),
                        advantage_max=number(adv.max()), derivative_active=active,
                        positive_weight=bool((weights[row][selected] > 0).any()),
                        reasons=causes))
                else:
                    result['omitted_actions'] += 1
    return result


def write_probe_diagnostic(record):
    """Separate metadata stream; no tensors, collectives or model operations."""
    import json
    import os
    from pathlib import Path
    root = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
    if not root:
        return
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
        directory = Path(root) / 'numeric'
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(record, rank=rank, pid=os.getpid(),
            run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
            commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
            config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
            data_identity=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'))
        with (directory / f'range-probe-{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
            stream.flush()
    except Exception as exc:
        # Metadata must never interrupt the existing collective sequence.
        # Missing receipts or this explicit error prevent independent acceptance.
        import sys
        try:
            print('GiGPO probe diagnostic write failed: ' + repr(exc), file=sys.stderr, flush=True)
        except Exception:
            pass  # A closed stderr cannot interrupt collective/backward order.


@torch.no_grad()
def active_joint_bound(old, current, advantages, mask, weights, step_ids, cliprange):
    """Return (local bound, invalid flag); callers synchronize before raising.

    Use the same native summation as weighted_policy_loss. Clipped and zero
    advantage branches cannot drive the exponent's backward derivative.
    """
    valid = mask.bool()
    if (step_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or step_ids.shape != mask.shape):
        return (torch.full((), -float('inf'), dtype=torch.float64, device=current.device),
                torch.ones((), dtype=torch.int32, device=current.device))
    bad = any(not bool(torch.isfinite(v[valid]).all())
              for v in (old, current, advantages, weights))
    bad = bad or bool((weights[valid] < 0).any())
    bound = torch.full((), -float('inf'), dtype=torch.float64, device=current.device)
    delta = torch.where(valid, current, 0.) - torch.where(valid, old, 0.)
    for row in range(len(delta)):
        for action in torch.unique(step_ids[row][valid[row]]):
            selected = valid[row] & (step_ids[row] == action)
            adv = advantages[row][selected]
            bad = bad or bool(action < 0) or not torch.allclose(adv, adv[:1].expand_as(adv))
            value = delta[row][selected].sum().double()
            bad = bad or not bool(torch.isfinite(value))
            # Exact-zero PG actions do not require exp(L). The original L and
            # all inputs still must be finite, and nonzero actions retain the
            # original strict ratio check, including clipped branches.
            exact_zero = bool((adv == 0).all())
            bad = bad or (not exact_zero and not bool(torch.isfinite(value.exp())))
            positive_weight = bool((weights[row][selected] > 0).any())
            a = adv[0]
            active = ((a < 0 and value >= math.log(1 - cliprange))
                      or (a > 0 and value <= math.log(1 + cliprange)))
            if positive_weight and active and torch.isfinite(value):
                bound = torch.maximum(bound, value)
    return bound, torch.tensor(int(bad), dtype=torch.int32, device=current.device)


@torch.no_grad()
def coordinated_scale(bound, bad, group=None, margin=40.):
    """Every rank executes badflag then MAX, including empty/invalid ranks."""
    bound, bad = bound.clone(), bad.clone()
    if dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(bound, op=dist.ReduceOp.MAX, group=group)
    if bool(bad):
        raise ValueError('Invalid GiGPO range probe; all ranks refuse optimizer step')
    exponent = torch.minimum(torch.zeros_like(bound), margin - bound)
    scale = exponent.exp().detach()
    if not torch.isfinite(scale) or not 0 < scale <= 1 or not torch.isfinite(1. / scale):
        raise ValueError('Unrepresentable positive GiGPO loss scale')
    return scale


class TransparentProbe:
    """Restore mutable forward state; changed FSDP boundaries are failures.

    Parameters are not copied or repaired here. Native acceptance separately
    checks their full identities. A no-grad forward must never mutate them.
    """
    def __init__(self, module, manager=None):
        self.module, self.manager = module, manager

    def _boundary(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(self.module, FSDP):
            from torch.distributed.fsdp._common_utils import TrainingState
            self.module._assert_state(TrainingState.IDLE)
        def layout(p):
            return (tuple(p.shape), str(p.dtype), str(p.device))
        return {name: (layout(p), layout(p._local_shard) if hasattr(p, '_local_shard') else None)
                for name, p in self.module.named_parameters()}

    def __enter__(self):
        from .recovery import rng_state, cpu_snapshot
        self.rng = rng_state()
        self.invalid = False
        try:
            self.boundary = self._boundary()
        except Exception:
            self.boundary = None
            self.invalid = True
        self.buffers = {name: cpu_snapshot(buffer) for name, buffer in self.module.named_buffers()}
        self.buffer_devices = {name: buffer.device for name, buffer in self.module.named_buffers()}
        self.modes = {name: submodule.training for name, submodule in self.module.named_modules()}
        self.manager_rng = {key: cpu_snapshot(getattr(self.manager, key, None))
                            for key in ('gen_random_states', 'torch_random_states')}
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from .recovery import restore_rng
        try:
            changed = self._boundary() != self.boundary
        except Exception:
            changed = True
        buffers = dict(self.module.named_buffers())
        changed = changed or buffers.keys() != self.buffers.keys()
        with torch.no_grad():
            for name, value in self.buffers.items():
                if (name in buffers and buffers[name].shape == value.shape
                        and buffers[name].dtype == value.dtype
                        and buffers[name].device == self.buffer_devices[name]):
                    buffers[name].copy_(value.to(buffers[name].device))
                else:
                    changed = True
                    parent_name, _, attribute = name.rpartition('.')
                    parent = dict(self.module.named_modules()).get(parent_name)
                    if parent is not None:
                        parent._buffers[attribute] = value.to(self.buffer_devices[name]).clone()
        for name, submodule in self.module.named_modules():
            if name in self.modes:
                submodule.training = self.modes[name]
            else:
                changed = True
        if self.manager is not None:
            for key, value in self.manager_rng.items():
                setattr(self.manager, key, value)
        restore_rng(self.rng)
        self.invalid = self.invalid or changed
        return False


def coordinated_badflag(bad, device, group=None):
    flag = torch.tensor(int(bad), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    if bool(flag):
        raise ValueError('GiGPO real forward differs from probe or has invalid loss on a rank')
