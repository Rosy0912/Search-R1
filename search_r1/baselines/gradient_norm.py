"""Same FSDP L2 clipping, with FP64 reductions to avoid FP32 square overflow.

Parameter partitioning and collective counting follow Torch 2.4 FSDP's
clip_grad_norm_. No gradient or update is discarded. Chunking bounds the
temporary FP64 allocation even for a large flattened parameter.
"""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class SynchronizedGradientNormRefusal(ValueError):
    """Nonfinite refusal after a completed FSDP bad-gradient MAX reduction.

    Only a group covering the default worker world can authorize the worker's
    all-rank partial-state path. Setup/dtype/collective failures stay ordinary
    exceptions and carry no completed-reduction evidence.
    """
    def __init__(self, message, *, matched_all_ranks=False):
        super().__init__(message)
        self.matched_all_ranks = matched_all_ranks


def _norm64(params, device, chunk_size=262144, uniform=True):
    grads = [p.grad for p in params if p.grad is not None]
    if uniform and len({g.dtype for g in grads}) > 1:
        raise ValueError('Requires uniform gradient dtype in each FSDP partition')
    norms = []
    for grad in grads:
        flat = grad.detach().reshape(-1)
        for chunk in flat.split(chunk_size):
            norms.append(torch.linalg.vector_norm(chunk.to(torch.float64), 2).to(device))
    return torch.linalg.vector_norm(torch.stack(norms), 2) if norms else torch.zeros((), device=device, dtype=torch.float64)


def _failure(grads, norm):
    # Diagnostic disk errors must not replace the optimizer refusal.
    try:
        _failure_report(grads, norm)
    except Exception:
        pass


def _failure_report(grads, norm):
    root = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
    if not root:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    counts = []
    for grad in grads:
        finite_count, maximum = 0, None
        for chunk in grad.detach().reshape(-1).split(262144):
            finite = torch.isfinite(chunk)
            valid = chunk[finite]
            finite_count += int(finite.sum())
            if valid.numel():
                value = float(valid.abs().max())
                maximum = value if maximum is None else max(maximum, value)
        counts.append(dict(count=grad.numel(), finite=finite_count, max_abs=maximum))
    p = Path(root) / 'numeric' / f'gradient-failure-{rank}.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(rank=rank, pid=os.getpid(), gradient_tensors=counts,
        global_norm64=float(norm), norm_scope='FSDP global sharded plus once-counted nonsharded'), allow_nan=True))


@torch.no_grad()
def clip_grad_norm64(module, max_norm):
    """L2 max_norm/(global_norm+1e-6), cast coefficient as native FSDP does."""
    if max_norm < 0:
        raise ValueError('Negative max_norm')
    if isinstance(module, FSDP):
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        from torch.distributed.fsdp._common_utils import TrainingState
        _lazy_init(module, module)
        if not module._is_root:
            raise RuntimeError('Gradient clipping requires root FSDP')
        module._assert_state(TrainingState.IDLE)
        sharded, nonsharded, grads = set(), set(), []
        sharded_order, nonsharded_order = [], []
        gradient_ids = set()
        for handle in module._all_handles:
            target = sharded if handle.uses_sharded_strategy else nonsharded
            ordered = sharded_order if handle.uses_sharded_strategy else nonsharded_order
            params = handle.flat_param._params if handle._use_orig_params else [handle.flat_param]
            for param in params:
                if param not in target:
                    target.add(param)
                    ordered.append(param)
                if param.grad is not None and id(param.grad) not in gradient_ids:
                    grads.append(param.grad)
                    gradient_ids.add(id(param.grad))
        for param in module.parameters():
            if param not in sharded and param not in nonsharded:
                nonsharded.add(param)
                nonsharded_order.append(param)
                if param.grad is not None and id(param.grad) not in gradient_ids:
                    grads.append(param.grad)
                    gradient_ids.add(id(param.grad))
        device = module.compute_device
        sharded_norm = _norm64(sharded_order, device)
        nonsharded_norm = _norm64(nonsharded_order, device, uniform=bool(sharded))
        # All ranks perform this reduction, including NO_SHARD/empty ranks.
        # A corrupt replicated gradient on one rank must reject on every rank.
        bad = (~(torch.isfinite(sharded_norm) & torch.isfinite(nonsharded_norm))).to(torch.int32)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=module.process_group)
        if bool(bad):
            _failure(grads, torch.full((), float('nan'), dtype=torch.float64, device=device))
            raise SynchronizedGradientNormRefusal(
                'Nonfinite actor gradient elements; optimizer step refused on all FSDP ranks',
                matched_all_ranks=(dist.get_world_size(module.process_group) == dist.get_world_size()))
        if sharded:
            norm = sharded_norm.square()
            dist.all_reduce(norm, group=module.process_group)
            if nonsharded:
                norm += nonsharded_norm.square()
            norm = norm.sqrt()
        else:
            norm = nonsharded_norm
        if module.cpu_offload.offload_params:
            norm = norm.cpu()
    else:
        params = list(module.parameters())
        grads = [p.grad for p in params if p.grad is not None]
        device = grads[0].device if grads else torch.device('cpu')
        norm = _norm64(params, device, uniform=False)
    if not torch.isfinite(norm):
        _failure(grads, norm)
        raise ValueError('Nonfinite actor gradient norm; optimizer step refused')
    coefficient = (float(max_norm) / (norm + 1e-6)).clamp(max=1.)
    for grad in grads:
        grad.mul_(coefficient.to(grad.device, grad.dtype))
    return norm
