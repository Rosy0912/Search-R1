"""Collective rejection of invalid updates; never replace bad values with zeros."""
import json
import torch
import torch.distributed as dist


def require_finite(fields, *, stage, process_group=None, positive=(), recoverable=False, rollback_available=False):
    tests={name:torch.isfinite(value).all() for name,value in fields.items()}
    device = ('cuda:'+str(torch.cuda.current_device()) if dist.is_initialized() and dist.get_backend(process_group)=='nccl'
              else (next(iter(fields.values())).device if fields else 'cpu'))
    for name in positive:
        tests[name]=(tests[name] & (fields[name]>0).all() if name in fields else torch.zeros((),dtype=torch.bool,device=device))
    local=(torch.stack([v.to(device) for v in tests.values()]).all().to(dtype=torch.int32)
           if tests else torch.zeros((),dtype=torch.int32,device=device))
    global_ok=local.clone()
    if dist.is_initialized():
        dist.all_reduce(global_ok,op=dist.ReduceOp.MIN,group=process_group)
    if not global_ok.item():
        rank=dist.get_rank() if dist.is_initialized() else 0
        diagnostic=dict(stage=stage,rank=rank,local_failed_fields=([k for k,v in tests.items() if not v.item()] if tests else ['no_numerical_fields']),
                        failure_on_another_rank=bool(local.item()))
        print('SEARCH_G1_REJECTED_NONFINITE_UPDATE='+json.dumps(diagnostic),flush=True)
        if recoverable:
            # Post-write recovery is allowed only inside a local Adam transaction.
            if stage not in ('before_gradient_clipping', 'before_optimizer_step',
                             'before_dual_clip_exp', 'before_backward',
                             'before_fp64_gradient_clipping', 'before_fp64_clip_write',
                             'after_fp64_gradient_clipping', 'measured_after_gradient_clipping') and not (
                                 stage=='after_optimizer_write' and rollback_available):
                raise RuntimeError('Invalid recoverable numerical stage: '+stage)
            from search_g1.policy_safety import RecoverableGuard
            raise RecoverableGuard(stage=stage, reasons=['nonfinite_or_empty_effective_numerical_input'],
                                   diagnostic=diagnostic)
        raise RuntimeError('Nonfinite or empty-mask GRPO update rejected at '+stage)
