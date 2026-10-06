"""Shadow the native AdamW implementation; finite collective admission precedes live writes."""
import copy
import torch

class OptimizerCommitFault(RuntimeError):
    def __init__(self, progress):
        self.commit_progress=progress
        super().__init__('Infrastructure fault during live optimizer commit: '+str(progress))

def infrastructure_fault(stage,error):
    return OptimizerCommitFault(dict(stage=stage,error=str(error),parameter_copies=0,
        state_entries=0,local_commit_complete=False))

def clone_tree(value):
    if torch.is_tensor(value): return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value,dict): return {k:clone_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [clone_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(clone_tree(v) for v in value)
    return copy.deepcopy(value)

def finite_tree(value):
    if torch.is_tensor(value): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite_tree(v) for v in value)
    if isinstance(value,float): return __import__('math').isfinite(value)
    return True

def transactional_adam_step(optimizer, group=None, consensus_fn=None):
    if consensus_fn is None:
        from .policy_safety import consensus
        consensus_fn=lambda value:consensus(value,group)
    # This checkout uses native torch AdamW on FSDP local flat parameters.
    # Custom optimizers/hooks could observe or mutate live state and cannot be shadowed safely.
    supported=type(optimizer) is torch.optim.AdamW
    supported=supported and not any(getattr(optimizer,name,{}) for name in ('_optimizer_step_pre_hooks','_optimizer_step_post_hooks'))
    import torch.optim.optimizer as optimizer_module
    supported=supported and not any(getattr(optimizer_module,name,{}) for name in ('_global_optimizer_pre_hooks','_global_optimizer_post_hooks'))
    supported=supported and not any(hasattr(optimizer,name) for name in ('grad_scale','found_inf'))
    if not consensus_fn(supported): raise infrastructure_fault('unsupported_implementation','Unsupported optimizer transaction implementation/hooks/AMP metadata')
    params=list(dict.fromkeys(p for g in optimizer.param_groups for p in g['params']))
    storage_keys=[(str(p.device),p.untyped_storage().data_ptr()) for p in params if p.numel()]
    if not consensus_fn(len(set(storage_keys))==len(storage_keys)):
        raise infrastructure_fault('storage_alias','Distinct optimizer parameters share storage; unsupported transaction alias')
    live_ok=all(bool(torch.isfinite(p).all()) for p in params) and finite_tree(optimizer.state)
    if not consensus_fn(live_ok): raise infrastructure_fault('live_state','Live optimizer/model already nonfinite; safe continuation impossible')
    grad_ok=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
    if not consensus_fn(grad_ok): return dict(committed=False,reason='nonfinite_candidate_gradient')
    built=False;error=None
    try:
        mapping={p:torch.nn.Parameter(p.detach().clone(memory_format=torch.preserve_format),requires_grad=p.requires_grad) for p in params}
        for p,q in mapping.items(): q.grad=None if p.grad is None else p.grad.detach().clone(memory_format=torch.preserve_format)
        groups=[dict(clone_tree({k:v for k,v in g.items() if k!='params'}),params=[mapping[p] for p in g['params']]) for g in optimizer.param_groups]
        candidate=torch.optim.AdamW(groups)
        candidate.defaults=clone_tree(optimizer.defaults)
        for p,state in optimizer.state.items(): candidate.state[mapping[p]]=clone_tree(state)
        built=True
    except Exception as exc: error=str(exc)
    if not consensus_fn(built): raise infrastructure_fault('shadow_prepare',error)
    stepped=False
    try: candidate.step();stepped=True
    except Exception as exc: error=str(exc)
    if not consensus_fn(stepped): raise infrastructure_fault('shadow_step',error)
    finite=all(bool(torch.isfinite(p).all()) for p in mapping.values()) and finite_tree(candidate.state)
    if not consensus_fn(finite): return dict(committed=False,reason='nonfinite_candidate_parameters_or_optimizer_state')
    # Allocate all new state dictionaries before changing any live tensor.
    ready=False
    try:
        states={p:candidate.state[q] for p,q in mapping.items() if q in candidate.state}
        ready=True
    except Exception as exc: error=str(exc)
    if not consensus_fn(ready): raise infrastructure_fault('commit_prepare',error)
    progress=dict(stage='live_commit',parameter_copies=0,state_entries=0,local_commit_complete=False)
    try:
        with torch.no_grad():
            for p,q in mapping.items(): p.copy_(q);progress['parameter_copies']+=1
            for p,state in states.items(): optimizer.state[p]=state;progress['state_entries']+=1
        progress['local_commit_complete']=True
    except Exception as exc: progress['error']=str(exc)
    if not consensus_fn(progress['local_commit_complete']): raise OptimizerCommitFault(progress)
    # LambdaLR's native step wrapper marks this only after a real live update.
    optimizer._opt_called=True
    return dict(committed=True,reason=None)
