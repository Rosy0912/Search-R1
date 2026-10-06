"""Matched refusal evidence: raw local worker state, never rollback or resume."""
import copy
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .policy_safety_runtime import PolicySafetyRefusal
from .training_state import atomic_save, cpu_copy, digest, finite, rng_state

PROTOCOL = 'tree_partial_state_v1'
IDENTITY_KEYS = {'job_id','run_id','code_commit','config_sha256','data_identity','model_identity',
                 'training_protocol','attempted_outer','completed_outer'}
ENV_KEYS = {'job_id':'SLURM_JOB_ID','run_id':'SEARCH_R1_BASELINE_RUN_ID','code_commit':'SEARCH_R1_BASELINE_COMMIT',
            'config_sha256':'SEARCH_R1_BASELINE_CONFIG_SHA256','data_identity':'SEARCH_R1_BASELINE_DATA_IDENTITY',
            'model_identity':'SEARCH_R1_MODEL_IDENTITY'}


def _gather(value):
    if not dist.is_initialized():return [value]
    values=[None]*dist.get_world_size();dist.all_gather_object(values,value);return values


def _validated_context(worker,data):
    if worker.config.actor.get('fresh200_protocol')!='tree_fresh200_five_leaf_v1':
        raise ValueError('Partial worker persistence requires explicit fresh200 profile')
    context=data.meta_info['tree_partial_context']
    if context['protocol']!=PROTOCOL or set(context['identity'])!=IDENTITY_KEYS:
        raise ValueError('Partial context protocol/complete driver identity differs')
    identity=context['identity']
    if (identity['training_protocol']!='tree_fresh200_five_leaf_v1'
            or type(identity['attempted_outer']) is not int or type(identity['completed_outer']) is not int
            or identity['attempted_outer']!=identity['completed_outer']+1 or not 0<=identity['completed_outer']<200
            or data.meta_info['baseline_outer_step']!=identity['attempted_outer']):
        raise ValueError('Partial attempted/completed outer identity invalid')
    if any(identity[key]!=os.environ[env] for key,env in ENV_KEYS.items()):
        raise ValueError('Partial source/run identity differs from actual worker environment')
    directory=Path(context['directory'])
    own_run=Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']).resolve().parent
    if (not directory.is_absolute() or not directory.resolve().is_relative_to(own_run)
            or directory.name!=f'partial_outer_{identity["attempted_outer"]:06d}' or directory.parent.name!='training_state'):
        raise ValueError('Partial state directory must remain in this own run and match attempted outer')
    # Metadata only, fixed before native update. No model/PT read occurs here.
    json.dumps(context,sort_keys=True,allow_nan=False)
    return copy.deepcopy(context)


def prepare_partial_worker_context(worker,data):
    local_error=None;context=None
    try:context=_validated_context(worker,data)
    except Exception as exc:local_error=repr(exc)
    records=_gather(dict(context=context,error=local_error))
    consistent=all(record['error'] is None and record['context']==records[0]['context'] for record in records)
    worker._partial_prepared_context=context if consistent else None
    if not consistent:
        raise PolicySafetyRefusal('Global partial worker context preparation refusal; local: '+str(local_error),matched_all_ranks=True)
    return context


def raw_partial_worker_snapshot(worker,identity,cuda):
    """Capture actual shards/gradients even if values are nonfinite; no gather."""
    if cuda:torch.cuda.synchronize(torch.cuda.current_device())
    model=worker.actor.actor_module
    parameters=dict(model.named_parameters())
    value=dict(version='tree_partial_worker_v1',identity=copy.deepcopy(identity),
        parameters={name:cpu_copy(parameter) for name,parameter in parameters.items()},
        local_shards={name:cpu_copy(parameter._local_shard) for name,parameter in parameters.items()
                      if hasattr(parameter,'_local_shard') and isinstance(parameter._local_shard,torch.Tensor)},
        gradients={name:None if parameter.grad is None else cpu_copy(parameter.grad) for name,parameter in parameters.items()},
        buffers={name:cpu_copy(value) for name,value in model.named_buffers()},
        optimizer=cpu_copy(worker.actor_optimizer.state_dict()),scheduler=cpu_copy(worker.actor_lr_scheduler.state_dict()),
        rng=rng_state(cuda),module_training_flags=[dict(name=name,training=module.training) for name,module in model.named_modules()],
        parameter_layout={name:dict(shape=list(parameter.shape),dtype=str(parameter.dtype),device=str(parameter.device),
            requires_grad=parameter.requires_grad,local_shard_shape=(list(parameter._local_shard.shape)
                if hasattr(parameter,'_local_shard') and isinstance(parameter._local_shard,torch.Tensor) else None))
            for name,parameter in parameters.items()},
        actual_param_offload_configured=worker._is_offload_param,actual_optimizer_offload_configured=worker._is_offload_optimizer,
        resume_permitted=False,automatic_rollback=False,outer_completed=False)
    state_finite=True;finite_error=None
    try:finite(value)
    except Exception as exc:state_finite=False;finite_error=repr(exc)
    value.update(state_finite=state_finite,state_finite_error=finite_error)
    return value


def persist_matched_partial_worker(worker,data,refusal,scheduler_before):
    """Return the same all-rank envelope after every rank attempts durable save.

    Unmatched errors cannot call any collective here. IO failures are included
    in the common result; no first Ray exception may hide another rank's save.
    """
    if not isinstance(refusal,PolicySafetyRefusal) or refusal.matched_all_ranks is not True:
        raise ValueError('Only matched all-rank PolicySafetyRefusal may save distributed partial state')
    started=time.monotonic();rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    context=None;context_error=None
    try:context=_validated_context(worker,data)
    except Exception as exc:context_error=repr(exc)
    contexts=_gather(dict(context=context,error=context_error))
    consistent=all(entry['error'] is None and entry['context']==contexts[0]['context'] for entry in contexts)
    canonical=next((entry['context'] for entry in contexts if entry['context'] is not None),None)
    driver_identity=canonical['identity'] if canonical else {}
    identity=dict(driver_identity,rank=rank,world_size=world)
    count=getattr(worker.actor,'_actual_committed_optimizer_calls',None)
    actual_after=worker.actor_lr_scheduler.last_epoch
    receipt=dict(rank=rank,world_size=world,identity=identity,
        attempted_outer=driver_identity.get('attempted_outer'),completed_outer=driver_identity.get('completed_outer'),
        committed_optimizer_calls=count if type(count) is int else None,
        scheduler_step_before_actual=scheduler_before,scheduler_step_after_actual=actual_after,
        status='PARTIAL_WORKER_STATE_INCOMPLETE',path=None,sha256=None,bytes=0,logical_state_sha256=None,
        state_finite=None,error=None,refusal=repr(refusal),matched_all_ranks=True,
        optimizer_count_source='optimizer.step successful return before any audit/publish IO',
        resume_permitted=False,exact_same_world_reload_verified=False,automatic_rollback=False,outer_completed=False)
    try:
        if not consistent:raise ValueError('All-rank partial context invalid/mismatched: '+str(context_error))
        if type(count) is not int or not 0<=count<=20:raise ValueError('Actual partial optimizer return counter missing or invalid')
        directory=Path(canonical['directory'])
        state_path=directory/f'worker-{rank:02d}.pt';receipt_path=directory/f'worker-{rank:02d}.json'
        cuda=dist.is_initialized() and dist.get_backend()=='nccl'
        saved=raw_partial_worker_snapshot(worker,identity,cuda)
        saved.update(committed_optimizer_calls=count,scheduler_step_before_actual=scheduler_before,
                     scheduler_step_after_actual=actual_after,refusal=repr(refusal))
        receipt['logical_state_sha256']=digest(saved)
        receipt['state_finite']=saved['state_finite'];receipt['state_finite_error']=saved['state_finite_error']
        receipt['path']=str(state_path);receipt['sha256']=atomic_save(saved,state_path)
        receipt['bytes']=state_path.stat().st_size
        del saved
        receipt.update(status='PARTIAL_WORKER_STATE_SAVED',optimizer_state_entries=len(worker.actor_optimizer.state),
            scheduler_advanced_actual=(actual_after!=scheduler_before),snapshot_save_seconds=time.monotonic()-started,
            receipt_path=str(receipt_path))
        receipt['receipt_sha256']=atomic_save(receipt,receipt_path,json_format=True)
    except Exception as exc:
        receipt.update(status='PARTIAL_WORKER_STATE_INCOMPLETE',error=repr(exc),snapshot_save_seconds=time.monotonic()-started)
    workers=sorted(_gather(receipt),key=lambda record:record['rank'])
    all_saved=(consistent and len(workers)==world and [entry['rank'] for entry in workers]==list(range(world))
               and all(entry['status']=='PARTIAL_WORKER_STATE_SAVED' for entry in workers))
    return dict(protocol=PROTOCOL,identity=driver_identity,identity_consistent=consistent,
        status='PARTIAL_WORKER_STATE_SAVED' if all_saved else 'PARTIAL_WORKER_STATE_INCOMPLETE',
        worker_states=workers,scheduler_advanced_actual=[
            entry['scheduler_step_after_actual']!=entry['scheduler_step_before_actual'] for entry in workers],
        resume_permitted=False,automatic_rollback=False,outer_completed=False,
        scope='Raw actual local worker state on matched refusal; no rollback, scheduler/outer credit, HF export or reload acceptance')
