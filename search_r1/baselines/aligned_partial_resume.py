"""Exact version3 outer1 PARTIAL restore; never an Adam/stage/scheduler update.

This narrow loader preserves an interrupted transaction's twenty existing writes.
The driver must separately restore its cursor/RNG/BAPO and finish its pending commit.
"""
from pathlib import Path
import copy,hashlib,json,math,os,re

OLD_SOURCE='7a501b38b7ed6b06bf39935b97fcb7bbd56ecc47'
OLD_RUN=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009/runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356')

def require(value,message):
    if not value:raise ValueError(message)

def exact_file(entry,root):
    p=Path(entry['path'])
    require(root in p.parents and p.is_file() and not p.is_symlink() and p.resolve()==p,'ordinary original native file required')
    require(type(entry.get('bytes'))is int and entry['bytes']==p.stat().st_size,'original native file size differs')
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    require(h.hexdigest()==entry['sha256'],'original native file SHA differs')
    return p

def validate_manifest(manifest):
    require(manifest.get('old_run_root')==str(OLD_RUN) and manifest.get('old_source_commit')==OLD_SOURCE,'only actual3356 outer1 partial is supported')
    target=Path(manifest['target_run_root'])
    require(target.is_absolute() and target.resolve()==target and target.parent.name=='runs' and target!=OLD_RUN,'distinct canonical continuation RUN required')
    job=manifest.get('target_job_id');require(type(job)is str and job.isdigit() and job!='3356','distinct actual continuation job required')
    require(target.name.endswith('-'+job),'continuation RUN/job mismatch')
    require(re.fullmatch(r'[0-9a-f]{40}',manifest.get('target_source_commit',''))is not None,'bound continuation source required')
    ranks=manifest.get('ranks');require(isinstance(ranks,list) and len(ranks)==8 and {r.get('rank')for r in ranks}==set(range(8)) and all(type(r['rank'])is int for r in ranks),'exact eight original native ranks required')
    events=manifest.get('rank_write_ledger');require(isinstance(events,list) and len(events)==8 and {e.get('rank')for e in events}==set(range(8)),'all original write events required')
    from .fresh200_contract import validate_rank_updates
    validate_rank_updates(events,1)
    require(all(e.get('attempt_id')==1 and e.get('baseline_aligned')is True and e.get('optimizer_steps')==20 and e.get('outer_commit_phase')=='WRITES_PREPARED' and e.get('accepted_complete_outer')is False for e in events),'original pending twenty-write transaction required')
    return target

def _finite(value):
    import torch
    if hasattr(value,'local_shards'):
        return all(_finite(s.tensor)for s in value.local_shards())
    if hasattr(value,'to_local')and hasattr(value,'placements'):return _finite(value.to_local())
    if torch.is_tensor(value):return bool(torch.isfinite(value).all().item())
    if isinstance(value,dict):return all(_finite(x)for x in value.values())
    if isinstance(value,(tuple,list)):return all(_finite(x)for x in value)
    if isinstance(value,float):return math.isfinite(value)
    return True

def validate_native(payload,report,rank,manifest):
    import torch
    from .full_state import state_digest
    require(all(type(payload.get(k))is int for k in ('version','rank','world_size','step','attempted_outer','previous_complete_outer','attempt_id')),'native counters require exact integer types')
    require(all(type(report.get(k))is int for k in ('rank','world_size','step','native_state_version','scheduler_last_epoch','optimizer_state_entries')),'native report counters require exact integer types')
    require(payload.get('version')==3 and payload.get('rank')==rank and type(payload.get('rank'))is int and payload.get('world_size')==8 and payload.get('step')==1 and payload.get('model_state_dict_type')=='SHARDED_STATE_DICT','exact v3 rank/world/outer1 required')
    require(payload.get('update_status')=='PARTIAL' and payload.get('attempted_outer')==1 and payload.get('previous_complete_outer')==0 and payload.get('attempt_id')==1,'actual original pending PARTIAL required')
    require(report.get('status')=='PARTIAL_CURRENT_STATE_SAVED' and report.get('native_state_version')==3 and report.get('rank')==rank and report.get('world_size')==8 and report.get('step')==1 and report.get('roundtrip')is False and type(report.get('pid'))is int and report['pid']>0,'original serialized partial sidecar required')
    require(state_digest(payload)==report.get('state_digest') and report.get('reloaded_state_digest')==report['state_digest'],'original serialized payload digest differs')
    event=next(e for e in manifest['rank_write_ledger']if e['rank']==rank)
    require(payload.get('latest_rank_event')==event and report.get('latest_rank_event')==event,'original rank write event differs')
    pending=payload.get('aligned_pending_outer')
    ledger_hash=hashlib.sha256(json.dumps(manifest['rank_write_ledger'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(isinstance(pending,dict)and all(type(pending.get(k))is int for k in ('actual_outer','attempt_id','scheduler_before')),'pending counters require exact integer types')
    require(pending==dict(actual_outer=1,attempt_id=1,scheduler_before=0,write_ledger_sha256=ledger_hash) and report.get('aligned_pending_outer')==pending,'original pending transaction identity differs')
    require(payload['scheduler'].get('last_epoch')==0 and report.get('scheduler_last_epoch')==0,'pending outer scheduler already advanced')
    require(isinstance(payload['model'],dict)and bool(payload['model'])and _finite(payload['model']),'empty or nonfinite model')
    require(type(report.get('model_dtensor_entries'))is int and report['model_dtensor_entries']>0 and sum(hasattr(v,'device_mesh')and hasattr(v,'to_local')for v in payload['model'].values())==report['model_dtensor_entries'],'actual FSDP DTensor topology required')
    state=payload['optimizer']['state'];groups=payload['optimizer']['param_groups']
    require(bool(state) and len(state)==report.get('optimizer_state_entries') and groups,'actual Adam state required')
    for item in state.values():
        step=item.get('step');require(torch.is_tensor(step) and step.ndim==0 and torch.isfinite(step).item() and step.item()==20,'Adam must preserve exactly twenty writes')
        require('exp_avg'in item and 'exp_avg_sq'in item and _finite(item),'nonfinite/incomplete Adam moments')
    expected_lr=1e-6/57
    require(all(type(g.get('lr'))in(int,float)and math.isfinite(g['lr'])and g['lr']==expected_lr for g in groups),'initial scheduler0 LR must preserve prescribed warmup1/57')
    require(payload['scheduler'].get('_last_lr')==[expected_lr]*len(groups),'saved scheduler LR differs from actual Adam LR')
    require('rng'in payload and 'generation_rng'in payload and state_digest(payload['generation_rng'])==report.get('generation_rng_before_digest')==report.get('generation_rng_after_digest'),'saved exact generation RNG differs')
    return payload

def _collective(worker,error,phase,manifest):
    import torch.distributed as dist
    rows=[None]*worker.world_size
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,allow_nan=False).encode()).hexdigest()
    dist.all_gather_object(rows,dict(rank=worker.rank,error=error,phase=phase,manifest_sha256=digest),group=worker.actor_module_fsdp.process_group)
    require(len(rows)==8 and {r['rank']for r in rows}==set(range(8)) and len({(r['phase'],r['manifest_sha256'])for r in rows})==1 and not any(r['error']for r in rows),'collective partial restore failed: '+repr(rows))

def load_rank(worker,manifest,witness_directory):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,ShardedStateDictConfig
    from verl.utils.fsdp_utils import load_fsdp_param_and_grad,offload_fsdp_param_and_grad,load_fsdp_optimizer,offload_fsdp_optimizer
    from .full_state import rng_state,restore_rng,state_digest
    loaded=report=None;error=None
    try:
        target=validate_manifest(manifest)
        require(worker.world_size==8 and worker.config.actor.get('baseline_aligned',False) and worker.config.actor.get('aligned_stage_commit_enabled',False),'actual aligned eight-rank transaction required')
        require(os.environ.get('BAPO_RUN_ID')==str(target) and os.environ.get('SLURM_JOB_ID')==manifest['target_job_id'],'actual continuation process identity differs')
        entry=next(e for e in manifest['ranks']if e['rank']==worker.rank)
        expected=OLD_RUN/'checkpoints/fresh200_state/attempt_1_candidate_outer_1_PARTIAL/ranks'
        path=exact_file(entry['state'],OLD_RUN);side=exact_file(entry['report'],OLD_RUN)
        require(path==expected/f'rank_{worker.rank}.pt'and side==path.with_suffix('.json'),'original partial native path differs')
        report=json.loads(side.read_bytes());require(report['path']==str(path)and report['sha256']==entry['state']['sha256'],'sidecar file binding differs')
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        validate_native(loaded,report,worker.rank,manifest)
        current_generation=worker.rollout_sharding_manager.generation_rng_state()
        require(loaded['generation_rng'].get('version')==current_generation.get('version')==1,'generation RNG version differs')
        for name in ('torch_random_states','gen_random_states'):
            prior,now=loaded['generation_rng'][name],current_generation[name]
            require((prior is None)==(now is None) and (prior is None or (torch.is_tensor(prior)and prior.dtype==now.dtype==torch.uint8 and prior.shape==now.shape and prior.ndim==1)),'generation RNG topology differs before restore')
        require(len(loaded['rng']['cuda'])==1 and loaded['rng']['cuda'][0]['device']==torch.cuda.current_device(),'rank-local CUDA RNG device differs')
        require(os.getpid()!=report['pid'],'cross-job loader unexpectedly has original PID')
        out=Path(witness_directory);require(target in out.parents and out.is_dir()and out.resolve()==out and not(out/f'rank_{worker.rank}.json').exists(),'fresh canonical witness directory required')
    except Exception as exc:error=repr(exc)
    _collective(worker,error,'VALIDATE_ORIGINAL_BEFORE_ANY_MODEL_LOAD',manifest)
    try:
        error=None
        try:
            if worker._is_offload_param:load_fsdp_param_and_grad(worker.actor_module_fsdp,torch.cuda.current_device(),load_grad=worker._is_offload_grad)
            if worker._is_offload_optimizer:load_fsdp_optimizer(worker.actor_optimizer,torch.cuda.current_device())
        except Exception as exc:error=repr(exc)
        _collective(worker,error,'READY_FOR_COLLECTIVE_MODEL_LOAD',manifest)
        error=None;actual=None
        try:
            with FSDP.state_dict_type(worker.actor_module_fsdp,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True)):
                worker.actor_module_fsdp.load_state_dict(loaded['model'],strict=True)
                worker.actor_optimizer.load_state_dict(loaded['optimizer'])
                worker.actor_lr_scheduler.load_state_dict(loaded['scheduler'])
                restore_rng(loaded['rng'])
                worker.rollout_sharding_manager.load_generation_rng_state(loaded['generation_rng'])
                worker._fresh200_last_update_event=copy.deepcopy(loaded['latest_rank_event'])
                worker._aligned_pending_outer=copy.deepcopy(loaded['aligned_pending_outer'])
                actual=dict(loaded)
                actual.update(model=worker.actor_module_fsdp.state_dict(),optimizer=worker.actor_optimizer.state_dict(),scheduler=worker.actor_lr_scheduler.state_dict(),rng=rng_state(torch.cuda.current_device()),generation_rng=worker.rollout_sharding_manager.generation_rng_state(),latest_rank_event=worker._fresh200_last_update_event,aligned_pending_outer=worker._aligned_pending_outer)
                require(state_digest(actual)==report['state_digest'],'restored v3 model/Adam/scheduler/RNG/pending state differs')
        except Exception as exc:error=repr(exc)
        _collective(worker,error,'EXACT_FULL_V3_STATE_RESTORED_NO_UPDATE',manifest)
        witness=dict(status='EXACT_CROSS_JOB_ALIGNED_PARTIAL_V3_RELOAD_OK',rank=worker.rank,world_size=8,step=1,attempt_id=1,old_pid=report['pid'],new_pid=os.getpid(),old_run_root=str(OLD_RUN),target_run_root=manifest['target_run_root'],target_job_id=manifest['target_job_id'],old_source_commit=OLD_SOURCE,target_source_commit=manifest['target_source_commit'],original_state_sha256=entry['state']['sha256'],state_digest=report['state_digest'],reloaded_state_digest=state_digest(actual),scheduler_last_epoch=0,adam_steps=20,aligned_pending_outer=copy.deepcopy(worker._aligned_pending_outer),generation_rng_digest=state_digest(actual['generation_rng']),no_adam_or_stage_observe_or_scheduler_step=True)
        error=None
        try:
            with(out/f'rank_{worker.rank}.json').open('x')as f:json.dump(witness,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
            descriptor=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        except Exception as exc:error=repr(exc)
        _collective(worker,error,'WITNESSES_PERSISTED',manifest)
        return witness
    finally:
        if worker._is_offload_param:offload_fsdp_param_and_grad(worker.actor_module_fsdp,offload_grad=worker._is_offload_grad)
        if worker._is_offload_optimizer:offload_fsdp_optimizer(worker.actor_optimizer)
