"""Observe native optimizer calls and compare complete local recovery state."""
import functools,hashlib,json,os,time
from pathlib import Path


def record(worker,kind,values):
    import torch
    root=Path(os.environ['COSEARCH_AUDIT_ROOT']);root.mkdir(parents=True,exist_ok=True)
    rank=torch.distributed.get_rank()
    role='reranker' if str(worker.role).startswith('reranker') else 'main'
    value=dict(values,kind=kind,role=role,rank=rank,world_size=torch.distributed.get_world_size(),pid=os.getpid(),job=os.environ['SLURM_JOB_ID'],epoch=time.time())
    with (root/(role+'-'+str(rank)+'.jsonl')).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def digest(value):
    import torch
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(json.dumps([str(v.dtype),list(v.shape)]).encode())
            raw=v.detach().contiguous().reshape(-1).view(torch.uint8)
            for start in range(0,raw.numel(),1024*1024):
                h.update(raw[start:start+1024*1024].cpu().numpy().tobytes())
        elif type(v).__module__.startswith('numpy'):
            if hasattr(v,'shape') and v.shape:
                h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
            else:add(v.item())
        elif isinstance(v,dict):
            for key in sorted(v,key=str):h.update(str(key).encode());add(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for item in v:add(item)
        else:h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
    add(value)
    return h.hexdigest()


def state(worker):
    optimizer=worker.actor_optimizer
    params=[p for group in optimizer.param_groups for p in group['params']]
    return {'local_model_sha256':digest([p.detach() for p in params]),
            'optimizer_sha256':digest(optimizer.state_dict()),
            'scheduler_sha256':digest(worker.actor_lr_scheduler.state_dict()),
            'rng_sha256':digest(worker.checkpoint_manager.get_rng_state())}


def install():
    from verl.workers.fsdp_workers import ActorRolloutRefWorker as Worker
    if getattr(Worker,'_cosearch_audit_installed',False):return
    if os.environ.get('COSEARCH_LEASED_UUIDS'):
        import gpu_identity
        gpu_identity.install()
    original_init=Worker.init_model
    @functools.wraps(original_init)
    def init(worker,*args,**kwargs):
        from worker_device import bind
        record(worker,'device_bound_before_init',bind())
        result=original_init(worker,*args,**kwargs)
        if worker._is_actor:
            import torch
            if torch.distributed.get_world_size()!=4:raise RuntimeError('Each role must use four ranks')
            props=torch.cuda.get_device_properties(torch.cuda.current_device())
            from probe_faiss_gpu import visible_cuda_uuids
            uuid=visible_cuda_uuids(torch.cuda.device_count())[torch.cuda.current_device()]
            if uuid not in os.environ['COSEARCH_LEASED_UUIDS'].split(','):raise RuntimeError('Worker uses an unleased GPU')
            record(worker,'initialized',{'uuid':uuid,'model':props.name,'total_memory_bytes':props.total_memory,'cgroup':Path('/proc/self/cgroup').read_text(),'mask':os.environ.get('CUDA_VISIBLE_DEVICES')})
        return result
    Worker.init_model=init
    original_update=Worker.update_actor
    @functools.wraps(original_update)
    def update(worker,data):
        step=int(data.meta_info['cosearch_outer_step'])
        count=[0]
        def completed(optimizer,args,kwargs):count[0]+=1
        hook=worker.actor_optimizer.register_step_post_hook(completed)
        before=int(worker.actor_lr_scheduler.last_epoch)
        try:result=original_update(worker,data)
        finally:hook.remove()
        after=int(worker.actor_lr_scheduler.last_epoch)
        if str(worker.role)=='actor_rollout' and count[0]!=20:
            raise RuntimeError('Main actual optimizer steps differ from20: '+str(count[0]))
        if after-before!=1:raise RuntimeError('Scheduler must advance once per update outer')
        record(worker,'update',{'outer':step,'optimizer_calls':count[0],'scheduler_before':before,'scheduler_after':after,'local_rows':len(data),'metrics':result.meta_info['metrics']})
        return result
    Worker.update_actor=update
    original_save=Worker.save_checkpoint
    @functools.wraps(original_save)
    def save(worker,local_path,*args,**kwargs):
        before=state(worker)
        result=original_save(worker,local_path,*args,**kwargs)
        import torch
        path=Path(local_path)/('recovery-state-rank-'+str(torch.distributed.get_rank())+'.json')
        path.write_text(json.dumps(before,sort_keys=True)+'\n')
        record(worker,'saved_state',dict(before,path=str(path)))
        return result
    Worker.save_checkpoint=save
    original_load=Worker.load_checkpoint
    @functools.wraps(original_load)
    def load(worker,local_path,*args,**kwargs):
        result=original_load(worker,local_path,*args,**kwargs)
        if local_path is not None:
            import torch
            path=Path(local_path)/('recovery-state-rank-'+str(torch.distributed.get_rank())+'.json')
            expected=json.loads(path.read_text());actual=state(worker)
            if actual!=expected:raise RuntimeError('Full local model/optimizer/scheduler recovery differs')
            record(worker,'recovery_state_equal',dict(actual,path=str(path)))
        return result
    Worker.load_checkpoint=load
    Worker._cosearch_audit_installed=True
    from runtime_fixes import install_worker
    install_worker()
