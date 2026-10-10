"""Durable rank-local optimizer state for the same four-rank FSDP layout."""
import hashlib, json, os, pathlib, random, tempfile
import numpy as np
import torch

def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x, list): return [cpu_tree(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu_tree(v) for v in x)
    return x

def equal_tree(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def durable_save(state,path):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.resume-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: torch.save(state,f);f.flush();os.fsync(f.fileno())
        restored=torch.load(name,map_location='cpu',weights_only=False)
        if not equal_tree(state,restored): raise RuntimeError('Resume serialization roundtrip differs')
        os.replace(name,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if os.path.exists(name):os.unlink(name)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest(),'serialization_roundtrip_pass':True}

def save_worker(worker,path):
    rank=torch.distributed.get_rank();world=torch.distributed.get_world_size()
    state={'version':1,'rank':rank,'world_size':world,
           'outer_step':int(pathlib.Path(path).name.split('_')[-1]),
           'optimizer':cpu_tree(worker.actor_optimizer.state_dict()),
           'scheduler':worker.actor_lr_scheduler.state_dict(),
           'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
           'model_local':{n:p.detach().cpu().clone() for n,p in worker.actor.actor_module.named_parameters()},
           'resource_updates':worker._resource_updates}
    if not state['optimizer']['state']:raise RuntimeError('Adam state absent after actual updates')
    receipt=durable_save(state,pathlib.Path(path)/('resume-rank%d.pt'%rank))
    # Exercise the actual optimizer/scheduler load APIs, preserving exact state.
    worker.actor_optimizer.load_state_dict(state['optimizer'])
    worker.actor_lr_scheduler.load_state_dict(state['scheduler'])
    if not equal_tree(cpu_tree(worker.actor_optimizer.state_dict()),state['optimizer']):raise RuntimeError('Actual optimizer restore differs')
    if not equal_tree(worker.actor_lr_scheduler.state_dict(),state['scheduler']):raise RuntimeError('Actual scheduler restore differs')
    receipt.update(rank=rank,world_size=world,outer_step=state['outer_step'],actual_optimizer_reload_pass=True)
    from pipeline import atomic
    atomic(pathlib.Path(path)/('resume-rank%d-verified.json'%rank),receipt)
    torch.distributed.barrier()

def install(worker_class):
    import functools
    original=worker_class.save_checkpoint
    @functools.wraps(original)
    def save(self,local_path,hdfs_path=None):
        result=original(self,local_path,hdfs_path)
        save_worker(self,local_path)
        return result
    worker_class.save_checkpoint=save
    original_init=worker_class.init_model
    @functools.wraps(original_init)
    def init(self,*args,**kwargs):
        result=original_init(self,*args,**kwargs)
        checkpoint=self.config.get('topd_resume_checkpoint',None)
        if self._is_actor and checkpoint:
            rank=torch.distributed.get_rank()
            state=torch.load(pathlib.Path(checkpoint)/('resume-rank%d.pt'%rank),map_location='cpu',weights_only=False)
            if state['rank']!=rank or state['world_size']!=torch.distributed.get_world_size():raise RuntimeError('Resume requires identical rank layout')
            params=dict(self.actor.actor_module.named_parameters())
            if params.keys()!=state['model_local'].keys():raise RuntimeError('Resume parameter layout differs')
            with torch.no_grad():
                for n,p in params.items():
                    value=state['model_local'][n]
                    if p.shape!=value.shape or p.dtype!=value.dtype:raise RuntimeError('Resume local model geometry differs')
                    p.copy_(value.to(p.device))
            self.actor_optimizer.load_state_dict(state['optimizer']);self.actor_lr_scheduler.load_state_dict(state['scheduler'])
            self._resource_updates=state['resource_updates']
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
            torch.distributed.barrier()
        return result
    worker_class.init_model=init

def save_driver(trainer):
    step=trainer.global_steps
    root=pathlib.Path(trainer.config.trainer.default_local_dir)/'actor'/('global_step_%d'%step)
    ranks=[json.loads((root/('resume-rank%d-verified.json'%r)).read_text()) for r in range(4)]
    if any(x['outer_step']!=step or not x['actual_optimizer_reload_pass'] for x in ranks):raise RuntimeError('Rank restore verification missing')
    state={'outer_step':step,'next_outer_step':step+1,'data_order':trainer._topd_data_order,
           'next_batch':trainer._topd_batch_index+1,'python_rng':random.getstate(),
           'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state()}
    receipt=durable_save(state,root/'driver-resume.pt')
    from pipeline import atomic
    atomic(root/'resume-complete.json',{'status':'ALL_FOUR_RANKS_AND_DRIVER_RESUME_STATE_VERIFIED','outer_step':step,'ranks':ranks,'driver':receipt})

def prepare_data(trainer):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn
    checkpoint=trainer.config.trainer.get('topd_resume_checkpoint',None)
    if checkpoint:
        state=torch.load(pathlib.Path(checkpoint)/'driver-resume.pt',map_location='cpu',weights_only=False)
        order=state['data_order'];trainer.global_steps=state['next_outer_step'];trainer._topd_resume_batch=state['next_batch']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    else:
        generator=torch.Generator().manual_seed(int(trainer.config.data.get('seed',42)))
        order=torch.randperm(len(trainer.train_dataset),generator=generator).tolist() if trainer.config.data.shuffle_train_dataloader else list(range(len(trainer.train_dataset)))
        trainer._topd_resume_batch=0
    trainer._topd_data_order=order
    trainer.train_dataloader=DataLoader(trainer.train_dataset,batch_size=trainer.config.data.train_batch_size,sampler=order,drop_last=True,collate_fn=collate_fn)
