"""Exercise the new restore wrapper and next Adam update on CPU fixtures."""
import hashlib, importlib.util, json, os, pathlib, random, sys, tempfile, types
import numpy as np
import torch

MODULE='/data/user/jxu957/experiments/topd-20261008/runtime/tools/topd-alpha1-launch-resume5-20261013/runtime/topd_resume_state.py'
sys.path.insert(0,str(pathlib.Path(MODULE).parent))
spec=importlib.util.spec_from_file_location('resume_checked',MODULE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1)
torch.distributed.get_rank=lambda:0
torch.distributed.get_world_size=lambda:4
torch.distributed.barrier=lambda:None
cuda_state={}
torch.cuda.get_rng_state=lambda:cuda_state['value'].clone()
torch.cuda.set_rng_state=lambda x:cuda_state.update(value=x.clone())

def worker():
    w=types.SimpleNamespace()
    w.actor=types.SimpleNamespace(actor_module=torch.nn.Linear(3,2))
    w.actor_optimizer=torch.optim.Adam(w.actor.actor_module.parameters(),lr=1e-6,foreach=False)
    w.actor_lr_scheduler=torch.optim.lr_scheduler.LambdaLR(w.actor_optimizer,lambda s:s/57 if s<57 else 1)
    w._resource_updates=[{'actual_outer_update_ordinal':i} for i in range(1,6)]
    return w

def update(w,x,y):
    w.actor_optimizer.zero_grad();loss=(w.actor.actor_module(x)-y).square().mean();loss.backward();w.actor_optimizer.step()

with tempfile.TemporaryDirectory(prefix='topd-resume-cpu-',dir='/data/user/jxu957/experiments/topd-20261008/environment') as tmp:
    root=pathlib.Path(tmp);audit=root/'audit';audit.mkdir();os.environ['R1_RESOURCE_AUDIT_DIRECTORY']=str(audit);os.environ['TOPD_RUN_DIRECTORY']=str(root)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    old=worker();x=torch.randn(4,3);y=torch.randn(4,2)
    for outer in range(5):
        for _ in range(20):update(old,x,y)
        old.actor_lr_scheduler.step()
    cuda_state['value']=torch.get_rng_state().clone()
    state={'rank':0,'world_size':4,'outer_step':5,'optimizer':m.cpu_tree(old.actor_optimizer.state_dict()),'scheduler':old.actor_lr_scheduler.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':cuda_state['value'].clone(),'model_local':{n:p.detach().clone() for n,p in old.actor.actor_module.named_parameters()},'resource_updates':old._resource_updates}
    proof=m.durable_save(state,root/'resume-rank0.pt');proof['rank']=0
    (root/'resume-complete.json').write_text(json.dumps({'status':'ALL_FOUR_RANKS_AND_DRIVER_RESUME_STATE_VERIFIED','outer_step':5,'ranks':[proof]}))
    order=torch.randperm(169028).tolist()
    driver={'outer_step':5,'next_outer_step':6,'next_batch':5,'data_order':order,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state()}
    m.durable_save(driver,root/'driver-resume.pt')
    class W:
        def init_model(self):
            self.__dict__.update(worker().__dict__);self._is_actor=True;self.config={'topd_resume_checkpoint':str(root)}
        def save_checkpoint(self,*a):pass
    m.install(W);restored=W();restored.init_model()
    assert m.equal_tree(restored.actor_optimizer.state_dict(),state['optimizer'])
    update(old,x,y);update(restored,x,y)
    assert m.equal_tree(old.actor_optimizer.state_dict(),restored.actor_optimizer.state_dict())
    assert m.equal_tree(dict(old.actor.actor_module.named_parameters()),dict(restored.actor.actor_module.named_parameters()))
    assert m.equal_tree(old.actor_lr_scheduler.state_dict(),restored.actor_lr_scheduler.state_dict())
    fake=types.ModuleType('verl.utils.dataset.rl_dataset');fake.collate_fn=lambda x:x;sys.modules['verl.utils.dataset.rl_dataset']=fake
    cfg=types.SimpleNamespace(trainer={'topd_resume_checkpoint':str(root)},data=types.SimpleNamespace(train_batch_size=512))
    trainer=types.SimpleNamespace(config=cfg,train_dataset=list(range(169028)))
    m.prepare_data(trainer)
    assert trainer.global_steps==6 and trainer._topd_resume_batch==5 and trainer._topd_data_order==order
    batch=next(b for i,b in enumerate(trainer.train_dataloader) if i==5)
    assert batch==order[2560:3072]
    print(json.dumps({'status':'CPU_RESTORE_WRAPPER_AND_NEXT_ADAM_UPDATE_EQUALITY_PASS','driver_next_outer':6,'driver_next_batch':5,'restored_optimizer_step_before_next':100,'GPU_restore_tested':False}))
