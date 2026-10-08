import importlib.util,json,pathlib,sys,hashlib,types,io,time
import torch
from omegaconf import OmegaConf,ListConfig,DictConfig
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
source=root/'runs/author-native-phase-aware-warm-five-m'
stage=root/'checkpoint-digest-v2'
sys.path.insert(0,str(root/'repo/verl'))
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=module(source/'tools/checkpoint_observer.py','old_digest')
new=module(stage/'checkpoint_observer.py','new_digest')
rows=[]
for p in (source/'evidence').glob('controller-*.jsonl'):
 for line in p.read_text().splitlines():
  x=json.loads(line)
  if x.get('kind')=='launch':rows.append(x)
assert len(rows)==1
cfg=OmegaConf.create(rows[0]['config'])
from verl.workers.config.optimizer import build_optimizer
checks=[]
for role in ['actor_rollout_ref','reranker_actor_rollout_ref']:
 model=torch.nn.Linear(3,2,dtype=torch.float32)
 optcfg=cfg[role].actor.optim
 optimizer=build_optimizer(model.parameters(),optcfg)
 betas=optimizer.param_groups[0]['betas'];assert isinstance(betas,ListConfig)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:1.0)
 manager=types.SimpleNamespace(model=model,optimizer=optimizer,lr_scheduler=scheduler)
 for _ in range(4):
  optimizer.zero_grad();model(torch.ones(2,3)).sum().backward();optimizer.step();scheduler.step()
 assert all(int(state['step'])==4 for state in optimizer.state.values())
 try:old.full_state_digest(manager)
 except TypeError as e:assert 'ListConfig' in str(e)
 else:raise AssertionError('Old actual-container failure was not reproduced')
 rng=torch.get_rng_state().clone();param_before=[x.detach().clone() for x in model.parameters()];moments_before=[{k:(v.clone() if torch.is_tensor(v) else v) for k,v in s.items()} for s in optimizer.state.values()]
 before=new.full_state_digest(manager)
 assert optimizer.param_groups[0]['betas'] is betas and torch.equal(torch.get_rng_state(),rng)
 assert all(torch.equal(a,b) for a,b in zip(param_before,model.parameters()))
 assert all(torch.equal(before_state[k],actual[k]) for before_state,actual in zip(moments_before,optimizer.state.values()) for k in actual)
 packed=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),packed);packed.seek(0)
 saved=torch.load(packed,weights_only=False,map_location='cpu')
 loaded_model=torch.nn.Linear(3,2,dtype=torch.float32);loaded_optimizer=build_optimizer(loaded_model.parameters(),optcfg);loaded_scheduler=torch.optim.lr_scheduler.LambdaLR(loaded_optimizer,lambda n:1.0)
 loaded_model.load_state_dict(saved['model']);loaded_optimizer.load_state_dict(saved['optimizer']);loaded_scheduler.load_state_dict(saved['scheduler'])
 after=new.full_state_digest(types.SimpleNamespace(model=loaded_model,optimizer=loaded_optimizer,lr_scheduler=loaded_scheduler))
 assert before==after
 checks.append(dict(role=role,betas_type=type(betas).__name__,actual_optimizer_type=type(optimizer).__name__,full_cpu_state_roundtrip_equal=True,digest=before))
plain={'a':[torch.arange(7,dtype=torch.float32),{'v':2.0}],'b':(1,None,True)}
assert old.tree_hash(plain)==new.tree_hash(plain)
container=OmegaConf.create({'a':[0.9,0.999],'b':{'x':2}})
assert new.tree_hash(container)==new.tree_hash({'a':[0.9,0.999],'b':{'x':2}})
assert new.tree_hash(OmegaConf.create([0.9,0.999]))!=new.tree_hash(OmegaConf.create([0.8,0.999]))
try:new.tree_hash(object())
except TypeError:pass
else:raise AssertionError('Unknown type silently accepted')
# Exercise the real installed observer wrappers around CPU-only checkpoint callbacks.
# This is synthetic checkpoint I/O, explicitly not FSDP distributed save/load proof.
import os,tempfile
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
records=[];fake_observe=types.ModuleType('observe_worker');fake_observe.write_record=lambda kind,value:records.append((kind,value));sys.modules['observe_worker']=fake_observe
calls={'save':0,'load':0}
def cpu_save(m,path,*args,**kwargs):
 calls['save']+=1;p=pathlib.Path(path);p.mkdir(exist_ok=True)
 torch.save(m.model.state_dict(),p/'model_world_size_1_rank_0.pt')
 torch.save(m.optimizer.state_dict(),p/'optim_world_size_1_rank_0.pt')
 torch.save(m.lr_scheduler.state_dict(),p/'extra_state_world_size_1_rank_0.pt')
 return 'save-sentinel'
def cpu_load(m,path,*args,**kwargs):
 calls['load']+=1;p=pathlib.Path(path)
 m.model.load_state_dict(torch.load(p/'model_world_size_1_rank_0.pt',weights_only=False))
 m.optimizer.load_state_dict(torch.load(p/'optim_world_size_1_rank_0.pt',weights_only=False))
 m.lr_scheduler.load_state_dict(torch.load(p/'extra_state_world_size_1_rank_0.pt',weights_only=False))
 return 'load-sentinel'
FSDPCheckpointManager.save_checkpoint=cpu_save;FSDPCheckpointManager.load_checkpoint=cpu_load
os.environ['SLURM_JOB_ID']='CPU_DIAGNOSTIC_NOT_SLURM_ALLOCATION'
manager.world_size=1;manager.rank=0;manager._native_observed_role='reranker'
for flag in ['should_save_model','should_save_optimizer','should_save_extra','should_load_model','should_load_optimizer','should_load_extra']:setattr(manager,flag,True)
new.install()
with tempfile.TemporaryDirectory(prefix='digest-wrapper-',dir=stage) as tmp:
 assert FSDPCheckpointManager.save_checkpoint(manager,tmp)=='save-sentinel'
 restored=types.SimpleNamespace(**manager.__dict__)
 assert FSDPCheckpointManager.load_checkpoint(restored,tmp)=='load-sentinel'
 assert calls=={'save':1,'load':1}
 assert [k for k,v in records]==['checkpoint_saved_full_digest','checkpoint_loaded_full_digest']
 assert records[0][1]['state']==records[1][1]['state']
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,checked_at=time.time(),scope='native CPU real optimizer container/digest serialization only; no FSDP GPU checkpoint or RL proof',checks=checks,old_sha256=hashlib.sha256((source/'tools/checkpoint_observer.py').read_bytes()).hexdigest(),new_sha256=hashlib.sha256((stage/'checkpoint_observer.py').read_bytes()).hexdigest(),source_job=3106,CUDA_initialized=False,RL_updates=0,synthetic_CPU_checkpoint_wrapper_roundtrip=True,FSDP_GPU_save_load_proven=False)))
