import json,math
from types import MethodType
from pathlib import Path
import torch,pytest
from tensordict import TensorDict
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.policy_safety_runtime import PolicySafetyRuntime,PolicySafetyRefusal
from search_r1.baselines.policy_safety import update_statistics,admission
from verl.trainer.ppo.core_algos import kl_penalty

def setup(monkeypatch,tmp_path,enabled=True,**thresholds):
 monkeypatch.setenv('SEARCH_R1_TREE_RANK_AUDIT_DIR',str(tmp_path));monkeypatch.setattr(TensorDict,'cuda',lambda self,*a,**k:self)
 import search_r1.baselines.tree_rank_audit as ta
 monkeypatch.setattr(ta,'enabled',lambda:False)
 model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(.01)
 a=DataParallelPPOActor.__new__(DataParallelPPOActor);a.actor_module=model;a.actor_optimizer=torch.optim.Adam(model.parameters(),lr=.01);a.use_ulysses_sp=False
 safety=dict(enabled=enabled,dual_clip=3.,kl_log_bound=30.,monitor_rows_per_rank=4,target_old_policy_kl=10.,negative_tail_limit=1.,initial_old_logprob_max_abs_delta=1e-7);safety.update(thresholds)
 a.config=OmegaConf.create(dict(ppo_mini_batch_size=4,ppo_micro_batch_size=2,ppo_epochs=1,state_masking=True,use_kl_loss=False,use_dynamic_bsz=False,baseline_method='tree_grpo',entropy_coeff=0.,clip_ratio=.2,grad_clip=1.,policy_safety=safety))
 def forward(self,micro_batch,temperature):
  x=self.actor_module(torch.ones(len(micro_batch),1)).expand(-1,3)
  x=x+torch.rand(())*0
  return x*0,x
 a._forward_micro_batch=MethodType(forward,a)
 batch=TensorDict(dict(responses=torch.ones(8,3,dtype=torch.long),input_ids=torch.ones(8,5,dtype=torch.long),attention_mask=torch.ones(8,5),position_ids=torch.arange(5).expand(8,-1),loss_mask=torch.ones(8,3),old_log_probs=torch.full((8,3),.01),advantages=torch.ones(8,3),policy_weights=torch.full((8,3),1/3)),batch_size=[8])
 return a,DataProto(batch=batch,meta_info={'temperature':1.,'baseline_outer_step':111})

def test_real_actor_monitor_onoff_rng_and_parameters(monkeypatch,tmp_path):
 torch.manual_seed(93);a,d=setup(monkeypatch,tmp_path/'off',False);m=a.update_policy(d);rng=torch.get_rng_state().clone();params=a.actor_module.weight.detach().clone()
 torch.manual_seed(93);b,e=setup(monkeypatch,tmp_path/'on');n=b.update_policy(e)
 assert torch.equal(rng,torch.get_rng_state());assert torch.equal(params,b.actor_module.weight.detach())
 assert m['actor/optimizer_steps']==n['actor/optimizer_steps']==2
 assert len(list((tmp_path/'on'/'policy-safety').glob('*.json')))==6

def test_actual_postwrite_refusal_count(monkeypatch,tmp_path):
 a,d=setup(monkeypatch,tmp_path,target_old_policy_kl=1e-12)
 before=a.actor_module.weight.detach().clone()
 with pytest.raises(PolicySafetyRefusal):a.update_policy(d)
 assert not torch.equal(before,a.actor_module.weight)
 events=[json.loads(p.read_text()) for p in (tmp_path/'policy-safety').glob('*.json')]
 assert any(e['phase']=='after_optimizer' and e['committed_optimizer_calls']==1 and not e['decision']['allowed'] for e in events)
 assert all('postclip_grad_norm' not in e for e in events) # actual measurement is not persisted

def test_confirm_audit_io_hides_committed_write(monkeypatch,tmp_path):
 a,d=setup(monkeypatch,tmp_path)
 import search_r1.baselines.tree_rank_audit as ta
 class FailAudit:
  def __init__(self,*args):pass
  def epoch(self):pass
  def micro(self,*args,**kwargs):pass
  def failure(self,error):pass
  def gradient(self,norm):raise OSError('simulated ratio publication failure')
 monkeypatch.setattr(ta,'enabled',lambda:True);monkeypatch.setattr(ta,'TreeRankAudit',FailAudit)
 before=a.actor_module.weight.detach().clone()
 with pytest.raises(PolicySafetyRefusal):a.update_policy(d)
 assert not torch.equal(before,a.actor_module.weight)
 assert a._policy_safety.committed==1 # repaired: actual write recorded before audit I/O
 assert len(list((tmp_path/'policy-safety').glob('*.json')))==3

def test_confirm_reference_subtract_overflow_bypasses_sync(monkeypatch,tmp_path):
 a,d=setup(monkeypatch,tmp_path);r=PolicySafetyRuntime(a,d.batch,d.meta_info,a.config.policy_safety)
 old=current=torch.tensor([[-60000.]],dtype=torch.float16);ref=torch.tensor([[60000.]],dtype=torch.float16);one=torch.ones_like(old)
 with pytest.raises(PolicySafetyRefusal):r.inputs(old,current,one,one,one,one,ref) # updated bytes synchronously reject
 with pytest.raises(ValueError,match='Nonfinite'):kl_penalty(current,ref,'low_var_kl')

def test_confirm_nonfinite_initial_threshold_accepted(monkeypatch,tmp_path):
 a,d=setup(monkeypatch,tmp_path,initial_old_logprob_max_abs_delta=float('nan'),target_old_policy_kl=float('inf'))
 with pytest.raises(PolicySafetyRefusal,match="startup"):PolicySafetyRuntime(a,d.batch,d.meta_info,a.config.policy_safety)

def test_confirm_admission_nonfinite_reduction_unchecked():
 x=torch.zeros(1,dtype=torch.float64);stats,_=update_statistics(x,x,-torch.ones_like(x),torch.ones_like(x),torch.ones_like(x))
 stats['mass']=torch.tensor(float('inf'));stats['kl_sum']=torch.tensor(float('inf'))
 out=admission(stats,.01,1.)
 assert not out['allowed'] and out['sampled_old_policy_kl'] is None # updated bytes reject global overflow

def test_monitor_validates_zero_weight_padding(monkeypatch,tmp_path):
 a,d=setup(monkeypatch,tmp_path)
 d.batch['attention_mask'][:,-1]=0;d.batch['policy_weights'][:,-1]=0;d.batch['policy_weights'][:,:2]=.5;d.batch['old_log_probs'][:,-1]=float('nan')
 r=PolicySafetyRuntime(a,d.batch,d.meta_info,a.config.policy_safety)
 rawmask=d.batch['loss_mask'].clone()
 assert r.measure('before_optimizer')['allowed']
 assert torch.equal(rawmask,d.batch['loss_mask'])


def distributed_io_worker(rank,rendezvous,output):
 import torch.distributed as dist
 dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
 monkeypatch=pytest.MonkeyPatch()
 try:
  root=Path(output)/str(rank);a,d=setup(monkeypatch,root)
  original=Path.open
  def failing(self,*args,**kwargs):
   if rank==1 and self.name.endswith('event-0002.json'):raise OSError('one-rank post-write I/O rejection')
   return original(self,*args,**kwargs)
  monkeypatch.setattr(Path,'open',failing)
  before=a.actor_module.weight.detach().clone()
  try:a.update_policy(d);raise AssertionError('must refuse')
  except PolicySafetyRefusal as error:
   assert a._policy_safety.committed==1 and not torch.equal(before,a.actor_module.weight)
   (Path(output)/f'result-{rank}.json').write_text(json.dumps(dict(rank=rank,committed=1,error=str(error),changed=True)))
 finally:
  monkeypatch.undo();dist.destroy_process_group()

def test_real_two_rank_actor_postwrite_io_refusal(tmp_path):
 import torch.multiprocessing as mp
 mp.spawn(distributed_io_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
 out=[json.loads((tmp_path/f'result-{r}.json').read_text()) for r in range(2)]
 assert all(x['committed']==1 and 'Global safety evidence-write failure' in x['error'] for x in out)


def startup_worker(rank,rendezvous,output,mode):
 import torch.distributed as dist
 dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
 monkeypatch=pytest.MonkeyPatch()
 try:
  root=Path(output)/str(rank);a,d=setup(monkeypatch,root)
  if rank==1:
   if mode=='threshold':a.config.policy_safety.target_old_policy_kl=float('nan')
   else:root.write_text('file instead of directory')
  try:PolicySafetyRuntime(a,d.batch,d.meta_info,a.config.policy_safety);raise AssertionError('must refuse')
  except PolicySafetyRefusal as error:
   assert 'startup' in str(error)
   (Path(output)/f'result-{rank}.json').write_text(json.dumps(dict(rank=rank,error=str(error))))
 finally:monkeypatch.undo();dist.destroy_process_group()

@pytest.mark.parametrize('mode',['threshold','mkdir'])
def test_two_rank_one_rank_startup_refusal(tmp_path,mode):
 import torch.multiprocessing as mp
 mp.spawn(startup_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path),mode),nprocs=2,join=True)
 assert all('Global safety startup refusal' in json.loads((tmp_path/f'result-{r}.json').read_text())['error'] for r in range(2))

def test_confirm_full_actor_padding_still_raises_before_loss_sync(monkeypatch,tmp_path):
 a,d=setup(monkeypatch,tmp_path)
 d.batch['attention_mask'][:,-1]=0;d.batch['policy_weights'][:,-1]=0;d.batch['policy_weights'][:,:2]=.5;d.batch['old_log_probs'][:,-1]=float('nan')
 before=a.actor_module.weight.detach().clone()
 with pytest.raises(ValueError,match='Nonfinite policy old'):a.update_policy(d)
 assert a._policy_safety.committed==0 and torch.equal(before,a.actor_module.weight)
 # monitor and safety.inputs ignore pad correctly, but weighted_policy_loss uses rawmask locally
 assert len(list((tmp_path/'policy-safety').glob('*.json')))==1
