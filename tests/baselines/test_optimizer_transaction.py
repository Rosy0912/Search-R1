import copy,threading
from concurrent.futures import ThreadPoolExecutor
import pytest,torch
from search_r1.baselines.optimizer_transaction import transactional_adam_step,OptimizerCommitFault

def equal(a,b):
 if torch.is_tensor(a): assert torch.equal(a,b) and a.dtype==b.dtype and a.device==b.device
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.bfloat16])
@pytest.mark.parametrize('foreach',[False,True])
@pytest.mark.parametrize('amsgrad',[False,True])
def test_native_adam_exact_lazy_and_existing_state(dtype,foreach,amsgrad):
 p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=dtype));q=torch.nn.Parameter(p.detach().clone())
 a=torch.optim.AdamW([p],lr=1e-3,foreach=foreach,amsgrad=amsgrad);b=torch.optim.AdamW([q],lr=1e-3,foreach=foreach,amsgrad=amsgrad)
 for i in range(3):
  p.grad=torch.tensor([.2,.3],dtype=dtype);q.grad=p.grad.clone()
  assert transactional_adam_step(a)['committed'];b.step()
  equal(p,q);equal(a.state_dict(),b.state_dict())

def test_rejected_candidate_keeps_live_lazy_state_and_step_unchanged():
 p=torch.nn.Parameter(torch.tensor([1.]));a=torch.optim.AdamW([p],lr=1e-3)
 before=copy.deepcopy(a.state_dict());p.grad=torch.tensor([1e30]);original=p.clone()
 result=transactional_adam_step(a)
 assert not result['committed'];equal(p,original);equal(a.state_dict(),before)

def test_existing_state_rejection_preserves_buffers_and_step():
 p=torch.nn.Parameter(torch.tensor([1.]));a=torch.optim.AdamW([p]);p.grad=torch.tensor([1.]);a.step()
 before=copy.deepcopy(a.state_dict());original=p.clone();p.grad=torch.tensor([1e30])
 assert not transactional_adam_step(a)['committed'];equal(p,original);equal(a.state_dict(),before)

def test_tied_parameter_and_group_metadata_not_replaced():
 p=torch.nn.Parameter(torch.tensor([1.]));alias=p
 a=torch.optim.AdamW([dict(params=[p],lr=.01,initial_lr=.02,tag='group')]);identity=id(p);p.grad=torch.tensor([.1])
 assert transactional_adam_step(a)['committed']
 assert id(p)==identity and alias is p and a.param_groups[0]['tag']=='group' and a.param_groups[0]['initial_lr']==.02

def test_commit_fault_is_fatal_not_skip():
 p=torch.nn.Parameter(torch.tensor([1.]));a=torch.optim.AdamW([p]);p.grad=torch.tensor([.1])
 calls=[]
 def vote(ok):
  calls.append(ok);return ok if len(calls)<9 else False
 with pytest.raises(OptimizerCommitFault) as fault:transactional_adam_step(a,consensus_fn=vote)
 assert fault.value.commit_progress['local_commit_complete'] is True

@pytest.mark.parametrize('fault',['gradient','candidate'])
def test_eight_rank_collective_rejection_leaves_every_live_state_unchanged(fault):
 barrier=threading.Barrier(8);votes={};lock=threading.Lock()
 def run(rank):
  iteration=0
  def consensus(ok):
   nonlocal iteration
   with lock:votes.setdefault(iteration,{})[rank]=bool(ok)
   barrier.wait();result=all(votes[iteration].values());barrier.wait();iteration+=1;return result
  p=torch.nn.Parameter(torch.tensor([1.]));a=torch.optim.AdamW([p]);before=copy.deepcopy(a.state_dict())
  p.grad=torch.tensor([float('nan') if fault=='gradient' else 1e30]) if rank==3 else torch.tensor([.1])
  result=transactional_adam_step(a,consensus_fn=consensus)
  assert not result['committed'];assert p.item()==1.;equal(a.state_dict(),before)
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(run,range(8)))


def test_native_tensor_lr_multigroup_and_scheduler_metadata():
 p=torch.nn.Parameter(torch.tensor([1.]));q=torch.nn.Parameter(torch.tensor([2.]))
 a=torch.optim.AdamW([dict(params=[p],lr=torch.tensor(.01)),dict(params=[q],lr=.02)],foreach=False)
 b=torch.optim.AdamW([dict(params=[torch.nn.Parameter(p.clone())],lr=torch.tensor(.01)),dict(params=[torch.nn.Parameter(q.clone())],lr=.02)],foreach=False)
 for x,y in zip([p,q],[g['params'][0] for g in b.param_groups]):x.grad=torch.tensor([.1]);y.grad=x.grad.clone()
 assert transactional_adam_step(a)['committed'];b.step()
 for x,g in zip([p,q],b.param_groups):equal(x,g['params'][0])
 equal(a.state_dict(),b.state_dict());assert a._opt_called is True


def test_distinct_overlapping_parameter_alias_rejected_before_commit():
 storage=torch.tensor([1.,2.]);p=torch.nn.Parameter(storage);q=torch.nn.Parameter(storage)
 a=torch.optim.AdamW([p,q]);p.grad=torch.ones_like(p);q.grad=torch.ones_like(q)
 with pytest.raises(RuntimeError,match='share storage'):transactional_adam_step(a)
 assert not a.state and p.tolist()==[1.,2.]


def test_actor_commit_fault_returns_truthful_fatal_event():
 from types import SimpleNamespace as NS
 from verl.workers.actor.dp_actor import DataParallelPPOActor
 p=torch.nn.Parameter(torch.tensor([1.]));a=torch.optim.AdamW([p]);p.grad=torch.tensor([.1])
 actor=NS(actor_optimizer=a,_safety_writes=4,_safety_processed=5,_safety_expected_writes=20,
          _safety_skips=[],_safety_attempt_id=7,_safety_actual_outer=3,_safety_rollout_identity='x',_safety_reports=[])
 for local_complete,expected in [(True,5),(False,4)]:
  progress=dict(parameter_copies=1,state_entries=1,local_commit_complete=local_complete)
  event=DataParallelPPOActor._fatal_update(actor,'optimizer_commit_fault',progress)['fresh200_update_state']
  assert event['status']=='INFRASTRUCTURE_FAULT' and event['fatal'] is True
  assert event['optimizer_steps']==expected and event['confirmed_previous_optimizer_steps']==4
  assert event['failed_commit_progress']==progress and event['attempt_id']==7
  assert event['update_complete'] is False and event['attempt_complete'] is False
