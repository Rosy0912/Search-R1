"""Execute real clip helper's structural branch with eight mock FSDP ranks."""
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import pytest,torch
import search_r1.baselines.gradient_norm as norm


def test_one_rank_mixed_dtype_reaches_same_structural_collective_before_any_norm_reduce(monkeypatch):
 class FakeFSDP:
  _is_root=True;compute_device=torch.device('cpu');process_group=None
  def _assert_state(self,state):pass
  def parameters(self):return []
 monkeypatch.setattr(norm,'FSDP',FakeFSDP)
 import torch.distributed.fsdp._runtime_utils as runtime
 monkeypatch.setattr(runtime,'_lazy_init',lambda *args:None)
 local=threading.local();barrier=threading.Barrier(8);values={};lock=threading.Lock();counts={}
 def all_reduce(tensor,op=None,group=None):
  rank=local.rank;index=local.count;local.count+=1
  with lock:values.setdefault(index,{})[rank]=tensor.clone()
  barrier.wait()
  result=torch.stack(list(values[index].values())).max(0).values
  tensor.copy_(result);barrier.wait()
 monkeypatch.setattr(norm.dist,'all_reduce',all_reduce)
 def run(rank):
  local.rank=rank;local.count=0
  module=FakeFSDP()
  p=torch.nn.Parameter(torch.ones(2));q=torch.nn.Parameter(torch.ones(2,dtype=torch.float64 if rank==3 else torch.float32))
  p.grad=torch.ones_like(p);q.grad=torch.ones_like(q)
  module._all_handles=[NS(uses_sharded_strategy=True,_use_orig_params=True,flat_param=NS(_params=[p,q]))]
  with pytest.raises(norm.GradientStructureFault,match='Collective clipping structure'):norm.clip_grad_norm64(module,1.)
  assert torch.equal(p.grad,torch.ones_like(p)) and torch.equal(q.grad,torch.ones_like(q))
  counts[rank]=local.count
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(run,range(8)))
 assert list(counts.values())==[1]*8 # No rank reaches finite/norm reduction or optimizer write.


@pytest.mark.parametrize('prior_commit_fault',[False,True])
def test_one_rank_ledger_io_failure_returns_all_rank_fatal_truth(monkeypatch,tmp_path,prior_commit_fault):
 import copy
 from pathlib import Path
 import search_r1.baselines.policy_safety as safety
 local=threading.local();barrier=threading.Barrier(8);flags={};events={}
 monkeypatch.setattr(safety,'distributed',lambda group=None:True)
 monkeypatch.setattr(safety,'collective_device',lambda group=None:torch.device('cpu'))
 monkeypatch.setattr(safety.dist,'get_world_size',lambda group=None:8)
 def reduce(flag,op=None,group=None):
  flags[local.rank]=flag.clone();barrier.wait();flag.copy_(torch.stack(list(flags.values())).min(0).values);barrier.wait()
 def gather(result,event,group=None):
  events[local.rank]=copy.deepcopy(event);barrier.wait()
  result[:]=[events[i] for i in range(8)];barrier.wait()
 monkeypatch.setattr(safety.dist,'all_reduce',reduce);monkeypatch.setattr(safety.dist,'all_gather_object',gather)
 original=Path.open
 def open_(path,*args,**kwargs):
  if path.name=='rank3.json':raise OSError('fixture rank-local IO failure')
  return original(path,*args,**kwargs)
 monkeypatch.setattr(Path,'open',open_)
 def run(rank):
  local.rank=rank
  event=dict(status='ABORTED',optimizer_steps=2,rank=rank,attempt_id=9)
  if prior_commit_fault:
   event.update(status='INFRASTRUCTURE_FAULT',update_complete=False,attempt_complete=False,postabort=True,fatal=True,optimizer_steps=4 if rank<7 else 3,
       confirmed_previous_optimizer_steps=3,reason='optimizer_commit_fault',
       failed_commit_progress=dict(parameter_copies=2,state_entries=1,local_commit_complete=rank<7))
  local_event,all_events=safety.persist_rank_update_event(tmp_path/f'rank{rank}.json',event)
  assert local_event['status']=='INFRASTRUCTURE_FAULT' and not local_event['update_complete']
  assert len(all_events)==8
  if prior_commit_fault:
   for i,e in enumerate(all_events):
    assert e['optimizer_steps']==(4 if i<7 else 3) and e['confirmed_previous_optimizer_steps']==3
    assert e['reason']=='optimizer_commit_fault'
    assert e['failed_commit_progress']==dict(parameter_copies=2,state_entries=1,local_commit_complete=i<7)
   assert all_events[3]['additional_infrastructure_faults'][0]['ledger_io_error'] is not None
   assert all_events[0]['additional_infrastructure_faults'][0]['ledger_io_error'] is None
  else:
   assert all(e['optimizer_steps']==2 for e in all_events)
   assert all_events[3]['failed_commit_progress']['ledger_io_error'] is not None
   assert all_events[0]['failed_commit_progress']['ledger_io_error'] is None
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(run,range(8)))
