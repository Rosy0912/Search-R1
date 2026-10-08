"""CPU observer contracts only: fake CUDA stats never certify hardware."""
import json,os,types
import pytest
import torch
from resource_audit import pre_backward_memory,MEMORY_STAT_KEYS,observed_update

def actor_context(tmp_path):
 return types.SimpleNamespace(_r1_forward_diagnostic_context={'directory':str(tmp_path),'rank':3,'job':'CPU_NOT_GPU','pid':os.getpid(),'starttime_ticks':1,'cgroup':'CPU_NOT_AUTHORITY','actual_outer_update_ordinal':1})
def stats():return {key:index+1 for index,key in enumerate(MEMORY_STAT_KEYS)}
def test_raw_original_stats_append_only_no_completed_update(tmp_path):
 actor=actor_context(tmp_path);calls=[];values=stats()
 fake=types.SimpleNamespace(cuda=types.SimpleNamespace(memory_stats=lambda:(calls.append('read') or values)))
 first=pre_backward_memory(actor,fake,1);second=pre_backward_memory(actor,fake,2)
 path=tmp_path/'actor-pre-backward-memory-rank3.jsonl'
 actual=[json.loads(line) for line in path.read_text().splitlines()]
 assert actual==[first,second] and calls==['read','read']
 assert first['original_cuda_memory_stats']==values and first['micro_forward_ordinal']==1
 assert first['scope']=='MICRO_FORWARD_COMPLETED_BEFORE_BACKWARD_NOT_COMPLETED_UPDATE_NOT_OPTIMIZER_STEP'
 assert path.stat().st_mode&0o777==0o600 and 'optimizer_step_records' not in first

def test_actual_observed_forward_record_survives_later_backward_failure(tmp_path,monkeypatch):
 import resource_audit
 actor=actor_context(tmp_path);actor.actor_optimizer=types.SimpleNamespace(param_groups=[{'lr':0.}])
 forward=lambda:(torch.tensor([1.]),);step=lambda:torch.tensor(0.)
 actor._forward_micro_batch=forward;actor._optimizer_step=step
 monkeypatch.setattr(resource_audit,'sampled_parameters',lambda actor:'CPU_SAMPLE')
 monkeypatch.setattr(torch.cuda,'memory_stats',stats)
 data=types.SimpleNamespace(batch={k:torch.tensor([1.]) for k in ('old_log_probs','advantages','loss_mask','ref_log_prob')})
 def original(data):
  actor._forward_micro_batch()
  assert (tmp_path/'actor-pre-backward-memory-rank3.jsonl').is_file()
  raise RuntimeError('original backward failure')
 with pytest.raises(RuntimeError,match='original backward failure'):observed_update(actor,original,data)
 assert actor._forward_micro_batch is forward and actor._optimizer_step is step
 assert len((tmp_path/'actor-pre-backward-memory-rank3.jsonl').read_text().splitlines())==1
 assert not list(tmp_path.glob('actor-updates*'))

@pytest.mark.parametrize('invalid',[True,-1,float('inf')])
def test_invalid_raw_stats_not_faked_or_published(tmp_path,invalid):
 values=stats();values[MEMORY_STAT_KEYS[0]]=invalid
 fake=types.SimpleNamespace(cuda=types.SimpleNamespace(memory_stats=lambda:values))
 with pytest.raises(ValueError):pre_backward_memory(actor_context(tmp_path),fake,1)
 assert not list(tmp_path.iterdir())

def test_no_context_does_not_touch_cuda():
 fake=types.SimpleNamespace(cuda=types.SimpleNamespace(memory_stats=lambda:pytest.fail('Unexpected CUDA query')))
 assert pre_backward_memory(types.SimpleNamespace(),fake,1) is None


def test_actual_micro_shape_total_nnz_journal(tmp_path,monkeypatch):
 import resource_audit
 actor=actor_context(tmp_path);actor.actor_optimizer=types.SimpleNamespace(param_groups=[{'lr':0.}]);actor._forward_micro_batch=lambda micro_batch,temperature:(torch.ones(2),);actor._optimizer_step=lambda:torch.tensor(0.)
 monkeypatch.setattr(resource_audit,'sampled_parameters',lambda actor:'CPU_NOT_FULL_HASH')
 monkeypatch.setattr(torch.cuda,'memory_stats',stats)
 monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0);monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
 from tensordict import TensorDict
 batch=TensorDict({'input_ids':torch.ones((2,8),dtype=torch.long),'attention_mask':torch.tensor([[1]*8,[1]*6+[0]*2],dtype=torch.long)},batch_size=[2])
 before={k:v.clone() for k,v in batch.items()}
 class Data:
  batch={k:torch.tensor([1.]) for k in ('old_log_probs','advantages','loss_mask','ref_log_prob')}
  def __len__(self):return 2
 data=Data()
 def original(data):return actor._forward_micro_batch(batch,1.)
 observed_update(actor,original,data)
 actual=json.loads((tmp_path/'actor-pre-backward-memory-rank3.jsonl').read_text())['actual_micro_input']
 assert actual['input_ids_shape']==[2,8] and actual['attention_mask_shape']==[2,8] and actual['attention_mask_sum_total_nnz']==14
 assert all(torch.equal(batch[k],v) for k,v in before.items())
