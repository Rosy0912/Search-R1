import copy,json,random,sys,types
from pathlib import Path
import numpy as np
import pytest
import torch
from verl import DataProto
from search_r1.llm_agent.generation import LLMGenerationManager
from validation_cost_hooks import wrap_trainer

class WG:
 def __init__(self,error=None):self.calls=[];self.error=error
 def generate_sequences(self,batch):
  random.random();np.random.rand();torch.rand(1)
  if self.error is not None:raise self.error
  rows=batch.batch['input_ids'].shape[0]
  out=DataProto.from_dict({'responses':torch.tensor([[10,11,12,0,0,0]]).repeat(rows,1)},meta_info={'test':torch.arange(rows)})
  self.calls.append((batch,out));return out

class Manager(LLMGenerationManager):
 def run_llm_loop(self,gen_batch,initial_input_ids):
  out=self._generate_with_gpu_padding(gen_batch);self.batch_search(['q']*len(gen_batch.batch['input_ids']));return out
 def _batch_search(self,queries):return {'result':[[] for _ in queries]}


def setup_case(error=None,coverage=8):
 class Trainer:
  def __init__(self):
   self.config=types.SimpleNamespace(do_search=True);self.global_steps=0;self.tokenizer=types.SimpleNamespace(pad_token_id=0,eos_token_id=2);self.actor_rollout_wg=WG(error);self.val_dataset=list(range(coverage));self.results={'val/evaluated_count/a':8,'val/f1/a':.2};self.outputs=[]
  def _prepare_final_validation(self):return 'final-ok'
  def _validate(self):
   manager=Manager.__new__(Manager);manager.actor_rollout_wg=self.actor_rollout_wg;manager.config=types.SimpleNamespace(num_gpus=8)
   for rows in [5,3]:
    batch=DataProto.from_dict({'input_ids':torch.tensor([[1,2]]).repeat(rows,1),'attention_mask':torch.ones(rows,2,dtype=torch.long),'position_ids':torch.tensor([[0,1]]).repeat(rows,1)})
    self.outputs.append(manager.run_llm_loop(batch,batch.batch['input_ids']))
   return self.results
 return Trainer


def rng():return (random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
def reset():random.seed(42);np.random.seed(42);torch.manual_seed(42)
def assert_rng(a,b):
 assert a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2])


def test_actual_uneven_padding_identity_rng_and_no_training_hook(tmp_path):
 reset();Plain=setup_case();plain=Plain();expected=plain._validate();off=rng()
 reset();Observed=setup_case();observed=Observed();old_loop=Manager.run_llm_loop;old_padding=Manager._generate_with_gpu_padding;old_search=Manager._batch_search;old_generate=observed.actor_rollout_wg.generate_sequences
 wrap_trainer(Observed,Manager,tmp_path)
 result=observed._validate();assert result is observed.results and result==expected;assert_rng(off,rng())
 assert Manager.run_llm_loop is old_loop and Manager._generate_with_gpu_padding is old_padding and Manager._batch_search is old_search
 assert observed.actor_rollout_wg.generate_sequences==old_generate and 'generate_sequences' not in vars(observed.actor_rollout_wg)
 for (b1,o1),(b2,o2) in zip(plain.actor_rollout_wg.calls,observed.actor_rollout_wg.calls):
  for key in b1.batch.keys():assert torch.equal(b1.batch[key],b2.batch[key])
  assert torch.equal(o1.batch['responses'],o2.batch['responses']) and torch.equal(o1.meta_info['test'],o2.meta_info['test'])
 summary=json.loads(next(tmp_path.glob('*.json')).read_text());assert summary['status']=='VALIDATION_COST_COMPLETE'
 c=summary['counters'];assert c['real_sample_rows']==8 and c['active_row_events']==8 and c['submitted_padding_row_events']==8 and c['submitted_rows']==c['returned_rows']==16
 assert c['returned_nonpad_model_token_events']==48 and c['submitted_prefill_token_events']==32 and c['http_attempts']==2 and c['http_query_events']==c['http_returned_query_rows']==8
 lines=next(tmp_path.glob('*.jsonl')).read_text();batch=observed.actor_rollout_wg.calls[0][0];observed.actor_rollout_wg.generate_sequences(batch);assert next(tmp_path.glob('*.jsonl')).read_text()==lines


def test_final_phase_and_pad_eos_convention(tmp_path):
 T=setup_case();t=T();t.tokenizer.eos_token_id=0;t.global_steps=500;wrap_trainer(T,Manager,tmp_path);assert t._prepare_final_validation()=='final-ok';t._validate()
 r=json.loads(next(tmp_path.glob('*.json')).read_text());assert r['phase']=='final' and r['pad_token_id']==r['eos_token_id']==0 and r['evaluated_rows']==8


def test_original_exception_identity_and_restoration(tmp_path):
 error=RuntimeError('original-error');T=setup_case(error);t=T();old=Manager.run_llm_loop;original=t.actor_rollout_wg.generate_sequences;wrap_trainer(T,Manager,tmp_path)
 with pytest.raises(RuntimeError) as exc:t._validate()
 assert exc.value is error and Manager.run_llm_loop is old and t.actor_rollout_wg.generate_sequences==original
 r=json.loads(next(tmp_path.glob('*.json')).read_text());assert r['status']=='VALIDATION_COST_FAILED' and r['evaluated_rows'] is None and r['counters']['generation_failed']==1


def test_coverage_fail_closed(tmp_path):
 T=setup_case(coverage=9);t=T();wrap_trainer(T,Manager,tmp_path)
 with pytest.raises(AssertionError):t._validate()
 assert json.loads(next(tmp_path.glob('*.json')).read_text())['status']=='VALIDATION_COST_FAILED'


def test_publication_failure_restores_and_preserves_original(tmp_path,monkeypatch):
 err=RuntimeError('original');T=setup_case(err);t=T();orig=t.actor_rollout_wg.generate_sequences;wrap_trainer(T,Manager,tmp_path)
 monkeypatch.setattr('validation_cost_hooks.os.link',lambda *a:(_ for _ in ()).throw(OSError('disk')))
 with pytest.raises(RuntimeError) as out:t._validate()
 assert out.value is err and t.actor_rollout_wg.generate_sequences==orig and not list(tmp_path.glob('*.json'))


def test_success_publication_failure_is_not_accepted(tmp_path,monkeypatch):
 T=setup_case();t=T();orig=t.actor_rollout_wg.generate_sequences;wrap_trainer(T,Manager,tmp_path)
 monkeypatch.setattr('validation_cost_hooks.os.link',lambda *a:(_ for _ in ()).throw(OSError('disk')))
 with pytest.raises(OSError):t._validate()
 assert t.actor_rollout_wg.generate_sequences==orig and not list(tmp_path.glob('*.json'))
