"""Standalone CPU fixture copied from the reviewed test; no pytest dependency."""
import copy,json,random,sys,types
from pathlib import Path
import numpy as np
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

