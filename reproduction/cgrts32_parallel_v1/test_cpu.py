"""Independent logical row/gradient checks and original contract integration."""
import ast,copy,hashlib,importlib.util,json,os,sys,unittest
from pathlib import Path
from .contract import *

class LayoutTests(unittest.TestCase):
 def test_32_groups_preserve_each_original_global128(self):
  order=reshard_original_eight_rows(2560);self.assertEqual(sorted(order),list(range(2560)))
  for op in range(20):
   expected={rank*320+op*16+j for rank in range(8) for j in range(16)}
   actual={order[rank*80+op*4+j] for rank in range(32) for j in range(4)}
   self.assertEqual(expected,actual);self.assertEqual(len(actual),128)
 def test_invalid_rows(self):
  for count in (512,2559,2561):
   with self.assertRaises(AssertionError):reshard_original_eight_rows(count)
 def test_actual_original_objective_and_20_adam_calls(self):
  import torch
  from search_r1.baselines.objectives import weighted_policy_loss
  torch.manual_seed(918);x=torch.randn(20,128,7,3,dtype=torch.float64);old=torch.randn(20,128,7,dtype=torch.float64)*.1;adv=torch.randn_like(old);mask=(torch.rand_like(old)>.3).double();weight=torch.rand_like(old)*mask
  def run(world,micro):
   w=torch.tensor([.03,-.02,.01],dtype=torch.float64,requires_grad=True);opt=torch.optim.AdamW([w],lr=1e-6,foreach=False);grads=[]
   local=128//world
   for op in range(20):
    opt.zero_grad()
    for rank in range(world):
     for begin in range(rank*local,(rank+1)*local,micro):
      sl=slice(begin,begin+micro);current=(x[op,sl]*w).sum(-1)
      loss=weighted_policy_loss(old[op,sl],current,adv[op,sl],mask[op,sl],weight[op,sl],.2,float(local),dual_clip=3.)[0]/world
      loss.backward()
    grads.append(w.grad.clone());opt.step()
   return w.detach(),grads,opt.state[w]
  direct=run(1,128)
  for world,micro in ((8,2),(32,1)):
   actual=run(world,micro);torch.testing.assert_close(actual[0],direct[0],atol=1e-13,rtol=1e-12)
   for a,b in zip(actual[1],direct[1]):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)
   self.assertEqual(int(actual[2]['step']),20)
   torch.testing.assert_close(actual[2]['exp_avg'],direct[2]['exp_avg'],atol=1e-13,rtol=1e-11)
 def test_real_original_config_nested_projection(self):
  from omegaconf import OmegaConf
  from .train import configuration
  p=read_plan();cfg=configuration(p,'cpu-fixture-host','10.1.1.1');original=OmegaConf.to_container(cfg,resolve=True)
  from .bootstrap import install
  install()
  from search_r1.baselines.controller import validate_config,validate_behavior_mu_config
  from search_r1.baselines.tree_fresh200 import validate_fresh200_config
  validate_config(cfg);validate_behavior_mu_config(cfg);validate_fresh200_config(cfg)
  self.assertEqual(original,OmegaConf.to_container(cfg,resolve=True))
  for path,value in [('trainer.nnodes',5),('actor_rollout_ref.actor.ppo_mini_batch_size',160),('actor_rollout_ref.actor.ppo_micro_batch_size',16),('actor_rollout_ref.actor.optim.lr',2e-6),('baselines.tree.continuation_budget_per_question',11)]:
   bad=copy.deepcopy(cfg);OmegaConf.update(bad,path,value)
   with self.assertRaises((AssertionError,ValueError)):validate_config(bad)
 def test_actual_ledger_32_and_old_inactive(self):
  from .bootstrap import install
  install();from search_r1.baselines.tree_guard_continue import validate_ledger
  def make(rows):return [dict(index=i+1,rows=rows,actual_writes=1,status='SUCCESS_ADAM_WRITE',local_row_indices=list(range(i*rows,(i+1)*rows))) for i in range(20)]
  validate_ledger(make(4),20)
  for size in (0,1,16):
   with self.assertRaises(ValueError):validate_ledger(make(size),20)
  broken=make(4);broken[1]['local_row_indices'][0]=0
  with self.assertRaises(ValueError):validate_ledger(broken,20)
  saved=os.environ.pop(ENV)
  try:validate_ledger(make(16),20)
  finally:os.environ[ENV]=saved
 def test_fit_adapter_original_complete_body(self):
  import types
  from .bootstrap import _trainer
  p=read_plan();source=Path(p['source'])/'verl/trainer/ppo/ray_trainer.py';tree=ast.parse(source.read_text());cls=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name=='RayPPOTrainer');fit=next(v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name=='fit')
  text=ast.get_source_segment(source.read_text(),fit)
  self.assertEqual(text.count('len(receipts) != 8'),1);self.assertIn('validate_completed_outer',text)
  # Import the actual module on CPU; no Ray initialization, model, or CUDA call.
  import verl.trainer.ppo.ray_trainer as m
  self.assertIn('admit_full_initialization',m.RayPPOTrainer.fit.__code__.co_names)
 def test_raw_generation_gate_before_semantic_removal(self):
  from types import SimpleNamespace
  from .runtime import raw_generation_record
  class Tokenizer:
   def decode(self,ids,skip_special_tokens=False):return ''.join('<|im_start|>' if t==151644 and not skip_special_tokens else 'a' for t in ids)
  item=SimpleNamespace(non_tensor_batch={'tree_raw_semantics':{'raw_ledger':[{'action':{'native_emission':{'raw_emitted_token_ids':[151644]*500}}}]}})
  self.assertTrue(raw_generation_record(item,Tokenizer())['long_im_start_repetition'])
  item.non_tensor_batch['tree_raw_semantics']['raw_ledger'][0]['action']['native_emission']['raw_emitted_token_ids']=[5]*500
  self.assertFalse(raw_generation_record(item,Tokenizer())['long_im_start_repetition'])
 def test_atomic_receipt_no_overwrite(self):
  import tempfile
  from .runtime import save
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'receipt.json';save(path,{'uid':'first'})
   with self.assertRaises(FileExistsError):save(path,{'uid':'second'})
   self.assertEqual(json.loads(path.read_text()),{'uid':'first'})
 def test_plan_gate_and_roles(self):
  p=read_plan();tasks=[]
  for i in range(2):tasks.append(dict(role=('PYTORCH_MASTER' if i==0 else 'PYTORCH_WORKER'),replicas=(1 if i==0 else 3),resource_spec=dict(accelerate_device_count=8,cpu_count=(64 if i==0 else 48),memory_gib=(900 if i==0 else 512),machine_types=['n2ls.ie.i60b'],rdma_name='ib-cluster-01')))
  validate_tasks(tasks);sub=dict(job=JOB,uid='fixture-not-real',plan_sha256='fixture');gate=dict(**sub,state='RUNNING',tasks=tasks);validate_gate(p,sub,gate)
  bad=copy.deepcopy(tasks);bad[1]['replicas']=4
  with self.assertRaises(AssertionError):validate_tasks(bad)
  bad=copy.deepcopy(gate);bad['uid']='wrong'
  with self.assertRaises(AssertionError):validate_gate(p,sub,bad)
 def test_shared_SDK_resource_objects_are_separated(self):
  from .operations import bind_layout
  resource=dict(accelerate_device_count=8,cpu_count=64,memory_gib=900,machine_types=['n2ls.ie.i60b'],rdma_name='ib-cluster-01')
  tasks=[dict(role='PYTORCH_MASTER',replicas=1,resource_spec=resource),dict(role='PYTORCH_WORKER',replicas=3,resource_spec=resource)]
  class Cluster:
   def _build_tasks(self,*args):return tasks
  c=Cluster();bind_layout(c);actual=c._build_tasks();validate_tasks(actual)
  self.assertIsNot(actual[0]['resource_spec'],actual[1]['resource_spec'])
  self.assertEqual(resource['memory_gib'],900)
if __name__=='__main__':unittest.main()
