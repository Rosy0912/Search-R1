import copy,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from tests.baselines.test_algorithms import manager,batch
from search_r1.baselines.rollout import SearchRollout,State
from search_r1.baselines.tree_audit import TreeAudit
from search_r1.baselines.controller import Controller

TREE=SimpleNamespace(rounds=1,expansions=2,leaves=3)
class TreeAuditTests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def generate(self,enabled,cap=4096):
  obj,worker,queries=manager(gpus=8,cap=cap,search_only=cap<100);engine=SearchRollout(obj);engine.rng.choice=lambda nodes:nodes[-1]
  if enabled:engine.audit=TreeAudit(self.root/'audit',self.root,1,1,['q1','q1','q2','q2'])
  out,src=engine.run(batch(obj.tokenizer,4),TREE);return engine,out,src,worker,queries
 def test_on_off_equivalence(self):
  a,b=self.generate(False),self.generate(True)
  for key in a[1].batch.keys():torch.testing.assert_close(a[1].batch[key],b[1].batch[key],rtol=0,atol=0)
  self.assertEqual(a[0].rng.getstate(),b[0].rng.getstate());self.assertEqual(a[0].cost,b[0].cost);self.assertEqual(a[4],b[4]);self.assertEqual(len(a[3].calls),len(b[3].calls))
  d=json.loads(b[0].audit.path.read_text());self.assertEqual(d['leaf_count'],12);self.assertEqual([(v['roots'],v['leaves']) for v in d['question_groups']],[(2,6),(2,6)]);self.assertNotIn('Paris',json.dumps(d));self.assertNotIn('capital',json.dumps(d))
 def test_truncation(self):
  e,*_=self.generate(True,45);self.assertTrue(any(not v['context_matches'] for v in json.loads(e.audit.path.read_text())['leaves']))
 def test_branch_pollution(self):
  for key,value in [('rolling',[9]),('source',99),('tree','bad'),('steps',[0])]:
   parent=State(0,[1],[1]);child=parent.branch();setattr(child,key,value);audit=TreeAudit(self.root/key,self.root,1,1,['q','q'])
   with self.assertRaisesRegex(ValueError,'branch_inheritance'):audit.branch(parent,child)
   self.assertEqual(json.loads(audit.path.read_text())['status'],'FAILED')
 def test_group_mask_steps_padding_uid_sources_pollution(self):
  e,out,src,_,_=self.generate(False);roots=[e.last_states[i] for i in [0,3,6,9]]
  for key in ['groups','mask','steps','padding','uid','sources']:
   audit=TreeAudit(self.root/key,self.root,1,1,['q']*4 if key=='groups' else ['q1','q1','q2','q2']);states=copy.deepcopy(e.last_states);data=copy.deepcopy(out);sources=src.copy()
   if key=='mask':states[0].model_mask[-1]=0
   if key=='steps':states[0].steps[-1]=-1
   if key=='padding':data.batch['info_mask'][0,-1]=1-data.batch['info_mask'][0,-1]
   if key=='uid':data.non_tensor_batch['tree_uid'][0]='bad'
   if key=='sources':sources[0]=99
   with self.assertRaises(ValueError):audit.finish(states,data,sources,roots,TREE,0)
 def test_controller_two_updates_disabled_zero_io(self):
  cfg=SimpleNamespace(method='tree_grpo',seed=42,context_alignment='report',gigpo=SimpleNamespace(anchor_mode='observation'),bapo=SimpleNamespace(patience=5),tree=TREE)
  def session(enabled):
   t=SimpleNamespace(config=SimpleNamespace(baselines=cfg),global_steps=1);c=Controller(t);rng=[];out=[]
   for step in [1,2]:
    t.global_steps=step;obj,_,_=manager(gpus=8);qa=SimpleNamespace(non_tensor_batch={'index':np.array(['q1','q1','q2','q2'])})
    with patch.dict(os.environ,{'SEARCH_R1_TREE_AUDIT_DIR':str(self.root/'controller') if enabled else '', 'SEARCH_R1_BASELINE_OUTPUT':str(self.root)}):
     if enabled:x,_=c.generate(obj,batch(obj.tokenizer,4),qa,{})
     else:
      with patch('pathlib.Path.open',side_effect=AssertionError('disabled IO')):x,_=c.generate(obj,batch(obj.tokenizer,4),qa,{})
    rng.append(c.tree_rng.getstate());out.append(x.batch)
   return rng,out
  a,b=session(False),session(True);self.assertEqual(a[0],b[0]);self.assertNotEqual(a[0][0],a[0][1])
  for x,y in zip(a[1],b[1]):
   for key in x.keys():torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
  self.assertTrue((self.root/'controller/step_2_rollout_2.json').exists())
 def test_overwrite_and_foreign_refused(self):
  self.generate(True)
  with self.assertRaises(FileExistsError):self.generate(True)
  with self.assertRaises(ValueError):TreeAudit(self.root.parent,self.root,1,1,['q','q'])

 def test_real_actor_gradient_and_parameter_on_off(self):
  from types import MethodType
  from tensordict import TensorDict
  from omegaconf import OmegaConf
  from verl.workers.actor.dp_actor import DataParallelPPOActor
  from search_r1.baselines.objectives import attach_policy_weights
  results=[]
  for enabled in [False,True]:
   engine,data,_,_,_=self.generate(enabled)
   width=data.batch['responses'].shape[1];mask=data.batch['info_mask'][:,-width:]*data.batch['attention_mask'][:,-width:]
   data.batch['loss_mask']=mask;data.batch['old_log_probs']=torch.zeros_like(mask).float();data.batch['advantages']=torch.arange(1,len(data)+1).float()[:,None].expand(-1,width)
   data.non_tensor_batch['uid']=np.array(['q1']*6+['q2']*6,dtype=object);attach_policy_weights(data,'tree_grpo');data.meta_info['temperature']=1.
   model=torch.nn.Linear(1,1,bias=False);model.weight.data.zero_();actor=DataParallelPPOActor.__new__(DataParallelPPOActor);actor.actor_module=model;actor.actor_optimizer=torch.optim.SGD(model.parameters(),lr=.1)
   actor.config=OmegaConf.create(dict(ppo_mini_batch_size=12,ppo_micro_batch_size=3,ppo_epochs=1,state_masking=True,use_kl_loss=False,use_dynamic_bsz=False,baseline_method='tree_grpo',entropy_coeff=0.,clip_ratio=.2,grad_clip=100.))
   def forward(self,micro_batch,temperature):
    x=self.actor_module(torch.ones(len(micro_batch),1)).expand(-1,width);return x*0+3,x
   actor._forward_micro_batch=MethodType(forward,actor)
   controller=Controller.__new__(Controller);controller.method='tree_grpo';controller.trainer=SimpleNamespace(global_steps=1);controller.latest_tree_audit=engine.audit
   with patch.dict(os.environ,{'SEARCH_R1_TREE_AUDIT_DIR':str(self.root/'audit') if enabled else ''}):
    controller.audit_actor_batch(data)
   captured=[];original_step=actor._optimizer_step
   def capture():
    captured.append(model.weight.grad.clone());return original_step()
   actor._optimizer_step=capture
   with patch.object(TensorDict,'cuda',lambda self,*a,**k:self):actor.update_policy(data)
   results.append((captured[0],model.weight.detach().clone()))
  self.assertTrue(torch.equal(results[0][0],results[1][0]));self.assertTrue(torch.equal(results[0][1],results[1][1]));self.assertGreater(float(results[0][0].abs().sum()),0)

 def test_real_padding_and_row_reorder(self):
  e,_,_,_,_=self.generate(False);states=copy.deepcopy(e.last_states)
  states[0].response=states[0].response[:-5];states[0].model_mask=states[0].model_mask[:-5];states[0].steps=states[0].steps[:-5]
  out,src=e.compose(states);roots=[e.last_states[i] for i in [0,3,6,9]]
  order=[11,0,7,1,8,2,9,3,10,4,6,5];out.reorder(torch.tensor(order));states=[states[i] for i in order];src=src[order]
  audit=TreeAudit(self.root/'padding_valid',self.root,1,1,['q1','q1','q2','q2']);audit.finish(states,out,src,roots,TREE,0)
  d=json.loads(audit.path.read_text());self.assertEqual(sum(r['padding_count'] for r in d['leaves']),5)
  idx=src.tolist().index(0);out.batch['step_ids'][idx,-1]=0
  audit=TreeAudit(self.root/'padding_bad',self.root,1,1,['q1','q1','q2','q2'])
  with self.assertRaises(ValueError):audit.finish(states,out,src,roots,TREE,0)

 def prepare_policy(self,data):
  from search_r1.baselines.objectives import attach_policy_weights
  width=data.batch['responses'].shape[1];data.batch['loss_mask']=data.batch['info_mask'][:,-width:].clone();data.batch['old_log_probs']=torch.zeros_like(data.batch['loss_mask']).float();data.batch['advantages']=torch.ones_like(data.batch['loss_mask']).float();data.non_tensor_batch['uid']=np.array(['q1']*6+['q2']*6,dtype=object);attach_policy_weights(data,'tree_grpo');return data
 def test_policy_reorder_and_mutation(self):
  engine,out,_,_,_=self.generate(True);data=self.prepare_policy(out);order=torch.tensor([11,0,7,1,8,2,9,3,10,4,6,5]);data.reorder(order)
  engine.audit.policy_batch(data,1);self.assertTrue(engine.audit.path.with_name(engine.audit.path.stem+'_policy_batch.json').exists())
  report=json.loads(engine.audit.path.with_name(engine.audit.path.stem+'_policy_batch.json').read_text());self.assertGreaterEqual(report['diagnostic_check_cost']['policy_wall_seconds'],0);self.assertGreaterEqual(report['diagnostic_check_cost']['policy_process_cpu_seconds'],0)
  for key in ['raw_mask','weights','oldlogprob','advantages','trajectory','tree','question','step','response','row_swap']:
   bad=copy.deepcopy(data);audit=copy.deepcopy(engine.audit);audit.directory=self.root/key;audit.directory.mkdir();audit.path=audit.directory/'rollout.json';audit.path.write_bytes(engine.audit.path.read_bytes());step=1
   width=bad.batch['responses'].shape[1];positions=torch.where(bad.batch['loss_mask'][0].bool())[0];pos=int(positions[0])
   if key=='raw_mask':bad.batch['loss_mask'][0,pos]=0
   if key=='weights':bad.batch['policy_weights'][0,pos]*=2
   if key=='oldlogprob':bad.batch['old_log_probs'][0,pos]=float('nan')
   if key=='advantages':bad.batch['advantages'][0,pos]=float('inf')
   if key=='trajectory':bad.non_tensor_batch['trajectory_uid'][0]='wrong'
   if key=='tree':bad.non_tensor_batch['tree_uid'][0]='wrong'
   if key=='question':bad.non_tensor_batch['uid'][0]='wrong'
   if key=='step':step=2
   if key=='response':bad.batch['responses'][0,pos]=777
   if key=='row_swap':bad.batch['step_ids'][0,pos]=99
   with self.assertRaises(ValueError):audit.policy_batch(bad,step)
 def test_policy_raw_info_preserved_with_empty_attention(self):
  engine,_,_,_,_=self.generate(False);states=copy.deepcopy(engine.last_states);states[0].response=[0]*len(states[0].response);out,src=engine.compose(states);roots=[engine.last_states[i] for i in [0,3,6,9]]
  audit=TreeAudit(self.root/'empty',self.root,1,1,['q1','q1','q2','q2']);audit.finish(states,out,src,roots,TREE,0);data=self.prepare_policy(out);self.assertGreater(int(data.batch['loss_mask'][0].sum()),0);self.assertEqual(int(data.batch['policy_weights'][0].sum()),0);data.batch['old_log_probs'][0]=float('nan');data.batch['advantages'][0]=float('inf');audit.policy_batch(data,1)
  result=json.loads(audit.path.with_name(audit.path.stem+'_policy_batch.json').read_text());self.assertEqual(result['empty_leaves'],1)
 def test_trainer_hook_order(self):
  import ast
  source=Path(__file__).parents[2]/'verl/trainer/ppo/ray_trainer.py';tree=ast.parse(source.read_text());names=[(n.func.attr,n.lineno) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
  mask=next(line for name,line in names if name=='_create_loss_mask');audit=next(line for name,line in names if name=='audit_actor_batch');update=next(line for name,line in names if name=='update_actor');self.assertLess(mask,audit);self.assertLess(audit,update)
