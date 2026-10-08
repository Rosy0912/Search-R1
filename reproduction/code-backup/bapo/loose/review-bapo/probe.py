import sys, ast, math, subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
repo=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-3/work/bapo')
sys.path[:0]=[str(repo),str(repo/'tests/baselines')]
from test_algorithms import manager,batch,padded,CharTokenizer,Worker
from test_models_and_driver import config
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.main_ppo import RewardManager
from search_r1.baselines.controller import Controller
from search_r1.baselines.bapo import BoundaryStage, answer_f1, decode_answers
ns={}; source=Path(__file__).with_name('author-reward.py').read_text();tree=ast.parse(source)
names={'normalize_answer','get_f1_score'}
selected=[node for node in tree.body if isinstance(node,ast.Import) and any(x.name in ['re','string'] for x in node.names) or isinstance(node,ast.ImportFrom) and node.module in ['collections','typing'] or isinstance(node,ast.FunctionDef) and node.name in names]
exec(compile(ast.Module(body=selected,type_ignores=[]),'author','exec'),ns)
for p,t in [('yes indeed','yes'),('no answer','no'),('the','a'),('Paris','paris'),('x y','x'),('noanswer','noanswer'),('','')]:
 assert answer_f1(p,[t])==ns['get_f1_score'](p,[t]),(p,t)
print('AUTHOR_SCORER_DIFF 7/7 match')
m,w,q=manager(gpus=1);calls=[]
def gen(data):
 calls.append(len(data)); n=len(data); ix=len(calls)
 answers=(['wrong']*5+['Paris']*5) if ix==1 else ['wrong' if ix==2 else 'I do not know']*n
 return DataProto.from_dict({'responses':padded([m.tokenizer.encode('<answer>'+a+'</answer>') for a in answers],0)},meta_info={'temperature':1.,'micro_batch_size':1,'use_dynamic_bsz':False,'max_token_len':4096})
w.generate_sequences=gen
qa=DataProto.from_dict(tensors={'dummy':torch.zeros(10,1)},non_tensors={'index':np.array(['q0']*5+['q1']*5,dtype=object),'reward_model':np.array([{'ground_truth':{'target':['Paris']}}]*10,dtype=object)})
c=Controller.__new__(Controller);c.config=SimpleNamespace(gigpo=SimpleNamespace(anchor_mode='observation'),seed=42,context_alignment='report',tree=None,bapo=SimpleNamespace(resample_attempts=2)); c.method='bapo';c.stage=BoundaryStage(early=False);c.ranker=None;c.trainer=SimpleNamespace(tokenizer=m.tokenizer)
metrics={};out,sources=c.generate(m,batch(m.tokenizer,10),qa,metrics)
assert calls==[10,5,5]; assert sources.tolist()==list(range(10));assert metrics['rollout/trajectories']==20
assert decode_answers(out,m.tokenizer,True)==['I do not know']*5+['Paris']*5
print('LATE_RESAMPLE calls',calls,'sources',sources.tolist(),'cost',metrics)
# Exact fit counterexample: two all-empty skipped actor iterations still exhaust budget.
cfg=config('bapo');cfg.trainer.total_epochs=1;cfg.trainer.test_freq=-1;cfg.trainer.save_freq=-1;cfg.trainer.critic_warmup=0;cfg.actor_rollout_ref.actor.use_kl_loss=True;cfg.actor_rollout_ref.rollout.n_agent=2
tr=RayPPOTrainer.__new__(RayPPOTrainer);tr.config=cfg;tr.tokenizer=CharTokenizer();tr.logger=SimpleNamespace(log=lambda **kw:None);wk=Worker(tr.tokenizer);wk.world_size=1;actor_calls=[]
def update(d):
 actor_calls.append(1);return DataProto(meta_info={'metrics':{'actor/optimizer_steps':0.,'actor/skipped_empty_minibatches':1.}})
wk.update_actor=update;tr.actor_rollout_wg=wk
ids=padded([tr.tokenizer.encode('Question: capital?')],0,True)
row={'input_ids':ids,'attention_mask':(ids!=0).long(),'position_ids':torch.arange(ids.shape[1])[None,:],'index':np.array(['q'],dtype=object),'reward_model':np.array([{'ground_truth':{'target':['Paris']}}],dtype=object),'data_source':np.array(['nq'],dtype=object)}
tr.train_dataloader=[row,row,row];tr.use_reference_policy=tr.use_rm=tr.use_critic=False;tr.total_training_steps=3;tr.val_reward_fn=None;tr.reward_fn=RewardManager(tr.tokenizer,0);tr.kl_ctrl=SimpleNamespace(value=0.)
ctr=Controller.__new__(Controller);ctr.trainer=tr;ctr.config=cfg.baselines;ctr.method='bapo';ctr.stage=BoundaryStage();ctr.ranker=ctr.teacher=None;tr.baseline_controller=ctr
frozen = ast.parse(subprocess.check_output(['git','-C',str(repo),'show','e07cb3e:search_r1/baselines/controller.py'],text=True))
old_fn=next(n for cls in frozen.body if isinstance(cls,ast.ClassDef) and cls.name=='Controller' for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='after_actor')
old_ns={};exec(compile(ast.Module(body=[old_fn],type_ignores=[]),'frozen-e07cb3e','exec'),old_ns)
ctr.after_actor=lambda d,m:old_ns['after_actor'](ctr,d,m)
with patch('search_r1.llm_agent.generation.LLMGenerationManager.batch_search',lambda self,queries:['Paris']*len(queries)):tr.fit()
assert len(actor_calls)==2 and tr.global_steps==3 and ctr.stage.best==float('-inf')
print('OUTER_BUDGET_COUNTEREXAMPLE iteration_budget=2 actor_calls=2 actual_outer_updates=0 terminal_global_steps=3 stage_observations=0')
