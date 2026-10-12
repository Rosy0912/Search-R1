import sys, importlib.util, hashlib
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd()/'tests/baselines'))
import torch, numpy as np
from search_r1.baselines.advantages import gigpo
from search_r1.baselines.rollout import SearchRollout
from verl.trainer.ppo.core_algos import compute_policy_loss
from test_algorithms import manager,batch
s=importlib.util.spec_from_file_location('author_gigpo','/tmp/gigpo-core-fixed-audit.py'); author=importlib.util.module_from_spec(s); s.loader.exec_module(author)
r=torch.tensor([[0.,1.],[0.,0.]])
m=torch.ones_like(r); steps=torch.tensor([[0,1],[0,1]])
for norm in [False,True]:
 actual=gigpo(r,m,steps,[['root','obs'],['root','obs']],['q','q'],normalize=norm)
 # author one row/action, episode outcome repeated on each step
 scalar=torch.tensor([[1.],[1.],[0.],[0.]])
 q=torch.tensor([.95,1.,0.,0.]); idx=np.array(['q']*4); traj=np.array(['a','a','b','b'])
 expected,_=author.compute_gigpo_outcome_advantage(scalar,q,torch.ones_like(scalar),np.array(['root','obs','root','obs']),idx,traj,mode='mean_std_norm' if norm else 'mean_norm')
 # Equal episode lengths: only std mode differs via official duplicated step std
 print('normalize',norm,'ours',actual.tolist(),'author',expected.reshape(2,2).tolist())
 if not norm: torch.testing.assert_close(actual,expected.reshape(2,2))
p=torch.tensor(0.,requires_grad=True)
loss,cf,kl=compute_policy_loss(torch.zeros((1,1)),p.expand(1,1),torch.zeros((1,1)),torch.zeros((1,1)),.2)
loss.backward(); print('ZERO_ACTION loss',loss.item(),'clipfrac',cf.item(),'kl',kl.item(),'gradient',p.grad.item())
obj,w,_=manager(gpus=1,cap=45,turns=4,search_only=True)
e=SearchRollout(obj); out,_=e.run(batch(obj.tokenizer))
print('CONTEXT_CAP mismatch_count',e.cost['context_mismatches'],'records',out.non_tensor_batch['action_records'][0])
print('GENERATED_PREFIXES',[obj.tokenizer.decode(x.batch['input_ids'][0]) for x in w.calls])
print('SCORING_PREFIX',obj.tokenizer.decode(out.batch['prompts'][0]),'retained_response',obj.tokenizer.decode(out.batch['responses'][0]))
# Variable episode length: official cross-step episode mean vs per-episode mean
sr=torch.tensor([[1.],[1.],[0.]])
print('AUTHOR_LENGTH_WEIGHTED episode',author.episode_norm_reward(sr,torch.ones_like(sr),np.array(['q']*3),np.array(['a','a','b'])).flatten().tolist())
print('OURS_PER_EPISODE episode',gigpo(torch.tensor([[1.],[0.]]),torch.ones(2,1),torch.zeros(2,1,dtype=torch.long),[['a'],['b']],['q','q'],omega=0).flatten().tolist())
# inherited microbatch/rank token weighting
p=torch.tensor(0.,requires_grad=True)
a=compute_policy_loss(torch.zeros(1,1),p.expand(1,1),torch.ones(1,1),torch.ones(1,1),.2)[0]
b=compute_policy_loss(torch.zeros(1,3),p.expand(1,3),torch.ones(1,3)*3,torch.ones(1,3),.2)[0]
((a+b)/2).backward(); print('LOCAL_MEAN gradient',p.grad.item(),'global_token_mean_expected',-2.5)
