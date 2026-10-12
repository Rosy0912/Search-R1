import sys, importlib.util, json
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-R1-RL-Baselines')
spec=importlib.util.spec_from_file_location('fixtures','/Users/shaoshitong/Documents/Search-R1-RL-Baselines/tests/baselines/test_algorithms.py'); f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
from types import SimpleNamespace
from search_r1.baselines.rollout import SearchRollout
from search_r1.baselines.advantages import tree_grpo
from verl.trainer.ppo.core_algos import compute_policy_loss
import torch
result={}
obj,worker,queries=f.manager(gpus=8)
e=SearchRollout(obj);e.rng.choice=lambda nodes:nodes[-1]
out,src=e.run(f.batch(obj.tokenizer,2), SimpleNamespace(rounds=1,expansions=2,leaves=3))
result['two_roots_8GPU']={'sources':src.tolist(),'tree_ids':len(set(out.non_tensor_batch['tree_uid'])),'cost':e.cost,'worker_batch_sizes':[len(x) for x in worker.calls],'retrieval_queries':len(queries)}
obj,worker,_=f.manager(search_only=True,cap=80,turns=4,gpus=1)
e=SearchRollout(obj);e.run(f.batch(obj.tokenizer));result['cap80']={'cost':e.cost,'retained_response_len':len(e.last_states[0].response),'rollout_context_len':len(e.last_states[0].rolling),'scoring_context_len':len(e.last_states[0].prompt+e.last_states[0].response),'equal':e.last_states[0].rolling==e.last_states[0].prompt+e.last_states[0].response}
r=torch.tensor([[1.],[0.],[0.],[0.]]);m=torch.ones_like(r);u=['q']*4;t=['a','a','b','b'];a=tree_grpo(r,m,u,t);p=[2,0,3,1]
result['zscore']={'values':a[:,0].tolist(),'permutation_equivariant':bool(torch.allclose(tree_grpo(r[p],m[p],[u[x] for x in p],[t[x] for x in p]),a[p])),'singleton':tree_grpo(torch.tensor([[1.]]),torch.ones(1,1),['q'],['a']).tolist()}
x=torch.tensor(0.,requires_grad=True);mask=torch.tensor([[1.,0.,0.],[1.,1.,1.]]);adv=torch.tensor([[1.,0.,0.],[-1.,-1.,-1.]])
l,_,_=compute_policy_loss(torch.zeros_like(mask),x.expand_as(mask),adv,mask,.2);l.backward();result['PPO_weight']={'loss':l.item(),'grad':x.grad.item(),'paper_equal_trajectory_grad':0.}
print(json.dumps(result,indent=2))
