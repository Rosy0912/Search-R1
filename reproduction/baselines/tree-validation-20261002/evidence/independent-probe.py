import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from types import SimpleNamespace
from tests.baselines.test_algorithms import manager,batch
from search_r1.baselines.rollout import SearchRollout
from search_r1.baselines.advantages import tree_grpo
from search_r1.baselines.objectives import attach_policy_weights,weighted_policy_loss
from verl import DataProto
out={}
m,w,q=manager(gpus=8); e=SearchRollout(m); e.rng.choice=lambda nodes:nodes[-1]
o,s=e.run(batch(m.tokenizer,2),SimpleNamespace(rounds=1,expansions=2,leaves=3))
assert s.tolist()==[0]*3+[1]*3 and len(q)==2
assert all(st.turns==2 and st.records[0]['text']=='<search>capital</search>' for st in e.last_states)
out['cross_root_continuations']={'cost':e.cost,'worker_rows':[len(c) for c in w.calls],'sources':s.tolist(),'distinct_trees':len(set(o.non_tensor_batch['tree_uid'])),'distinct_trajectories':len(set(o.non_tensor_batch['trajectory_uid']))}
r=torch.tensor([[1.],[0.],[0.],[0.],[1.],[1.]])
a=tree_grpo(r,torch.ones_like(r),['q']*6,['t0']*3+['t1']*3)
expected=(r-r.mean())/(r.std()+1e-6)
expected[:3]+=(r[:3]-r[:3].mean())/(r[:3].std()+1e-6)
expected[3:]+=(r[3:]-r[3:].mean())/(r[3:].std()+1e-6)
assert torch.allclose(a,expected)
p=torch.tensor([4,1,3,0,5,2]);ap=tree_grpo(r[p],torch.ones_like(r),['q']*6,[(['t0']*3+['t1']*3)[i] for i in p])
assert torch.allclose(ap,a[p])
out['six_leaf_advantage']={'values':a.squeeze().tolist(),'independent_zscores':True,'permutation_equivariant':True}
mask=torch.tensor([[1.,0.,0.],[1.,1.,1.],[0.,0.,0.]])
d=DataProto.from_dict({'responses':torch.zeros(3,3,dtype=torch.long),'attention_mask':mask.long(),'info_mask':mask.long()})
attach_policy_weights(d,'tree_grpo'); weights=d.batch['policy_weights']; x=torch.tensor(0.,requires_grad=True)
a=torch.tensor([[1.,0.,0.],[-1.,-1.,-1.],[0.,0.,0.]])
old=torch.where(mask.bool(),torch.zeros_like(mask),torch.full_like(mask,float('inf')))
loss,_,_=weighted_policy_loss(old,x.expand_as(mask),a,mask,weights,.2,3.)
loss.backward(); assert torch.isfinite(x.grad) and abs(x.grad.item())<1e-6
out['unequal_length_empty_leaf']={'loss':loss.item(),'gradient':x.grad.item(),'row_weight_sums':weights.sum(-1).tolist()}
m,w,q=manager(search_only=True,cap=80,turns=4,gpus=8);e=SearchRollout(m);o,s=e.run(batch(m.tokenizer))
assert e.cost['context_mismatches']>0
out['authorized_window_deviation']={'mismatches':e.cost['context_mismatches'],'retained_response':len(e.last_states[0].response),'rolling':len(e.last_states[0].rolling),'scoring':len(e.last_states[0].prompt+e.last_states[0].response)}
files=['search_r1/baselines/rollout.py','search_r1/baselines/advantages.py','search_r1/baselines/controller.py','search_r1/baselines/objectives.py','verl/workers/actor/dp_actor.py']
out['reviewed_sha256']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
print(json.dumps(out,indent=2))
