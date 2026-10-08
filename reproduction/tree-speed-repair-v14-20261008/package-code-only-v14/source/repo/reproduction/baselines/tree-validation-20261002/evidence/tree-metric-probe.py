import sys,inspect,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
import numpy as np
from tests.baselines import test_paper_metric_logging as fixture
namespace=fixture.__dict__.copy()
exec(inspect.getsource(fixture.run).replace("'gigpo'","'tree_grpo'"),namespace)
small,sw=namespace['run'](1);large,lw=namespace['run'](2)
torch.testing.assert_close(sw,lw)
for k in ('actor/pg_loss','actor/entropy_loss','actor/kl_loss','actor/pg_clipfrac','actor/ppo_kl'):
 np.testing.assert_allclose(small[k],large[k],atol=1e-7)
from search_r1.baselines.objectives import weighted_policy_loss
mask=torch.tensor([[1.,0.,0.],[1.,1.,1.],[0.,0.,0.],[1.,1.,0.]])
weights=mask/mask.sum(-1,keepdim=True).clamp_min(1)
old=torch.zeros_like(mask);current=torch.tensor([[.5,0,0],[-.4,-.4,-.4],[0,0,0],[.1,.1,0.]])
a=torch.tensor([[1.,0,0],[-1.,-1.,-1.],[0,0,0],[2.,2.,0.]])
full=weighted_policy_loss(old,current,a,mask,weights,.2,4.)
# Unequal rows and an empty leaf: DP averaging must preserve minibatch
# normalized loss, and clip/PPO-KL average over effective leaf mass.
parts=[torch.tensor([0]),torch.tensor([1,2]),torch.tensor([3])]
loss=clip=kl=0.
for rows in parts:
 x,y,z=weighted_policy_loss(old[rows],current[rows],a[rows],mask[rows],weights[rows],.2,4.)
 mass=weights[rows].sum()
 loss+=x;clip+=y*mass/weights.sum();kl+=z*mass/weights.sum()
assert torch.allclose(loss,full[0]) and torch.allclose(clip,full[1]) and torch.allclose(kl,full[2])
print(json.dumps({'tree_real_actor_micro1':small,'tree_real_actor_micro2':large,'parameter_delta_equal':True,'unequal_length_empty_leaf_metric_mass':weights.sum().item(),'unequal_micros_and_full_loss_clip_kl_equal':True,'reference_loss_clip_kl':[x.item() for x in full]},indent=2))
