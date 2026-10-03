"""Four-rank FSDP check of archived vs physical-split actual PPO methods.
Run manually with torch.distributed.run; does not launch the RL experiment.
"""
import os,sys,json
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import patch
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensordict import TensorDict
sys.path.insert(0,str(Path(__file__).parent))
from test_physical_micro import method,UP,patch_actor,extracted,algos,functional,append
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
torch.manual_seed(42+rank)
source=(UP/'workers/actor/dp_actor.py').read_text()
original=method(source,'update_policy');adapted=method(patch_actor(source),'update_policy')
step=extracted(source,'_optimizer_step',dict(torch=torch,FSDP=FSDP))
mask=(torch.rand(32,7,device='cuda')>.45).float();mask[rank*4:rank*4+4]=0
if os.environ.get('E1_CHECK_TRAJECTORY')=='1':mask[:,0]=1
batch=TensorDict(dict(responses=torch.zeros(32,7,dtype=torch.long,device='cuda'),input_ids=torch.randn(32,7,device='cuda'),attention_mask=mask,position_ids=torch.zeros(32,7,dtype=torch.long,device='cuda'),loss_mask=mask,old_log_probs=torch.randn(32,7,device='cuda')*.4,ref_log_prob=torch.randn(32,7,device='cuda')*.4,advantages=torch.randn(32,7,device='cuda')),batch_size=[32])
class Data:
 meta_info={'temperature':1.}
 def select(self,**kwargs):return SimpleNamespace(batch=batch)
def actor():
 model=torch.nn.Linear(1,1,device='cuda')
 with torch.no_grad():model.weight.fill_(.17);model.bias.fill_(-.2)
 model=FSDP(model,device_id=rank)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
 calls=[]
 def forward(micro_batch,temperature):
  calls.append(len(micro_batch));lp=model(micro_batch['input_ids'].unsqueeze(-1)).squeeze(-1)
  return torch.sigmoid(lp),lp
 obj=SimpleNamespace(config=SimpleNamespace(ppo_mini_batch_size=16,ppo_micro_batch_size=8,use_dynamic_bsz=False,state_masking=True,use_kl_loss=True,clip_ratio=.2,entropy_coeff=.001,kl_loss_type='low_var_kl',kl_loss_coef=.001,grad_clip=1.),actor_module=model,actor_optimizer=opt,_forward_micro_batch=forward,calls=calls)
 obj._optimizer_step=MethodType(step,obj);return obj
a,b=actor(),actor()
cap=int(os.environ.get('E1_CHECK_PHYSICAL','4'));assert cap in [1,2,4,8]
os.environ['E1_PHYSICAL_MICRO']=str(cap);os.environ['SEARCH_E1_TRAJECTORY_BATCHING']=os.environ.get('E1_CHECK_TRAJECTORY','0')
# Ray exposes one GPU per worker; torchrun exposes all four, so map the
# upstream TensorDict.cuda() default to this test worker's local GPU.
with patch.object(TensorDict, 'cuda', lambda self: self.to(torch.device('cuda',rank))):
 ma=original(a,Data());mb=adapted(b,Data())
sa=a.actor_module.state_dict();sb=b.actor_module.state_dict()
for k in sa:torch.testing.assert_close(sa[k],sb[k],atol=1e-6,rtol=1e-6)
for k in ma:torch.testing.assert_close(torch.tensor(ma[k]),torch.tensor(mb[k]),atol=1e-6,rtol=1e-6)
assert len(a.calls)==4 and len(b.calls)==32//cap
print(json.dumps(dict(rank=rank,result='FSDP_PHYSICAL_EQUIVALENCE_OK',original_calls=len(a.calls),physical_calls=len(b.calls))),flush=True)
dist.barrier();dist.destroy_process_group()
