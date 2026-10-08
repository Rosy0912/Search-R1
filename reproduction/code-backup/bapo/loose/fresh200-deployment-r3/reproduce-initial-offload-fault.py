import os,sys,json
from pathlib import Path
assert os.uname().nodename=='h100-5'
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r3-20261006');assert sys.prefix==str(R/'env')
sys.path.insert(0,str(R/'repo'))
import torch
from torch.optim.lr_scheduler import LambdaLR
from search_r1.baselines.fresh200_contract import fresh200_lr_factor
from verl.utils.fsdp_utils import offload_fsdp_optimizer
p=torch.nn.Parameter(torch.ones(2));o=torch.optim.AdamW([p],lr=1e-6);s=LambdaLR(o,fresh200_lr_factor)
def snap():return dict(state_entries=len(o.state),states=[list(x) for x in o.state.values()],scheduler_epoch=s.last_epoch,lr=[g['lr'] for g in o.param_groups],lr_equal=all(g['lr']==1e-6/57 for g in o.param_groups))
before=snap();offload_fsdp_optimizer(o);after=snap();print(json.dumps(dict(torch=torch.__version__,source_commit='379b14c65e127cd3a71524f7783deaebafbbdb41',before=before,after=after,proves_empty_state_created=before['state_entries']==0 and after['state_entries']==1 and after['states']==[[]],scope='ACTUAL_TARGET_CPU_ORIGINAL_OFFLOAD_CALL_NOT_GPU_TRAINING')))
