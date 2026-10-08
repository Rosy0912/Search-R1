import inspect,torch,json
from unittest.mock import patch
from verl.utils.fsdp_utils import offload_fsdp_optimizer,load_fsdp_optimizer
from search_r1.baselines.fresh200_contract import fresh200_lr_factor
from search_r1.baselines.optimizer_transaction import transactional_adam_step
ns={'torch':torch}
for fn in (offload_fsdp_optimizer,load_fsdp_optimizer):
    code=inspect.getsource(fn).replace('state = optimizer.state[param]','state = optimizer.state.get(param)\n            if state is None:\n                continue')
    exec(code,ns)
p=torch.nn.Parameter(torch.tensor([1.]));o=torch.optim.AdamW([p],lr=1e-6);s=torch.optim.lr_scheduler.LambdaLR(o,fresh200_lr_factor)
assert len(o.state)==0 and s.last_epoch==0 and o.param_groups[0]['lr']==1e-6/57
with patch('torch.cuda.empty_cache'):offload_fsdp_optimizer(o)
assert len(o.state)==1 and o.state[p]=={}
print(json.dumps(dict(original_bug='CONFIRMED',torch=torch.__version__,state_entries_after_original=1,scheduler_epoch=s.last_epoch,lr=o.param_groups[0]['lr'])))
p=torch.nn.Parameter(torch.tensor([1.]));idle=torch.nn.Parameter(torch.tensor([2.]));o=torch.optim.AdamW([p,idle],lr=1e-6)
with patch('torch.cuda.empty_cache'):
    ns['offload_fsdp_optimizer'](o);ns['load_fsdp_optimizer'](o,'cpu')
assert len(o.state)==0
p.grad=torch.tensor([.25]);assert transactional_adam_step(o,consensus_fn=lambda x:x)['committed']
assert p in o.state and idle not in o.state and o.state[p]['step'].item()==1
before={k:v.clone() if torch.is_tensor(v) else v for k,v in o.state[p].items()};keys=set(o.state)
with patch('torch.cuda.empty_cache'):
    ns['offload_fsdp_optimizer'](o);ns['load_fsdp_optimizer'](o,'cpu')
assert set(o.state)==keys and all(torch.equal(o.state[p][k],v) for k,v in before.items())
o.state[idle]={}
with patch('torch.cuda.empty_cache'):ns['offload_fsdp_optimizer'](o);ns['load_fsdp_optimizer'](o,'cpu')
assert idle in o.state and o.state[idle]=={}
print('CANDIDATE_GET_NONE_SKIP_PASS: fresh state untouched, populated buffers preserved, idle param lazy state preserved, explicit existing empty state retained, real shadow Adam write correct')
