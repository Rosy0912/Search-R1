"""CPU-only digest roundtrip. Does not test actual FSDP loading or RL."""
import copy, json
from types import SimpleNamespace
import torch
from checkpoint_observer import full_state_digest

model=torch.nn.Linear(4,2)
model.register_buffer('persistent',torch.arange(3,dtype=torch.float32))
model.register_buffer('not_saved',torch.arange(2,dtype=torch.float32),persistent=False)
optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
model(torch.ones(1,4)).square().sum().backward()
optimizer.step();optimizer.zero_grad();scheduler.step()
manager=SimpleNamespace(model=model,optimizer=optimizer,lr_scheduler=scheduler)
rng=torch.get_rng_state().clone()
before=full_state_digest(manager)
model_state=copy.deepcopy(model.state_dict())
optimizer_state=copy.deepcopy(optimizer.state_dict())
scheduler_state=copy.deepcopy(scheduler.state_dict())
assert torch.equal(rng,torch.get_rng_state())
with torch.no_grad():
    next(model.parameters()).add_(1.)
assert full_state_digest(manager)!=before
model.load_state_dict(model_state)
optimizer.load_state_dict(optimizer_state)
scheduler.load_state_dict(scheduler_state)
assert full_state_digest(manager)==before
assert torch.equal(rng,torch.get_rng_state())
print(json.dumps(dict(synthetic_cpu_only=True,actual_fsdp_reload=False,real_rl_updates=0,
    complete_local_master_adam_scheduler_digest_roundtrip=True,rng_unchanged=True,passed=True)))
