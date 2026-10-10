"""Synthetic CPU exercise of actual Adam-call counting and complete state comparison."""
import json,os,sys,types
from pathlib import Path
import torch
import runtime_audit as audit
ROOT=Path(os.environ['COSEARCH_AUDIT_ROOT']);ROOT.mkdir(exist_ok=True)
torch.distributed.get_rank=lambda:0
torch.distributed.get_world_size=lambda:4
class Worker:
    role='actor_rollout'
    checkpoint_manager=types.SimpleNamespace(get_rng_state=lambda:{'cpu':torch.get_rng_state()})
    def __init__(self):
        self.p=torch.nn.Parameter(torch.tensor([1.,2.]));self.actor_optimizer=torch.optim.AdamW([self.p],lr=1e-6)
        self.actor_lr_scheduler=torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer,lambda step:min(step/57,1))
    def update_actor(self,data):
        for _ in range(data.steps):
            self.actor_optimizer.zero_grad();self.p.square().sum().backward();self.actor_optimizer.step()
        self.actor_lr_scheduler.step()
        return types.SimpleNamespace(meta_info={'metrics':{'loss':float(self.p.detach().square().sum())}})
    def save_checkpoint(self,local_path,*a,**kw):
        Path(local_path).mkdir(exist_ok=True)
        torch.save({'model':self.p.detach(),'optimizer':self.actor_optimizer.state_dict(),'scheduler':self.actor_lr_scheduler.state_dict()},Path(local_path)/'synthetic.pt')
    def load_checkpoint(self,local_path,*a,**kw):
        if local_path is None:return
        s=torch.load(Path(local_path)/'synthetic.pt',weights_only=False)
        self.p.data.copy_(s['model']);self.actor_optimizer.load_state_dict(s['optimizer']);self.actor_lr_scheduler.load_state_dict(s['scheduler'])
class Data:
    steps=20;meta_info={'cosearch_outer_step':1}
    def __len__(self):return 640
module=types.ModuleType('verl.workers.fsdp_workers');module.ActorRolloutRefWorker=Worker
sys.modules['verl.workers.fsdp_workers']=module
fixes=types.ModuleType('runtime_fixes');fixes.install_worker=lambda:None
sys.modules['runtime_fixes']=fixes
audit.install();w=Worker();w.update_actor(Data());w.save_checkpoint(str(ROOT/'ckpt'))
v=Worker();v.load_checkpoint(str(ROOT/'ckpt'));assert audit.state(w)==audit.state(v)
bad=Data();bad.steps=19
try:v.update_actor(bad)
except RuntimeError as e:assert 'steps differ' in str(e)
else:raise AssertionError('Must reject wrong update count')
p=ROOT/'ckpt/recovery-state-rank-0.json';s=json.loads(p.read_text());s['optimizer_sha256']='wrong';p.write_text(json.dumps(s))
try:Worker().load_checkpoint(str(ROOT/'ckpt'))
except RuntimeError as e:assert 'recovery differs' in str(e)
else:raise AssertionError('Must reject changed recovery state')
print(json.dumps({'status':'SYNTHETIC_CPU_AUDIT_PASS','adam_calls':20,'wrong_count_rejected':True,'complete_state_equal':True,'changed_state_rejected':True,'gpu_runtime_accepted':False}))
