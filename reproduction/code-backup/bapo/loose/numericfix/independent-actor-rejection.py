import pathlib,sys,os,json,types,hashlib
repo=pathlib.Path('work/bapo-numerics').resolve();sys.path.insert(0,str(repo));os.environ.pop('SEARCH_R1_BASELINE_AUDIT_DIR',None)
import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.gradient_norm import clip_grad_norm64
records=[]
for bad in (float('nan'),float('inf'),float('-inf')):
 m=torch.nn.Linear(2,1,bias=False);m.weight.grad=torch.tensor([[bad,1.]])
 before=m.weight.detach().clone();grad=m.weight.grad.clone();calls=[]
 fake=types.SimpleNamespace(actor_module=m,config=types.SimpleNamespace(grad_clip=1.),actor_optimizer=types.SimpleNamespace(step=lambda:calls.append('step')))
 refused=False
 try:DataParallelPPOActor._optimizer_step(fake)
 except ValueError as e:refused='Nonfinite' in str(e)
 assert refused and not calls and torch.equal(m.weight,before)
 torch.testing.assert_close(m.weight.grad,grad,equal_nan=True,rtol=0,atol=0)
 records.append({'bad':repr(bad),'actual_optimizer_step_refused':True,'optimizer_calls':0,'gradient_not_mutated':True,'weight_not_mutated':True})
print(json.dumps({'scope':'LOCAL_CPU_ACTUAL_ACTOR_REJECTION_ONLY','records':records,'source_file_sha256':hashlib.sha256((repo/'verl/workers/actor/dp_actor.py').read_bytes()).hexdigest()},indent=2))
