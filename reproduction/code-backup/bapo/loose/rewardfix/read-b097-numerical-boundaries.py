import pathlib,sys,json,hashlib,os,subprocess,types
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097');sys.path.insert(0,str(root/'repo'));os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from verl.trainer.ppo.core_algos import kl_penalty
from verl.workers.actor.dp_actor import DataParallelPPOActor
from hydra import compose,initialize_config_dir
with initialize_config_dir(config_dir=str(root/'repo/verl/trainer/config'),version_base=None):
 cfg=compose(config_name='ppo_baselines',overrides=['baselines.method=bapo'])
records=[]
for delta in [1.,3.,80.,89.,99.,-99.]:
 logprob=torch.tensor([-1.-delta] if delta>=0 else [-1.],dtype=torch.float32,requires_grad=True)
 ref=torch.tensor([-1.] if delta>=0 else [-1.+delta],dtype=torch.float32)
 out=kl_penalty(logprob,ref,'low_var_kl');out.sum().backward()
 records.append(dict(ref_minus_actor=float((ref-logprob).detach()[0]),logprob=float(logprob.detach()[0]),ref_logprob=float(ref[0]),forward=float(out.detach()[0]),forward_finite=bool(torch.isfinite(out).all()),grad=repr(float(logprob.grad[0])),gradient_finite=bool(torch.isfinite(logprob.grad).all())))
model=torch.nn.Linear(2,1,bias=False,dtype=torch.float32);model.weight.grad=torch.full_like(model.weight,1e20)
original=model.weight.grad.clone();fake=types.SimpleNamespace(actor_module=model,config=types.SimpleNamespace(grad_clip=1.0),actor_optimizer=torch.optim.SGD(model.parameters(),lr=1.0))
try:
 value=DataParallelPPOActor._optimizer_step(fake);optimizer_result={'result':repr(value)}
except Exception as e:optimizer_result={'exception':type(e).__name__,'message':str(e)}
optimizer_result.update(elements_finite_before=bool(torch.isfinite(original).all()),fp64_reference_norm=float(torch.linalg.vector_norm(original.double())),gradient_after=model.weight.grad.tolist())
files={p:hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest() for p in ['verl/trainer/ppo/core_algos.py','verl/workers/actor/dp_actor.py','verl/trainer/config/ppo_baselines.yaml']}
print(json.dumps(dict(source_commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),device='cpu',torch_version=torch.__version__,source_files=files,config={'use_kl_loss':cfg.actor_rollout_ref.actor.use_kl_loss,'kl_loss_type':cfg.actor_rollout_ref.actor.kl_loss_type,'kl_loss_coef':cfg.actor_rollout_ref.actor.kl_loss_coef,'actor_fsdp_param_offload':cfg.actor_rollout_ref.actor.fsdp_config.param_offload},kl_actual_function_module=kl_penalty.__module__,kl_boundaries=records,actual_optimizer_step=optimizer_result),indent=2))
