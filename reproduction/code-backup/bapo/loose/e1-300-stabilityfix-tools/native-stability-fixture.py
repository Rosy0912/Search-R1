"""Actual eight-rank NCCL, production actor guard and worker DataProto return.

No CPU CUDA shims and no source patching. Toy forward isolates the actual PPO
backward; synthetic distinct rank metrics exercise the production worker hook.
Neither case is evidence of language-model learning or a signed receipt.
"""
import contextlib,copy,hashlib,json,math,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
repo=Path(os.environ['SEARCH_R1_REPO']).resolve()
source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
assert source==os.environ['E1_EXPECTED_SOURCE']=='39cdcdd1404213b897bb0de99cc52d72707682f0'
assert repo==Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-20261005/repo')
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
sys.path.insert(0,str(repo))
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from search_r1.baselines.actor_metrics import aggregate_sequence_mean_actor_metrics
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
dist.init_process_group('nccl');device=torch.device('cuda',local)
assert dist.get_world_size()==8
for name in ('verl.workers.actor.dp_actor','verl.workers.fsdp_workers','search_r1.baselines.actor_metrics'):
 assert repo in Path(sys.modules[name].__file__).resolve().parents
try:
    base=torch.nn.Linear(16,1,bias=False).double().to(device)
    with torch.no_grad():base.weight.fill_(.005)
    model=FSDP(base,device_id=device,use_orig_params=False,sharding_strategy=ShardingStrategy.FULL_SHARD,sync_module_states=True)
    opt=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=0.)
    cfg=OmegaConf.create(dict(ppo_mini_batch_size=16,ppo_micro_batch_size=2,ppo_epochs=1,
       state_masking=True,use_kl_loss=False,use_dynamic_bsz=False,loss_reduction='sequence_mean',
       baseline_method='bapo',clip_ratio=.2,entropy_coeff=0.,grad_clip=1.))
    def forward(micro_batch,temperature):
        assert micro_batch['input_ids'].device==device
        value=-3.+model(micro_batch['input_ids']).squeeze(-1)
        return value*0.,value
    instance=SimpleNamespace(config=cfg,actor_module=model,actor_optimizer=opt,
       _forward_micro_batch=forward,ulysses_sequence_parallel_size=1)
    instance._optimizer_step=lambda:DataParallelPPOActor._optimizer_step(instance)
    data=DataProto.from_dict(dict(responses=torch.zeros(16,1,dtype=torch.long),
       input_ids=torch.ones(16,1,16,dtype=torch.float64),attention_mask=torch.ones(16,2,dtype=torch.long),
       position_ids=torch.zeros(16,2,dtype=torch.long),loss_mask=torch.ones(16,1,dtype=torch.float64),
       old_log_probs=torch.full((16,1),-32.92,dtype=torch.float64),
       advantages=-torch.ones(16,1,dtype=torch.float64)),meta_info={'temperature':1.})
    metrics=DataParallelPPOActor.update_policy(instance,data)
    assert metrics['actor/optimizer_steps']==1
    assert math.isclose(sum(metrics['actor/pg_loss']),math.exp(20),rel_tol=1e-12)
    assert max(metrics['actor/grad_norm'])==0.
    assert all(p.grad is not None and torch.equal(p.grad,torch.zeros_like(p.grad)) for p in model.parameters())
    guard=dict(case='actual_FSDP_actor_delta30_negativeA_guard',delta=30.,advantage=-1.,
       pg_loss=sum(metrics['actor/pg_loss']),preclip_norm=max(metrics['actor/grad_norm']),
       optimizer_steps=metrics['actor/optimizer_steps'],clamp_zero_backward=True,
       parameter_gradient_all_zero=True,input_device_before_forward='cpu',forward_device=str(device))
    del model,base,opt;torch.cuda.empty_cache();dist.barrier()

    def record():
        return {'actor/pg_loss':[float(rank+1)]*20,'actor/kl_loss':[float(rank+2)]*20,
          'actor/entropy_loss':[float(rank+3)]*20,'actor/ppo_kl':[float(rank+4)]*20,
          'actor/pg_clipfrac':[rank*.1]*20,'actor/grad_norm':[7.]*20,
          'bapo/grad_norm_min':7.,'bapo/grad_norm_max':7.,'actor/optimizer_steps':20,
          'actor/skipped_empty_minibatches':0,'actor/kl_coef':.001,
          'bapo/loss_tokens':(rank+1)*11,'bapo/old_logprob_nonfinite':0,'bapo/new_logprob_nonfinite':0}
    original=record();saved=copy.deepcopy(original)
    class Sharding:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def preprocess_data(self,data):return data
        def postprocess_data(self,data):return data
    class Scheduler:
        last_epoch=4
        def get_last_lr(self):return [1e-6*self.last_epoch/114]
        def step(self):self.last_epoch+=1
    worker=SimpleNamespace(_is_actor=True,_is_offload_param=False,_is_offload_optimizer=False,
       config=OmegaConf.create({'actor':{'baseline_method':'bapo','loss_reduction':'sequence_mean','ppo_epochs':1},
                               'rollout':{'tensor_model_parallel_size':1}}),
       ulysses_sequence_parallel_size=1,ulysses_sharding_manager=Sharding(),world_size=8,rank=rank,
       actor=SimpleNamespace(update_policy=lambda data:copy.deepcopy(original)),actor_lr_scheduler=Scheduler(),
       flops_counter=SimpleNamespace(estimate_flops=lambda tokens,seconds:(1.,10.)))
    payload=DataProto.from_dict({'responses':torch.zeros(1,1)},meta_info={'global_token_num':[1]*8})
    returned=ActorRolloutRefWorker.update_actor(worker,payload)
    assert original==saved
    outputs=[None]*8;dist.all_gather_object(outputs,returned)
    concatenated=DataProto.concat(outputs);global_metrics=concatenated.meta_info['metrics']
    assert all(x.meta_info==outputs[0].meta_info for x in outputs)
    assert global_metrics['actor/pg_loss']==[4.5]*20
    assert global_metrics['actor/kl_loss']==[5.5]*20
    assert global_metrics['actor/entropy_loss']==[6.5]*20
    assert global_metrics['actor/ppo_kl']==[7.5]*20
    assert math.isclose(global_metrics['actor/pg_clipfrac'][0],.35,abs_tol=1e-15)
    assert global_metrics['bapo/loss_tokens']==396
    assert global_metrics['actor/grad_norm']==[7.]*20
    assert global_metrics['actor/optimizer_steps']==20 and global_metrics['actor/scheduler_outer_steps']==5
    assert math.isclose(global_metrics['actor/lr_before_outer'],4e-6/114,rel_tol=1e-15)
    assert math.isclose(global_metrics['actor/lr'],5e-6/114,rel_tol=1e-15)
    assert worker.actor_lr_scheduler.last_epoch==5
    # Unknown replicated-clock disagreement is rejected on every actual rank.
    broken=dict(global_metrics,**{'actor/scheduler_outer_steps':5+rank})
    rejected=False
    try:aggregate_sequence_mean_actor_metrics(broken)
    except ValueError as exc:
        rejected='Replicated' in str(exc)
    assert rejected
    hook=dict(case='actual_worker_NCCL8_DataProto_return_concat',global_metrics=global_metrics,
       local_original_metrics=saved,local_original_unchanged=True,all_returns_identical=True,
       scheduler_calls=1,replicated_clock_mismatch_rejected=True)
    files=('verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py',
       'search_r1/baselines/objectives.py','search_r1/baselines/actor_metrics.py')
    out=dict(status='PASS_NATIVE_STABILITY_NCCL8',source_commit=source,rank=rank,world=8,
       device=str(device),backend=dist.get_backend(),sys_prefix=sys.prefix,repo=str(repo),torch=torch.__version__,
       fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       source_files_sha256={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},records=[guard,hook])
    output=Path(os.environ['E1_STABILITY_FIXTURE_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
    (output/f'rank-{rank}.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print('NATIVE_STABILITY_NCCL8_JSON='+json.dumps(out,sort_keys=True,allow_nan=False),flush=True)
    dist.barrier()
finally:dist.destroy_process_group()
