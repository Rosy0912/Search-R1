"""Native8: actual HF decoder/FSDP/manual offload and actor twenty-step replay."""
import os
# Match Ray's physical per-worker CUDA visibility before Torch/vLLM initializes contexts.
physical_rank=int(os.environ['LOCAL_RANK'])
allocated_devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(allocated_devices)==8
os.environ['CUDA_VISIBLE_DEVICES']=allocated_devices[physical_rank]
os.environ['LOCAL_RANK']='0'
import datetime,json,random
from pathlib import Path
from functools import partial
from torch.distributed.device_mesh import init_device_mesh
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import Qwen2Config,Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.fsdp_utils import offload_fsdp_param_and_grad,load_fsdp_param_and_grad,offload_fsdp_optimizer,load_fsdp_optimizer
from search_r1.baselines import recovery
from search_r1.baselines.instruct300 import equal_state
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
from search_r1.baselines.objectives import weighted_policy_loss
from verl.trainer.ppo.core_algos import kl_penalty
rank=int(os.environ['RANK']);device_id=0;torch.cuda.set_device(device_id)
assert torch.cuda.device_count()==1
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=600))
assert dist.get_world_size()==8 and torch.__version__.startswith('2.4')
root=Path(os.environ['NUMERIC_FIXTURE_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
def event(name,**details):
    with (root/f'progress-rank-{rank}.jsonl').open('a') as out:
        out.write(json.dumps(dict(event=name,rank=rank,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.current_device(),**details))+'\n')
event('initialized')
torch.manual_seed(42);random.seed(42);np.random.seed(42)
# Genuine CausalLM ModelOutput, nested decoder wrapping, FP32 reduction and runtime manual offload.
qcfg=Qwen2Config(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.,tie_word_embeddings=False)
qcfg._attn_implementation='flash_attention_2'
hf=Qwen2ForCausalLM(qcfg).cuda()
hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
model=FSDP(hf,use_orig_params=False,device_id=device_id,
    auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Qwen2DecoderLayer}),
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32))
opt=torch.optim.AdamW(model.parameters(),lr=1e-6)
sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(1.,step/57))
class CopyEngine:
    def sync_model_weights(self, params, load_format):
        assert load_format=='hf' and params
        self.copied={name:value.detach().cpu().clone() for name,value in params.items()}
        assert all(torch.isfinite(value).all() for value in self.copied.values())
    def offload_model_weights(self):
        assert self.copied
        self.copied=None
engine=CopyEngine()
mesh=init_device_mesh('cuda',(8,1),mesh_dim_names=('dp','tp'))
manager=FSDPVLLMShardingManager(model,engine,qcfg,full_params=True,device_mesh=mesh)
actor=DataParallelPPOActor(OmegaConf.create(dict(ppo_mini_batch_size=16,ppo_micro_batch_size=2,ppo_epochs=1,
    state_masking=True,use_kl_loss=True,kl_loss_type='low_var_kl',kl_loss_coef=.001,entropy_coeff=0.,
    use_dynamic_bsz=False,ulysses_sequence_parallel_size=1,use_remove_padding=True,baseline_method='gigpo',
    clip_ratio=.2,grad_clip=1.,complete_internal_state=True,joint_loss_scaling=True)),model,opt)
actor.rollout_sharding_manager=manager

def offload():
    offload_fsdp_param_and_grad(model,offload_grad=True);offload_fsdp_optimizer(opt)

forward_count=[0]
original_forward=actor._forward_micro_batch
def counted_forward(micro_batch,temperature):
    if torch.is_grad_enabled():
        forward_count[0]+=1
        if forward_count[0]%8==0:event('training_forward',micro_count=forward_count[0])
    return original_forward(micro_batch=micro_batch,temperature=temperature)
actor._forward_micro_batch=counted_forward

def update(high_ratio=False):
    forward_count[0]=0
    load_fsdp_param_and_grad(model,device_id,load_grad=True);load_fsdp_optimizer(opt,device_id)
    with manager:
        assert engine.copied is not None
        torch.rand(17,device='cuda')
    assert engine.copied is None
    inputs=torch.randint(0,256,(320,16),device='cuda');responses=inputs[:,-8:]
    attention=torch.ones_like(inputs);position=torch.arange(16,device='cuda').expand(320,16)
    mask=torch.ones(320,8,device='cuda');mask[:,1]=0;mask[::7,4:]=0
    if high_ratio and rank==0:mask.zero_()
    ids=torch.where(mask.bool(),torch.arange(8,device='cuda')//2,-1).expand(320,8).clone()
    with torch.no_grad():
        logp=torch.cat([actor._forward_micro_batch(dict(input_ids=inputs[i:i+4],responses=responses[i:i+4],attention_mask=attention[i:i+4],position_ids=position[i:i+4]),1.)[1] for i in range(0,320,4)])
    adv=torch.randn(320,4,device='cuda').repeat_interleave(2,dim=1)
    old_logp=logp.clone()
    if high_ratio:
        adv.fill_(-1.)
        for row in range(320):
            for action in torch.unique(ids[row][mask[row].bool()]):
                selected=(ids[row]==action)&mask[row].bool()
                old_logp[row,selected]-=100./int(selected.sum())
    from search_r1.baselines.loss_scaling import TransparentProbe
    before=recovery.checkpoint_pause_identity(model,opt,sch,rank,manager,True)
    with TransparentProbe(model,manager) as probe:
        with torch.no_grad():
            actor._forward_micro_batch(dict(input_ids=inputs[:2],responses=responses[:2],attention_mask=attention[:2],position_ids=position[:2]),1.)
    assert not probe.invalid
    assert before==recovery.checkpoint_pause_identity(model,opt,sch,rank,manager,True)
    batch=DataProto.from_dict(dict(input_ids=inputs,responses=responses,attention_mask=attention,position_ids=position,
        loss_mask=mask,step_ids=ids,old_log_probs=old_logp,ref_log_prob=logp-.01,advantages=adv,
        policy_weights=mask/mask.sum(-1,keepdim=True).clamp_min(1)),meta_info={'temperature':1.})
    event('logprob_complete')
    batch=batch.to('cuda')  # Match ActorRolloutRefWorker.update_actor's actual TensorDict device boundary.
    event('actor_update_start',rows=len(batch),tensor_device=str(batch.batch.device))
    result=actor.update_policy(batch)
    assert forward_count[0]==160
    assert result['actor/probe_micro_batches']==[8]*20
    if high_ratio:assert all(0<s<1 for s in result['actor/joint_loss_scale'])
    result['actual_micro_batches']=forward_count[0]
    assert result['actor/optimizer_steps']==20 and result['actor/skipped_empty_minibatches']==0
    assert all(np.isfinite(x) and x>0 for x in result['actor/grad_norm'])
    sch.step();offload();event('outer_complete',scheduler=sch.last_epoch,optimizer_steps=result['actor/optimizer_steps'],micro_count=forward_count[0]);return result

records=[]
for outer in range(5):
    result=update();records.append(dict(outer=outer+1,optimizer_steps=result['actor/optimizer_steps'],micro_batches=result['actual_micro_batches'],scheduler=sch.last_epoch,grad_norms=result['actor/grad_norm']))
assert all(s['step'].item()==100 for s in opt.state.values()) and sch.last_epoch==5
load_fsdp_param_and_grad(model,device_id,load_grad=True)
full_state=model.state_dict()
if rank==0:
    hf.save_pretrained(root/'hf-step5',state_dict={k:v.detach().cpu() for k,v in full_state.items()},safe_serialization=True)
    reloaded=Qwen2ForCausalLM.from_pretrained(root/'hf-step5',local_files_only=True)
    for name,value in reloaded.state_dict().items():
        torch.testing.assert_close(value,full_state[name].cpu(),rtol=0,atol=0)
    del reloaded
del full_state
dist.barrier();offload()
state=recovery.worker_training_state(model,opt,sch,rank,8,5,manager,complete_internal=True)
assert state['local_shards'] and state['buffers']
receipt=recovery.atomic_torch_save(state,root/f'rank-{rank}.pt')
update();expected=recovery.worker_training_state(model,opt,sch,rank,8,6,manager,complete_internal=True)
loaded=torch.load(receipt['path'],map_location='cpu',weights_only=False)
assert recovery.load_worker_training_state(loaded,model,opt,sch,rank,8,manager)==5
assert equal_state(state,recovery.worker_training_state(model,opt,sch,rank,8,5,manager,complete_internal=True))
update();replayed=recovery.worker_training_state(model,opt,sch,rank,8,6,manager,complete_internal=True)
assert equal_state(expected,replayed),'Saved HF/FSDP internal state failed next-outer replay'
assert all(s['step'].item()==120 for s in opt.state.values()) and sch.last_epoch==6
high_result=update(high_ratio=True)
assert all(s['step'].item()==140 for s in opt.state.values()) and sch.last_epoch==7
# Real NCCL row reduction with entire rank0 having zero effective tokens.
gen=torch.Generator().manual_seed(31)
mask=(torch.rand(128,8,generator=gen)>.3).float();mask[:16]=0
ids=torch.where(mask.bool(),torch.arange(8)//2,-1).expand(128,8)
weights=mask/mask.sum(-1,keepdim=True).clamp_min(1)
adv=torch.randn(128,4,generator=gen).repeat_interleave(2,dim=1)
features=torch.randn(128,8,generator=gen)*.02
old=torch.randn(128,8,generator=gen)*.1
all_tensors=[x.cuda() for x in (old,adv,mask,weights,ids,features)]
def loss(theta,lo,hi,denom):
    o,a,m,w,s,f=[x[lo:hi] for x in all_tensors];current=o+theta*f
    pg=weighted_policy_loss(o,current,a,m,w,.2,denom,s)[0]
    return pg+.001*(kl_penalty(current,o-.07,'low_var_kl')*w).sum()/denom
full=torch.tensor(.9,device='cuda',requires_grad=True);loss(full,0,128,128).backward()
part=torch.tensor(.9,device='cuda',requires_grad=True)
scalar=torch.zeros((),device='cuda',dtype=torch.float64)
for row in range(rank*16,(rank+1)*16,2):
    value=loss(part,row,row+2,16);scalar+=value.detach();value.backward()
dist.all_reduce(scalar);scalar.div_(8)
torch.testing.assert_close(scalar,loss(full,0,128,128).detach(),rtol=1e-6,atol=1e-10)
dist.all_reduce(part.grad);part.grad.div_(8)
torch.testing.assert_close(part.grad,full.grad,rtol=1e-6,atol=1e-10)
# Full-vector FP64 oracle for the actual actor's scale + FSDP DP + fused clip
# and the exact vector delivered to Adam. This table is a mathematical fixture;
# the genuine HF/FlashAttention/manual-offload execution is tested separately above.
class LogProbabilityTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values=torch.nn.Parameter(torch.full((128,8),.25,device='cuda'))
    def forward(self, rows):return self.values[rows]
scaled_oracles=[]
for joint in (0.,100.):
    table=FSDP(LogProbabilityTable(),device_id=0,use_orig_params=False)
    table_opt=torch.optim.AdamW(table.parameters(),lr=1e-6)
    cfg=OmegaConf.create(OmegaConf.to_container(actor.config,resolve=True))
    table_actor=DataParallelPPOActor(cfg,table,table_opt)
    table_actor._forward_micro_batch=lambda micro_batch,temperature:(torch.zeros_like(micro_batch['responses'],dtype=torch.float32),table(micro_batch['input_ids'][:,0]))
    m=torch.ones(128,8,device='cuda');m[:16]=0;m[:,1]=0
    step=torch.where(m.bool(),torch.arange(8,device='cuda')//2,-1).expand(128,8).clone()
    w=m/m.sum(-1,keepdim=True).clamp_min(1)
    current=torch.full((128,8),.25,device='cuda')
    old=current.clone()
    for row in range(128):
        for action in torch.unique(step[row][m[row].bool()]):
            selected=(step[row]==action)&m[row].bool()
            old[row,selected]-=joint/int(selected.sum())
    a=torch.full_like(current,-1.)
    reference=current-.07
    oracle=current.double().detach().requires_grad_()
    # Match native Float32 action sums while preserving their derivative.
    deltas=[]
    for row in range(128):
        d=oracle[row]*0
        for action in torch.unique(step[row][m[row].bool()]):
            selected=(step[row]==action)&m[row].bool()
            value=(oracle[row,selected]-old[row,selected].double()).sum()
            rounded=(current[row,selected]-old[row,selected]).sum().double()
            value=value+(rounded-value.detach())
            d=d+selected*value
        deltas.append(d)
    ratio=torch.stack(deltas).exp()
    expected_loss=(torch.maximum(-a.double()*ratio,-a.double()*ratio.clamp(.8,1.2))*w.double()).sum()/128
    expected_loss=expected_loss+.001*(kl_penalty(oracle,reference.double(),'low_var_kl')*w.double()).sum()/128
    expected_loss.backward()
    original_norm=torch.linalg.vector_norm(oracle.grad)
    clipped=(oracle.grad*(1/(original_norm+1e-6)).clamp(max=1)).float().flatten()
    lo,hi=rank*16,(rank+1)*16
    inp=torch.zeros(16,16,device='cuda',dtype=torch.long);inp[:,0]=torch.arange(lo,hi,device='cuda')
    data=DataProto.from_dict(dict(input_ids=inp,responses=inp[:,-8:],attention_mask=torch.ones_like(inp),position_ids=torch.arange(16,device='cuda').expand_as(inp),loss_mask=m[lo:hi],step_ids=step[lo:hi],policy_weights=w[lo:hi],old_log_probs=old[lo:hi],ref_log_prob=reference[lo:hi],advantages=a[lo:hi]),meta_info={'temperature':1.}).to('cuda')
    captured=[]
    real_step=table_opt.step
    def checked_adam_step(*args,**kwargs):
        captured.append(torch.cat([parameter.grad.detach().flatten().clone() for parameter in table.parameters()]))
        return real_step(*args,**kwargs)
    table_opt.step=checked_adam_step
    with torch.no_grad():table(inp[:2,0])  # Initialize the same lazy FSDP boundary before the probe.
    actual=table_actor.update_policy(data)
    assert len(captured)==1 and actual['actor/optimizer_steps']==1
    torch.testing.assert_close(captured[0],clipped[rank*128:(rank+1)*128],rtol=2e-5,atol=1e-7)
    torch.testing.assert_close(torch.tensor(actual['actor/grad_norm'][0],device='cuda',dtype=torch.float64),original_norm,rtol=2e-5,atol=1e-8)
    actual_scalar=torch.tensor(actual['actor/pg_loss'][0]+.001*actual['actor/kl_loss'][0],device='cuda',dtype=torch.float64)
    torch.testing.assert_close(actual_scalar,expected_loss.detach(),rtol=2e-5,atol=1e-8)
    scaled_oracles.append(dict(joint=joint,full_global_rows=128,dp=8,micro=2,scale=actual['actor/joint_loss_scale'][0],epsilon=1e-6,expected_original_norm=float(original_norm),actual_original_norm=actual['actor/grad_norm'][0],adam_input=captured[0].cpu().tolist(),expected_adam_input=clipped[rank*128:(rank+1)*128].cpu().tolist()))
# Real NO_SHARD: a rank-local unscaled norm overflow must reject every rank.
from torch.distributed.fsdp import ShardingStrategy
from search_r1.baselines.gradient_norm import clip_grad_norm64
from search_r1.baselines.loss_scaling import active_joint_bound,coordinated_scale
replicated=FSDP(torch.nn.Linear(2,1,bias=False).cuda(),device_id=0,
                sharding_strategy=ShardingStrategy.NO_SHARD,use_orig_params=False)
replicated(torch.ones(1,2,device='cuda')).sum().backward()
for parameter in replicated.parameters():parameter.grad.fill_(1e30 if rank==0 else 0.)
rejected=False
try:clip_grad_norm64(replicated,1.,1e-300)
except ValueError as error:
    rejected='unscaled actor gradient norm' in str(error)
assert rejected
for parameter in replicated.parameters():parameter.grad.fill_(1e-41)
clip_grad_norm64(replicated,1.,1e-40)
for parameter in replicated.parameters():
    torch.testing.assert_close(parameter.grad,torch.full_like(parameter.grad,.1),rtol=.001,atol=1e-5)
x=torch.zeros(1,1,device='cuda')
steps=torch.full_like(x,float('nan')) if rank==0 else torch.zeros_like(x,dtype=torch.long)
bound,bad=active_joint_bound(x,x,torch.ones_like(x),torch.ones_like(x),torch.ones_like(x),steps,.2)
rejected=False
try:coordinated_scale(bound,bad)
except ValueError as error:rejected='all ranks' in str(error)
assert rejected
dist.barrier()
(root/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,world=8,torch=torch.__version__,
    recovery_file=recovery.__file__,actor_file=__import__('inspect').getfile(DataParallelPPOActor),records=records,
    state_receipt=receipt,decoder_autowrap=True,hf_model_output=True,manual_offload=True,
    full_hf_save_reload=True,real_sharding_manager_context=True,copy_only_engine=True,remove_padding=True,flash_attention2=True,next_outer_replay_equal=True,sequence_pg_kl_dp8_micro2_equal=True,empty_rank_zero=True,
    scaled_fullvector_fp64_adam_oracles=scaled_oracles,no_shard_unscaled_overflow_allrank_refusal=True,no_shard_large_coefficient_finite=True,invalid_rank_probe_coordinated_refusal=True,scaled_probe_state_equal=True,scaled_high_joint100_actual_actor_finite=True,scaled_high_ratio_optimizer_steps=high_result['actor/optimizer_steps'],optimizer_steps_at5=100,optimizer_steps_at6=120,scheduler_steps_at5=5,scheduler_steps_at6=6),indent=2))
dist.barrier();dist.destroy_process_group()
