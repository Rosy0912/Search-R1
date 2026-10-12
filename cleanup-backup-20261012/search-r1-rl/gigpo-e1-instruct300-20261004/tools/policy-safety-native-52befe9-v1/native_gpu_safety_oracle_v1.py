"""Native8: actual HF decoder/FSDP/manual offload and actor ten-step replay."""
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
actor=DataParallelPPOActor(OmegaConf.create(dict(ppo_mini_batch_size=32,ppo_micro_batch_size=2,ppo_epochs=1,
    state_masking=True,use_kl_loss=True,kl_loss_type='low_var_kl',kl_loss_coef=.001,entropy_coeff=0.,
    use_dynamic_bsz=False,ulysses_sequence_parallel_size=1,use_remove_padding=True,baseline_method='gigpo',
    clip_ratio=.2,grad_clip=1.,complete_internal_state=True,joint_loss_scaling=True,policy_safety=dict(enabled=True,mode='native_fixture',dual_clip=3.,kl_log_bound=30.,monitor_first_global_rows=256,target_old_policy_kl=100.,negative_tail_limit=1.))),model,opt)

actor.rollout_sharding_manager = manager
os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']=str(root/'audit')
os.environ['SEARCH_R1_BASELINE_RUN_ID']='safety-native-tiny-oracle'
os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY']='SYNTHETIC_NATIVE_FIXTURE_NOT_COMMON_TRAINING_DATA'
os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256']=__import__('hashlib').sha256(OmegaConf.to_yaml(actor.config).encode()).hexdigest()
# Observe the actual imported production denominator in every native actor mini.
import search_r1.baselines.objectives as production_objectives
original_row_denominator = production_objectives.row_denominator
observed_denominators = []
def observed_row_denominator(rows, device):
    value = original_row_denominator(rows, device)
    assert rows == 32 and float(value) == 32
    observed_denominators.append(float(value))
    return value
production_objectives.row_denominator = observed_row_denominator

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

def update():
    forward_count[0]=0
    observed_denominators.clear()
    load_fsdp_param_and_grad(model,device_id,load_grad=True);load_fsdp_optimizer(opt,device_id)
    with manager:
        assert engine.copied is not None
        torch.rand(17,device='cuda')
    assert engine.copied is None
    inputs=torch.randint(0,256,(320,16),device='cuda');responses=inputs[:,-8:]
    attention=torch.ones_like(inputs);position=torch.arange(16,device='cuda').expand(320,16)
    mask=torch.ones(320,8,device='cuda');mask[:,1]=0;mask[::7,4:]=0
    ids=torch.where(mask.bool(),torch.arange(8,device='cuda')//2,-1).expand(320,8).clone()
    with torch.no_grad():
        logp=torch.cat([actor._forward_micro_batch(dict(input_ids=inputs[i:i+4],responses=responses[i:i+4],attention_mask=attention[i:i+4],position_ids=position[i:i+4]),1.)[1] for i in range(0,320,4)])
    adv=torch.randn(320,4,device='cuda').repeat_interleave(2,dim=1)
    batch=DataProto.from_dict(dict(input_ids=inputs,responses=responses,attention_mask=attention,position_ids=position,
        loss_mask=mask,step_ids=ids,old_log_probs=logp,ref_log_prob=logp-.01,advantages=adv,
        policy_weights=mask/mask.sum(-1,keepdim=True)),meta_info={'temperature':1.,'baseline_outer_step':sch.last_epoch+1})
    event('logprob_complete')
    batch=batch.to('cuda')  # Match ActorRolloutRefWorker.update_actor's actual TensorDict device boundary.
    event('actor_update_start',rows=len(batch),tensor_device=str(batch.batch.device))
    result=actor.update_policy(batch)
    assert observed_denominators == [32.] * 10
    assert forward_count[0]==160
    result['actual_micro_batches']=forward_count[0]
    assert result['actor/optimizer_steps']==10 and result['actor/skipped_empty_minibatches']==0
    assert all(np.isfinite(x) and x>0 for x in result['actor/grad_norm'])
    sch.step();offload();event('outer_complete',scheduler=sch.last_epoch,optimizer_steps=result['actor/optimizer_steps'],micro_count=forward_count[0]);return result

records=[]
for outer in range(5):
    result=update();records.append(dict(outer=outer+1,optimizer_steps=result['actor/optimizer_steps'],micro_batches=result['actual_micro_batches'],scheduler=sch.last_epoch,grad_norms=result['actor/grad_norm']))
assert all(s['step'].item()==50 for s in opt.state.values()) and sch.last_epoch==5
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
assert all(s['step'].item()==60 for s in opt.state.values()) and sch.last_epoch==6
# Real NCCL row reduction with entire rank0 having zero effective tokens.
gen=torch.Generator().manual_seed(31)
mask=(torch.rand(256,8,generator=gen)>.3).float();mask[:32]=0
ids=torch.where(mask.bool(),torch.arange(8)//2,-1).expand(256,8)
weights=mask/mask.sum(-1,keepdim=True).clamp_min(1)
adv=torch.randn(256,4,generator=gen).repeat_interleave(2,dim=1)
features=torch.randn(256,8,generator=gen)*.02
old=torch.randn(256,8,generator=gen)*.1
all_tensors=[x.cuda() for x in (old,adv,mask,weights,ids,features)]
def loss(theta,lo,hi,denom):
    o,a,m,w,s,f=[x[lo:hi] for x in all_tensors];current=o+theta*f
    pg=weighted_policy_loss(o,current,a,m,w,.2,denom,s,dual_clip=3.)[0]
    return pg+.001*(kl_penalty(current,o-.07,'low_var_kl')*w).sum()/denom
full=torch.tensor(.9,device='cuda',requires_grad=True);loss(full,0,256,256).backward()
part=torch.tensor(.9,device='cuda',requires_grad=True)
scalar=torch.zeros((),device='cuda',dtype=torch.float64)
for row in range(rank*32,(rank+1)*32,2):
    value=loss(part,row,row+2,32);scalar+=value.detach();value.backward()
dist.all_reduce(scalar);scalar.div_(8)
torch.testing.assert_close(scalar,loss(full,0,256,256).detach(),rtol=1e-6,atol=1e-10)
dist.all_reduce(part.grad);part.grad.div_(8)
torch.testing.assert_close(part.grad,full.grad,rtol=1e-6,atol=1e-10)
# Production safety schema and actual synchronized refusal before Adam.
from safety_audit_v1 import validate_update, validate_guard
updates=[json.loads(line) for line in (root/'audit'/f'{rank}.jsonl').read_text().splitlines()]
assert len(updates)==7
for row in updates:validate_update(row)
guards=[json.loads(line) for line in (root/'audit/policy-safety'/f'rank-{rank}.jsonl').read_text().splitlines()]
assert len(guards)==140
for row in guards:validate_guard(row,'native_fixture',os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'])
assert all(row['allowed'] for row in guards)
assert all(row['statistics']['valid_token_count']>0 for row in guards)
assert all(row['fixed_monitor_hashes']==guards[20*(i//20)]['fixed_monitor_hashes'] for i,row in enumerate(guards))
load_fsdp_param_and_grad(model,device_id,load_grad=True);load_fsdp_optimizer(opt,device_id)
inputs=torch.randint(0,256,(320,16),device='cuda');responses=inputs[:,-8:]
attention=torch.ones_like(inputs);position=torch.arange(16,device='cuda').expand(320,16)
with torch.no_grad():
    old=torch.cat([actor._forward_micro_batch(dict(input_ids=inputs[i:i+2],responses=responses[i:i+2],attention_mask=attention[i:i+2],position_ids=position[i:i+2]),1.)[1] for i in range(0,320,2)])
mask=torch.ones_like(old);ids=torch.zeros_like(old,dtype=torch.long)
bad_batch=DataProto.from_dict(dict(input_ids=inputs,responses=responses,attention_mask=attention,position_ids=position,
    loss_mask=mask,step_ids=ids,old_log_probs=old.clone(),ref_log_prob=old.clone(),advantages=-torch.ones_like(old),
    policy_weights=mask/8),meta_info={'temperature':1.,'baseline_outer_step':99}).to('cuda')
if rank==5:bad_batch.batch['old_log_probs'][0,0]=float('nan')
before_bad=recovery.checkpoint_pause_identity(model,opt,sch,rank,manager,True)
try:
    actor.update_policy(bad_batch)
    raise AssertionError('Invalid single-rank old probability was admitted')
except RuntimeError as error:
    assert 'POLICY_SAFETY_REJECTED' in str(error)
after_bad=recovery.checkpoint_pause_identity(model,opt,sch,rank,manager,True)
assert before_bad==after_bad and all(s['step'].item()==60 for s in opt.state.values())
# All ranks reached this collective after the same rejection; no hanging forward/extra Adam.
refusals=torch.tensor(1,device='cuda');dist.all_reduce(refusals);assert int(refusals)==8
(root/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,world=8,torch=torch.__version__,
    recovery_file=recovery.__file__,actor_file=__import__('inspect').getfile(DataParallelPPOActor),records=records,
    state_receipt=receipt,decoder_autowrap=True,hf_model_output=True,manual_offload=True,
    full_hf_save_reload=True,real_sharding_manager_context=True,copy_only_engine=True,remove_padding=True,flash_attention2=True,next_outer_replay_equal=True,sequence_pg_kl_dp8_micro2_equal=True,empty_rank_zero=True,dual_clip=3.,safety_scope='native_fixture_not_production_calibration',single_invalid_rank_synchronous_refusal=True,no_extra_adam_after_refusal=True,raw_joint_exp_not_computed=True,monitor_guards=140,
    row_oracle=dict(observed_scalar=float(scalar),expected_scalar=float(loss(full,0,256,256).detach()),observed_gradient=float(part.grad),expected_gradient=float(full.grad)),production_denominator_per_mini=32,global_row_denominator=256,production_manager_probe=True,optimizer_steps_at5=50,optimizer_steps_at6=60,scheduler_steps_at5=5,scheduler_steps_at6=6),indent=2))
dist.barrier();dist.destroy_process_group()
