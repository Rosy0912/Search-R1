"""Actual same-world8 FSDP/offload/Adam replay, within the owned allocation."""
import json,os,random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from verl.utils.fsdp_utils import offload_fsdp_param_and_grad,load_fsdp_param_and_grad,offload_fsdp_optimizer,load_fsdp_optimizer
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from search_r1.baselines.training_state import worker_snapshot,restore_worker,snapshot_worker_to_file,digest,atomic_save
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl')
assert dist.get_world_size()==8
torch.manual_seed(42);random.seed(42);np.random.seed(42)
model=FSDP(torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.SiLU(),torch.nn.Linear(32,8)).cuda(),
           use_orig_params=False,device_id=rank,mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32))
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
scheduler=get_constant_schedule_with_warmup(optimizer,57)
identity=dict(rank=rank,world_size=8,job=os.environ['SLURM_JOB_ID'],code=os.environ['SEARCH_R1_BASELINE_COMMIT'])
root=Path(os.environ['TREE_NATIVE_STATE_OUTPUT']);root.mkdir(parents=True,exist_ok=True)

def storage_diagnostic():
    rows = {}
    for name, parameter in model.named_parameters():
        shard = getattr(parameter, '_local_shard', None)
        rows[name] = dict(parameter_device=str(parameter.device), parameter_ptr=parameter.data_ptr(),
                          shard_device=None if shard is None else str(shard.device),
                          shard_ptr=None if shard is None else shard.data_ptr(),
                          same_storage=shard is not None and parameter.data_ptr()==shard.data_ptr(),
                          parameter_digest=digest(parameter), shard_digest=None if shard is None else digest(shard))
    return rows

def differences(left, right, path='state'):
    if digest(left)==digest(right): return []
    if isinstance(left, dict) and isinstance(right, dict) and left.keys()==right.keys():
        return [item for key in left for item in differences(left[key],right[key],path+'.'+str(key))]
    row=dict(path=path,left_digest=digest(left),right_digest=digest(right))
    if isinstance(left,torch.Tensor) and isinstance(right,torch.Tensor):
        row.update(left_shape=list(left.shape),right_shape=list(right.shape),left_dtype=str(left.dtype),right_dtype=str(right.dtype))
        if left.shape==right.shape and left.dtype==right.dtype:
            row['different_elements']=int((left!=right).sum())
            if left.is_floating_point(): row['max_absolute_delta']=float((left.double()-right.double()).abs().max()) if left.numel() else 0.
    return [row]

def offload():
    offload_fsdp_param_and_grad(model,offload_grad=True);offload_fsdp_optimizer(optimizer)

def update():
    load_fsdp_param_and_grad(model,rank,load_grad=True);load_fsdp_optimizer(optimizer,rank)
    for mini in range(24):
        optimizer.zero_grad()
        for micro in range(8):
            x=torch.randn(2,16,device='cuda');model(x).float().square().mean().div(8).backward()
        model.clip_grad_norm_(1.);optimizer.step()
    scheduler.step();offload()

for outer in range(5):update()
record=snapshot_worker_to_file(model,optimizer,scheduler,root/f'rank-{rank:02d}.pt',identity,verify=True,cuda=True)
saved=torch.load(root/f'rank-{rank:02d}.pt',map_location='cpu',weights_only=False)
assert scheduler.last_epoch==5 and all(s['step'].item()==120 for s in optimizer.state.values())
step5_storage=storage_diagnostic()
update();uninterrupted_state=worker_snapshot(model,optimizer,scheduler,identity,cuda=True)
uninterrupted=digest(uninterrupted_state)
atomic_save(uninterrupted_state,root/f'rank-{rank:02d}-uninterrupted6.pt')
uninterrupted_storage=storage_diagnostic()
restore_worker(model,optimizer,scheduler,saved,identity)
assert digest(worker_snapshot(model,optimizer,scheduler,identity,cuda=True))==record['logical_state_sha256']
restored_storage=storage_diagnostic()
update();replayed_state=worker_snapshot(model,optimizer,scheduler,identity,cuda=True)
replayed=digest(replayed_state)
atomic_save(replayed_state,root/f'rank-{rank:02d}-replayed6.pt')
atomic_save(dict(rank=rank,step5_storage=step5_storage,uninterrupted_storage=uninterrupted_storage,
    restored_storage=restored_storage,replayed_storage=storage_diagnostic(),
    uninterrupted_digest=uninterrupted,replayed_digest=replayed,
    differences=differences(uninterrupted_state,replayed_state)),root/f'rank-{rank:02d}-diagnostic.json',json_format=True)
assert uninterrupted==replayed,'Actual FSDP continuation differs after same-world fullstate restore'
assert scheduler.last_epoch==6 and all(s['step'].item()==144 for s in optimizer.state.values())
(root/f'rank-{rank:02d}.json').write_text(json.dumps(dict(record,status='PASS',same_world_sixth_outer_replay_equal=True,
    optimizer_steps_at5=120,optimizer_steps_at6=144,physical_gpu_uuid_environment=os.environ.get('CUDA_VISIBLE_DEVICES'))))
dist.barrier();dist.destroy_process_group()
