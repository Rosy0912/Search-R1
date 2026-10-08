"""Real8rank checkpoint and next-update equivalence; no3B policy training."""
import datetime,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'repo'))
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
from search_r1.baselines import recovery
rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
assert dist.get_world_size()==8 and torch.__version__.startswith('2.4.0')
records=[]
for orig in (False,True):
 for offload in (False,True):
  torch.manual_seed(42)
  def build():
   module=FSDP(torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.Tanh(),torch.nn.Linear(16,8)),device_id=device,use_orig_params=orig,cpu_offload=CPUOffload(offload_params=offload),mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32))
   optimizer=torch.optim.AdamW(module.parameters(),lr=1e-6)
   scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,step/142))
   recovery.rebuild_scheduler_at_logical_step(scheduler,200)
   return module,optimizer,scheduler
  model,optimizer,scheduler=build()
  x=torch.arange(32,device=device,dtype=torch.float32).reshape(2,16)/32+rank/100
  def update(module,opt,sched):
   opt.zero_grad(set_to_none=True)
   module(x).square().sum().backward()
   opt.step();sched.step()
  for _ in range(5):update(model,optimizer,scheduler)
  assert scheduler.last_epoch==205
  saved=recovery.worker_training_state(model,optimizer,scheduler,rank,8,205)
  path=Path(os.environ['NUMERIC_FIXTURE_OUTPUT'])/f'roundtrip-{orig}-{offload}-rank{rank}.pt'
  receipt=recovery.atomic_torch_save(saved,path)
  # Reference: uninterrupted next actual optimizer update.
  update(model,optimizer,scheduler)
  expected={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
  expected_opt=recovery.cpu_snapshot(optimizer.state_dict())
  restored,newopt,newsched=build()
  payload=torch.load(path,map_location='cpu',weights_only=False)
  assert recovery.file_sha256(path)==receipt['sha256']
  assert recovery.load_worker_training_state(payload,restored,newopt,newsched,rank,8)==205
  update(restored,newopt,newsched)
  assert newsched.last_epoch==206
  for name,p in restored.named_parameters():torch.testing.assert_close(p.detach().cpu(),expected[name],rtol=0,atol=0)
  for key,slot in newopt.state_dict()['state'].items():
   for field,value in slot.items():
    if isinstance(value,torch.Tensor):torch.testing.assert_close(value.cpu(),expected_opt['state'][key][field],rtol=0,atol=0)
  identity=recovery.checkpoint_pause_identity(restored,newopt,newsched,rank)
  assert identity==recovery.checkpoint_pause_identity(restored,newopt,newsched,rank)
  try:
   bad=dict(payload);bad['layout']=dict(payload['layout'],world_size=4)
   recovery.load_worker_training_state(bad,restored,newopt,newsched,rank,8)
   raise AssertionError('Wrong shard world accepted')
  except ValueError:pass
  records.append(dict(case='disk_state_restore_next_update_exact',orig=orig,offload=offload,scheduler_step=206,receipt=receipt))
  del model,optimizer,scheduler,saved,payload,restored,newopt,newsched
  torch.cuda.empty_cache();dist.barrier()
out=Path(os.environ['NUMERIC_FIXTURE_OUTPUT']);out.mkdir(exist_ok=True)
(out/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,world=8,torch=torch.__version__,recovery_file=recovery.__file__,records=records),indent=2))
dist.barrier();dist.destroy_process_group()
