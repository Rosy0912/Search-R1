"""One in-allocation 32-rank collective and tiny fresh FSDP admission."""
import datetime,json,os,socket,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from .contract import read_plan,sha
from .runtime import save

def main():
 pp=sys.argv[1];p=read_plan(pp);root=Path(p['root']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==32
 torch.set_num_threads(1);torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=240))
 h=json.loads((root/f'node-{rank//8}/hardware.json').read_text());me=dict(rank=rank,local_rank=local,node=rank//8,hostname=socket.gethostname(),uuid=h['devices'][local][0].strip(),name=torch.cuda.get_device_name(local))
 all_devices=[None]*32;dist.all_gather_object(all_devices,me)
 assert len({r['uuid'] for r in all_devices})==32 and len({r['hostname'] for r in all_devices})==4 and len({r['name'] for r in all_devices})==1
 for dtype in (torch.float32,torch.bfloat16):
  x=torch.ones(262144,device='cuda',dtype=dtype);dist.all_reduce(x);assert bool((x==32).all())
 torch.manual_seed(137);m=FSDP(torch.nn.Linear(127,19).cuda(),device_id=local,use_orig_params=False);opt=torch.optim.AdamW(m.parameters(),lr=1e-6,foreach=False)
 for _ in range(3):
  opt.zero_grad(set_to_none=True);loss=m(torch.ones(4,127,device='cuda')).square().mean();loss.backward();assert all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in m.parameters());opt.step()
 steps=[int(v['step'].item()) for v in opt.state.values()];assert steps and set(steps)=={3}
 save(root/f'warmup/rank-{rank:02d}.json',dict(at=time.time(),**me,plan_sha256=sha(pp),tiny_actual_Adam=steps,loss=float(loss.detach()),full_collective_world=32,not_production_model_or_speedup=True))
 dist.barrier()
 if rank==0:save(root/'warmup/all32-identities.json',all_devices)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
