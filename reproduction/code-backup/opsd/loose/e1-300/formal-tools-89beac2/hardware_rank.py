"""Small native8 NCCL/BF16/Flash gate; no model, optimizer or RL data."""
import json,os,re,socket
from pathlib import Path
import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);job=os.environ['SLURM_JOB_ID'];root=Path(os.environ['E1_FORMAL_RUN'])
if int(os.environ['WORLD_SIZE'])!=8 or socket.gethostname().split('.')[0]!='h100-10':raise ValueError('Wrong hardware topology')
status=Path('/proc/self/status').read_text();cgroup=Path('/proc/self/cgroup').read_text()
if set(map(int,re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split()))!={os.getuid()} or not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',cgroup,re.M):raise ValueError('Foreign hardware process')
torch.cuda.set_device(local);torch.cuda.current_stream(local);prop=torch.cuda.get_device_properties(local)
# Torch2.4 lacks properties.uuid: CUDA device index maps through actual selected
# CUDA_VISIBLE_DEVICES selectors to the already recorded NVML UUID inventory.
import subprocess,csv,io
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True))))
selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',');selector=selected[local]
match=[r for r in rows if r[0].strip()==selector or r[1].strip()==selector]
if len(match)!=1 or 'H100' not in prop.name:raise ValueError('CUDA selector/physical inventory differs')
uuid=match[0][1].strip()
# Actual CUDA PCI identity confirms selector mapping, rather than a fabricated
# future worker assignment. cuda runtime UUID is read via libcudart when present.
import ctypes
lib=ctypes.CDLL('libcuda.so.1')
if lib.cuInit(0)!=0:raise ValueError('CUDA driver initialization failed')
device=ctypes.c_int()
if lib.cuCtxGetDevice(ctypes.byref(device))!=0:raise ValueError('Actual CUDA current context unavailable')
raw=(ctypes.c_ubyte*16)();rc=lib.cuDeviceGetUuid_v2(ctypes.byref(raw),device)
if rc!=0:raise ValueError('Actual CUDA context UUID unavailable')
h=bytes(raw).hex();actual='GPU-'+h[:8]+'-'+h[8:12]+'-'+h[12:16]+'-'+h[16:20]+'-'+h[20:]
if actual!=uuid:raise ValueError('CUDA context UUID differs from allocated physical selector')
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
dist.init_process_group('nccl')
try:
 x=torch.tensor([rank+1.],dtype=torch.bfloat16,device='cuda');dist.all_reduce(x)
 if x.item()!=36:raise ValueError('Actual NCCL BF16 allreduce failed')
 q=torch.randn((1,32,4,64),dtype=torch.bfloat16,device='cuda');value=flash_attn_func(q,q,q,dropout_p=0.0,causal=True);torch.cuda.synchronize()
 if value.shape!=q.shape or not torch.isfinite(value).all():raise ValueError('Actual Flash BF16 kernel failed')
 record={'rank':rank,'world_size':8,'local_rank':local,'job_id':job,'pid':os.getpid(),'uid':os.getuid(),'cgroup':cgroup,'source_commit':'89beac273f783d8e48d7d3e34da040f753514d90','uuid':uuid,'actual_cuda_uuid':actual,'gpu_name':prop.name,'allreduce_value':x.item(),'flash_finite':True,'torch':torch.__version__,'scope':'Small native hardware only; no official3B capacity/model/OPSD math claim'}
 with (root/'evidence'/f'hardware-rank-{rank}.json').open('x') as f:json.dump(record,f,sort_keys=True,allow_nan=False)
finally:dist.destroy_process_group()
