"""Tiny original AdamW CUDA device correctness; NO_CAPACITY_NO_TRAINING_PROOF.
Run only under ROOT-approved allocation/lease supervisor; never standalone admission.
"""
import sys
sys.dont_write_bytecode=True
import os,pathlib,json,hashlib,inspect,importlib.util,time,subprocess,gc
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
source=pathlib.Path(os.environ['R1_STREAM_SOURCE']).resolve()
out=pathlib.Path(os.environ['R1_DEVICE_PROOF_OUTPUT']).resolve()
assert root in source.parents and root/'logs' in out.parents
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NODELIST')=='h100-11'
assert os.environ.get('R1_DEVICE_PROOF_FULL8_LEASES_HELD')=='true'
assert os.environ.get('R1_APPROVED_DEVICE_CORRECTNESS_SCOPE')=='ORIGINAL_ADAMW_CUDA_DEVICE_CORRECTNESS_NOT_CAPACITY'
fd=int(os.environ['R1_DEVICE_PROOF_LEASE_FD']);os.fstat(fd)
manifest=source/'r1-files.json';m=json.loads(manifest.read_text())
for name,r in m['files'].items():
 p=source/name
 assert not p.is_symlink() and p.stat().st_size==r['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
import torch,torch.optim.adamw as original
assert torch.__version__.split('+')[0]=='2.4.0' and torch.cuda.device_count()==1
spec=importlib.util.spec_from_file_location('owned_streamed_adamw',source/'verl/workers/actor/streamed_adamw.py');stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)
public=original.AdamW._init_group;body=inspect.unwrap(public)
original.AdamW._init_group=torch._dynamo.disable(body)
a=[torch.nn.Parameter(torch.ones((256,256),device='cuda')),torch.nn.Parameter(torch.ones((128,64),device='cuda'))]
b=[torch.nn.Parameter(p.detach().clone()) for p in a]
o=original.AdamW(a,foreach=False);reference=original.AdamW(b,foreach=False);events=[];hooks=[]
o.register_step_pre_hook(lambda *args:hooks.append('pre'));o.register_step_post_hook(lambda *args:hooks.append('post'))
original_zeros=torch.zeros_like
old_proxy=stream._CPUInitTorch.zeros_like

def proxy(self,p,**kw):
 result=old_proxy(self,p,**kw)
 events.append({'input_device':str(p.device),'output_device':str(result.device),'shape':list(p.shape),'dtype':str(result.dtype)})
 assert p.device.type=='cuda' and result.device.type=='cpu'
 return result
stream._CPUInitTorch.zeros_like=proxy
result={'scope':'ORIGINAL_ADAMW_CUDA_DEVICE_CORRECTNESS_NOT_CAPACITY_NOT_TRAINING','job':os.environ['SLURM_JOB_ID'],'pid':os.getpid(),'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'status':'PENDING'}
def own_gpu_records():
 r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return [line for line in r.stdout.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
result['gpu_before_release_records']=own_gpu_records()
expected=os.environ['R1_DEVICE_PROOF_EXPECTED_UUID']
assert result['gpu_before_release_records'] and all(row.split(',')[1].strip()==expected for row in result['gpu_before_release_records'])
primary=None
try:
 for step in range(3):
  for pa,pb in zip(a,b):pa.grad=torch.full_like(pa,step*.25);pb.grad=pa.grad.clone()
  stream.step_cpu_offloaded_adamw(o);reference.step();torch.cuda.synchronize()
  for pa,pb in zip(a,b):
   assert torch.equal(pa,pb)
   for key in o.state[pa]:assert o.state[pa][key].device.type=='cpu' and torch.equal(o.state[pa][key],reference.state[pb][key].cpu())
 assert len(events)==4 and hooks==['pre','post']*3
 result.update(status='DEVICE_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING',initializer_events=events,hooks=hooks,steps=3,all_parameter_and_state_equal=True)
except BaseException as error:primary=error;result.update(status='FAIL',error=repr(error));raise
finally:
 stream._CPUInitTorch.zeros_like=old_proxy;original.AdamW._init_group=public
 result['namespace_restored']=not stream._ACTIVE and original.AdamW._init_group is public
 try:
  torch.cuda.synchronize()
  result['allocator_before_release']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
  for optimizer in (o,reference):
   for param in optimizer.param_groups[0]['params']:
    param.grad=None;param.data=param.detach().cpu()
    for key,value in optimizer.state[param].items():
     if isinstance(value,torch.Tensor):optimizer.state[param][key]=value.cpu()
  del value
  torch.cuda.synchronize()
  if not hasattr(torch._C,'_cuda_clearCublasWorkspaces'):raise RuntimeError('Fixed Torch2.4 workspace release API absent')
  torch._C._cuda_clearCublasWorkspaces();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
  result['allocator_after_release']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
  result['gpu_after_release_records']=own_gpu_records()
  assert result['allocator_after_release']=={'allocated':0,'reserved':0}
  result['all_parameters_and_states_cpu']=all(param.device.type=='cpu' and all(not isinstance(v,torch.Tensor) or v.device.type=='cpu' for v in optimizer.state[param].values()) for optimizer in (o,reference) for param in optimizer.param_groups[0]['params'])
  assert result['all_parameters_and_states_cpu']
 except BaseException as release_error:
  result['release_error']=repr(release_error);result['status']='FAIL'
  if primary is None:primary=release_error
 result['completed_epoch']=time.time()
 out.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 f=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(f,'w') as handle:json.dump(result,handle,indent=2);handle.flush();os.fsync(handle.fileno())
 d=os.open(out.parent,os.O_RDONLY);os.fsync(d);os.close(d)
if primary is not None:raise primary
print(json.dumps({'status':result['status'],'scope':result['scope'],'output':str(out)}),flush=True)
