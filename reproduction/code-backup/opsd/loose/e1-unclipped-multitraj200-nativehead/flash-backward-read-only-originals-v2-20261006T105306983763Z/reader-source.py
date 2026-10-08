import os,sys,json,base64,hashlib,stat,platform,importlib.metadata as md
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
 raise RuntimeError('CPU-only reader requires CUDA_VISIBLE_DEVICES empty')
import torch
before=torch.cuda.is_initialized()
if before: raise RuntimeError('CUDA already initialized')
versions={name:md.version(name) for name in ('torch','transformers','flash-attn')}
def packagepath(dist,path):
 return str(md.distribution(dist).locate_file(path))
targets={
 'torch_contextlib':packagepath('torch','torch/utils/_contextlib.py'),
 'torch_init':packagepath('torch','torch/__init__.py'),
 'torch_cuda_init':packagepath('torch','torch/cuda/__init__.py'),
 'hf_flash_utils':packagepath('transformers','transformers/modeling_flash_attention_utils.py'),
 'hf_qwen2':packagepath('transformers','transformers/models/qwen2/modeling_qwen2.py'),
 'hf_modeling_utils':packagepath('transformers','transformers/modeling_utils.py'),
 'flash_interface':packagepath('flash-attn','flash_attn/flash_attn_interface.py'),
 'flash_init':packagepath('flash-attn','flash_attn/__init__.py'),
}
def read_file(path):
 st0=os.lstat(path)
 if not stat.S_ISREG(st0.st_mode):raise RuntimeError('nonregular source '+path)
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  a=os.fstat(fd)
  if (a.st_dev,a.st_ino,a.st_uid,a.st_size)!=(st0.st_dev,st0.st_ino,st0.st_uid,st0.st_size):raise RuntimeError('open identity changed')
  parts=[]
  while True:
   b=os.read(fd,1024*1024)
   if not b:break
   parts.append(b)
  raw=b''.join(parts);z=os.fstat(fd);end=os.lstat(path)
  def sig(s):return [s.st_dev,s.st_ino,s.st_uid,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
  if sig(a)!=sig(z) or sig(z)!=sig(end) or len(raw)!=a.st_size:raise RuntimeError('unstable source '+path)
  return {'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'stat':sig(a),'stat_after':sig(z),'body_base64':base64.b64encode(raw).decode()}
 finally:os.close(fd)
files={k:read_file(p) for k,p in targets.items()}
# Enumerate installed source references without importing or running flash kernels.
refs=[]
for distribution,package in [('flash-attn','flash_attn'),('transformers','transformers')]:
 base=packagepath(distribution,package)
 for dirpath,dirs,names in os.walk(base):
  dirs[:]=[d for d in dirs if d!='__pycache__']
  for name in names:
   if not name.endswith('.py'):continue
   p=os.path.join(dirpath,name)
   raw=open(p,'rb').read()
   if b'FLASH_ATTENTION_DETERMINISTIC' in raw:
    refs.append({'file':read_file(p),'matches':[{'line':i,'text':s} for i,s in enumerate(raw.decode().splitlines(),1) if 'FLASH_ATTENTION_DETERMINISTIC' in s]})
after=torch.cuda.is_initialized()
if after:raise RuntimeError('CPU source reader unexpectedly initialized CUDA')
print(json.dumps({'version':1,'scope':'CPU_ONLY_INSTALLED_SOURCE_READ_NOT_GPU_OR_JOB_ENVIRONMENT_PROOF','hostname':platform.node(),'uid':os.getuid(),'python':sys.executable,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'FLASH_ATTENTION_DETERMINISTIC_reader_environment_only':os.environ.get('FLASH_ATTENTION_DETERMINISTIC'),'torch_cuda_initialized_before':before,'torch_cuda_initialized_after':after,'versions':versions,'files':files,'installed_environment_switch_references':refs},sort_keys=True))
