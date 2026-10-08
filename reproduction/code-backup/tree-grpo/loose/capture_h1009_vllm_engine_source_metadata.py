"""Read same-node installed source, not weights/tensors; no package imports."""
import base64,gzip,hashlib,json,socket,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
base=root/'env/lib/python3.10/site-packages/vllm'
assert base.is_dir()
names=['engine/llm_engine.py']
files={}
for name in names:
 p=base/name
 if not p.is_file():raise ValueError('Missing exact native source: '+name)
 b=p.read_bytes();assert len(b)<1024**2 and p.resolve().is_relative_to(base.resolve())
 files[name]=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),gzip_base64=base64.b64encode(gzip.compress(b)).decode())
print(json.dumps(dict(status='INSTALLED_VLLM_SOURCE_CAPTURE_ONLY_NO_RUNTIME_ACCEPTANCE',hostname=socket.gethostname(),files=files,package_imports=[],tensor_files_loaded=False,model_initialized=False,gpu_called=False,optimizer_writes=0),sort_keys=True,allow_nan=False))
