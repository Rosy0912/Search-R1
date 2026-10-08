"""Hash reused runtime files, local GPU FAISS library and build headers in place."""
import json,hashlib,os,sys,subprocess
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 return h.hexdigest()
records=[]
roots=[(root/'env').resolve(),Path('/mnt/data/shaoshitong/search-r1-rl/gigpo/retriever-gpu-lib'),Path('/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include')]
for directory in roots:
 for p in sorted(directory.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
   records.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
import torch,transformers,ray,vllm
value=dict(status='READ_AND_HASHED',executable=sys.executable,prefix=sys.prefix,python=sys.version,
 torch=torch.__version__,transformers=transformers.__version__,ray=ray.__version__,vllm=vllm.__version__,
 source_commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),files=records)
p=root/'logs/actual-environment-full-sha-v1.json';p.write_text(json.dumps(value,indent=2))
print(json.dumps(dict(path=str(p),files=len(records),sha256=digest(p),torch=torch.__version__)))
