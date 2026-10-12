"""Check existing shared assets and CPU imports; no training or GPU allocation."""
from pathlib import Path
import os,json,subprocess,datetime
assert os.uname().nodename=='h100-11'
b=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl');root=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
pths={'model':b/'bapo-fresh200-safety-20261006/assets/qwen2.5-3b-instruct-aa8e725','common':b/'bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data','retrieval':b/'bapo/assets-root/assets','env':root/'env/bin/python'}
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paths':{}}
for k,p in pths.items():
 r['paths'][k]={'path':str(p),'exists':p.exists(),'children':[x.name for x in p.iterdir()] if p.is_dir() else []}
code='import sys,json,torch,ray,vllm,faiss,transformers;print(json.dumps({"python":sys.executable,"torch":torch.__version__,"ray":ray.__version__,"vllm":vllm.__version__,"transformers":transformers.__version__,"faiss_gpu":hasattr(faiss,"StandardGpuResources")}))'
e=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');e.pop('LD_LIBRARY_PATH',None)
p=subprocess.run([str(pths['env']),'-B','-c',code],env=e,capture_output=True,text=True,timeout=45)
r['imports']={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(r))
