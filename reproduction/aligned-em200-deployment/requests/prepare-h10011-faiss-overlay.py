"""Expose existing shared GPU FAISS in the isolated task prefix; no global changes."""
from pathlib import Path
import os,subprocess,json,datetime
assert os.uname().nodename=='h100-11'
old=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/retriever-gpu-lib');target=old
if old.is_symlink():
 raw=os.readlink(old)
 target=Path('/mnt/shared/workspace/training_data_node_5'+raw[len('/dataset'):]) if raw.startswith('/dataset/') else old.resolve()
assert target.is_dir(),str(target)
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-20261009');p=r/'retriever-gpu-lib'
if p.is_symlink():assert os.readlink(p)==str(target)
else:assert not p.exists();p.symlink_to(target)
env=dict(os.environ,PYTHONPATH=str(p),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LD_LIBRARY_PATH',None)
p=subprocess.run([str(r/'env/bin/python'),'-B','-c','import torch,faiss,json;print(json.dumps({"path":faiss.__file__,"gpu":hasattr(faiss,"StandardGpuResources")}))'],env=env,capture_output=True,text=True,timeout=30)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'shared_gpu_overlay':str(target),'imports':{'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}}))
