"""Create only a task-local virtualenv prefix using existing shared packages."""
import os,json,datetime,subprocess,hashlib
from pathlib import Path
assert os.uname().nodename=='h100-11'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-20261009')
source=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/env/pyvenv.cfg')
pkgs=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/env/lib/python3.10/site-packages')
assert pkgs.is_dir() and (pkgs/'torch').is_dir()
e=r/'env';(e/'bin').mkdir(parents=True,exist_ok=True);(e/'lib/python3.10').mkdir(parents=True,exist_ok=True)
cfg=e/'pyvenv.cfg'
if cfg.exists():assert cfg.read_bytes()==source.read_bytes()
else:cfg.write_bytes(source.read_bytes())
for p,t in [(e/'bin/python',Path('/usr/bin/python3.10')),(e/'bin/python3',Path('/usr/bin/python3.10')),(e/'lib/python3.10/site-packages',pkgs)]:
 if p.is_symlink():assert os.readlink(p)==str(t)
 else:assert not p.exists();p.symlink_to(t)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LD_LIBRARY_PATH',None)
code='import sys,json,torch,ray,vllm,faiss,transformers,flash_attn,tensordict;print(json.dumps({"prefix":sys.prefix,"torch":torch.__version__,"ray":ray.__version__,"vllm":vllm.__version__,"transformers":transformers.__version__,"faiss_gpu":hasattr(faiss,"StandardGpuResources")}))'
p=subprocess.run([str(e/'bin/python'),'-B','-c',code],env=env,capture_output=True,text=True,timeout=55)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'packages':str(pkgs),'prefix_created_only':True,'imports':{'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}}))
