import os,json,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'repo'),HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1');env.pop('LD_LIBRARY_PATH',None)
p=subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/deployment/cohost-node11-v11/submit_formal.py')],cwd=r/'repo',env=env,capture_output=True,text=True,timeout=70)
print(json.dumps({'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}));raise SystemExit(p.returncode)
