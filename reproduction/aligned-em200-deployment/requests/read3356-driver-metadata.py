import os,subprocess,json
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009')
code="""import torch,json
from pathlib import Path
p=Path('runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356/checkpoints/fresh200_state/attempt_1_candidate_outer_1_PARTIAL/driver.pt')
d=torch.load(p,map_location='cpu',weights_only=False)
print(json.dumps({k:(str(type(v)) if k in ('rng','sampler') else v)for k,v in d.items()},default=repr))
print(json.dumps({'sampler':{k:v for k,v in d['sampler'].items()if k not in ('order','generator')},'orderfirst':d['sampler']['order'][:10]}))"""
env=dict(os.environ,PYTHONPATH=str(root/'repo'),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1');env.pop('LD_LIBRARY_PATH',None)
p=subprocess.run([str(root/'env/bin/python'),'-B','-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=45)
print(json.dumps({'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
