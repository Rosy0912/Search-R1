import json,os,subprocess,hashlib,re
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006')
p=r/'logs/cpu-gate-durable.log'
assert not p.exists() and not (r/'logs/cpu-gate-durable-exit.json').exists()
env=dict(os.environ,PYTHONPATH=str(r/'repo'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
with p.open('x') as log:
 c=subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/cpu-tests.py')],cwd=r/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
record={'exit_code':c.returncode,'source_commit':'484c0b30fa73fa5354bdb3452f007134ec5db13f','log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE','training_submitted':False}
with (r/'logs/cpu-gate-durable-exit.json').open('x') as f:f.write(json.dumps(record)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(c.returncode)
