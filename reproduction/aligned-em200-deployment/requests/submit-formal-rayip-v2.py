import os,subprocess,json
from pathlib import Path
R=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
env=os.environ.copy();env.update(PYTHONPATH=str(R/'repo'),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
p=subprocess.run([str(R/'env/bin/python'),'-B',str(R/'tools/deployment/formal-rayip-v2/submit_formal.py')],cwd=R/'repo',env=env,capture_output=True,text=True,timeout=120)
print(json.dumps({'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'scope':'ACTUAL_SINGLE_FORMAL_SUBMISSION_NOT_COMPLETION'}));raise SystemExit(p.returncode)
