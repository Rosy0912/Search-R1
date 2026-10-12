import os,json,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
(r/'evidence').mkdir(exist_ok=True)
assert (r/'evidence').resolve()==r/'evidence'
out=r/'evidence/first5-cpu3359-20261009-1835'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LD_LIBRARY_PATH',None);env.pop('PYTHONOPTIMIZE',None)
p=subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/deployment/first5-node11-3359/aligned_first5_cpu_evidence.py'),'--output',str(out)],cwd=r/'repo',env=env,capture_output=True,text=True,timeout=1500)
print(json.dumps({'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'output':str(out)}))
if (out/'cpu-evidence.json').exists():print((out/'cpu-evidence.json').read_text())
raise SystemExit(p.returncode)
