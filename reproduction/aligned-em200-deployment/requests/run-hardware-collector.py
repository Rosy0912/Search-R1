import subprocess,os,json
from pathlib import Path
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
p=r/'evidence'
if p.is_symlink() or p.resolve()!=p:raise RuntimeError('canonical own evidence directory required')
p.mkdir(exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
result=subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/deployment/collect_hardware_evidence.py'),'--output',str(p/'hardware3345-terminal-raw-20261009-observation1')],cwd=r/'repo',env=env,capture_output=True,text=True,timeout=90)
print(json.dumps(dict(exit=result.returncode,stdout=result.stdout,stderr=result.stderr,scope='ACTUAL_TARGET_CPU_COLLECTOR_ONLY_NOT_GPU_ACCEPTANCE')));raise SystemExit(result.returncode)
