from pathlib import Path
import os,subprocess,json,hashlib,datetime
root=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');repo=root/'repo'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo))
cmd=[str(root/'env/bin/python'),'-B',str(root/'tools/deployment/capture_baseline_order.py'),'--aligned-root',str(repo),'--common-manifest','/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data/manifest.json','--model',str(root/'assets/qwen2.5-3b-instruct-aa8e725'),'--output',str(root/'inputs/baseline-observed-order.json')]
log=root/'logs/baseline-order-capture.log';record=root/'logs/baseline-order-capture.json'
assert not log.exists() and not record.exists(),'inspect existing order capture before retry'
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with log.open('x')as f:p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1500)
v={'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':p.returncode,'command':cmd,'log':str(log),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'tail':log.read_text()[-4500:],'scope':'ACTUAL_NEW_COMMON169028_REFERENCE_LOADER_CAPTURE_NOT_HISTORICAL169103'}
with record.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v));raise SystemExit(p.returncode)
