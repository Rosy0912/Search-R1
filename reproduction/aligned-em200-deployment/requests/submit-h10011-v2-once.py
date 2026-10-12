"""Run the installed single-use node11 submission; never replay an unknown intent."""
import os,subprocess,json,datetime
from pathlib import Path
assert os.uname().nodename=='h100-11'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v2-20261009')
e=dict(os.environ,PYTHONPATH=str(r/'tools')+':'+str(r/'retriever-gpu-lib')+':'+str(r/'repo'),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');e.pop('LD_LIBRARY_PATH',None)
p=subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/deployment/cohost-node11-v2/submit_formal.py')],env=e,capture_output=True,text=True,timeout=65)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'scope':'ACTUAL_NODE11_SINGLE_SUBMISSION_RESPONSE_NOT_TRAINING'}));raise SystemExit(p.returncode)
