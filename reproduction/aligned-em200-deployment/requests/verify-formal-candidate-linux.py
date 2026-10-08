import json,subprocess,os
from pathlib import Path
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');candidate=r/'tools/deployment/candidate-formal-v1'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'repo')+':'+str(candidate),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
results={}
for name,args in [('versions',['-c','import sys,ray,click,json;print(json.dumps(dict(prefix=sys.prefix,ray=ray.__version__,ray_file=ray.__file__,click=click.__version__,click_file=click.__file__)))']),('help',[str(candidate/'ray_cli_compat.py'),'--help']),('unbound_admission',[str(candidate/'formal_admission.py'),'verify'])]:
 p=subprocess.run([str(r/'env/bin/python'),'-B']+args,cwd=r/'repo',env=env,capture_output=True,text=True,timeout=60);results[name]=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
verified=results['help']['exit']==0 and results['unbound_admission']['exit']==1 and 'unbound independent acceptance/authorization' in results['unbound_admission']['stderr'] and json.loads(results['versions']['stdout'])['prefix']==str(r/'env')
print(json.dumps(dict(scope='ACTUAL_TARGET_LINUX_CLI_HELP_AND_UNBOUND_REFUSAL_ONLY_NO_GPU_HEAD_OR_TRAINING',verified=verified,results=results)));raise SystemExit(0 if verified else 1)
