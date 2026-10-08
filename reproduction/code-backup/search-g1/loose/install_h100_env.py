"""Build only the project's new Python3.10 environment from verified offline wheels."""
import json,os,shutil,subprocess,time
from pathlib import Path
BASE=Path('/dataset/shaoshitong/Search-G1-common8');PREP=BASE/'preparation'
status=PREP/'environment-status.json';env=BASE/'env';assets=BASE/'bootstrap-assets'
pid=os.getpid();ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
def publish(stage,**kwargs):
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=pid,start_ticks=ticks,unix=time.time(),**kwargs)));temp.replace(status)
def run(argv):subprocess.run(argv,check=True)
try:
 proof=json.loads((PREP/'assets-direct-status.json').read_text());assert proof['accepted'] is True
 assert not env.exists(),'Do not mutate an existing environment without inspecting its state'
 publish('CREATING_PRIVATE_ENV')
 run(['/usr/bin/python3.10','-m','venv','--without-pip',str(env)])
 pip=['/root/miniconda3/bin/python','-m','pip','--python',str(env/'bin/python'),'install','--no-index','--find-links',str(assets/'wheelhouse'),'--no-build-isolation']
 run(pip+['setuptools==70.3.0','wheel'])
 publish('INSTALLING_OFFLINE_CORE')
 run(pip+['torch==2.4.0','vllm==0.6.3','ray==2.40.0','transformers==4.46.3',
     'numpy==1.26.4','tensordict==0.5.0','hydra-core','codetiming','datasets',
     'dill','pandas','pyarrow','requests','fastapi','uvicorn','wandb','IPython','faiss-cpu==1.8.0.post1'])
 site=env/'lib/python3.10/site-packages';source=assets/'env/lib/python3.10/site-packages'
 for name in ('flash_attn','flash_attn-2.7.4.post1.dist-info','flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so'):
  p=source/name;target=site/name
  if p.is_dir():shutil.copytree(p,target)
  else:shutil.copy2(p,target)
 publish('VERIFYING_CORE_IMPORTS')
 run([str(env/'bin/python'),'-c','import torch,vllm,ray,transformers,flash_attn,faiss,pyarrow; print(torch.__version__,vllm.__version__,ray.__version__,transformers.__version__,flash_attn.__version__)'])
 publish('CORE_READY_EXTRA_PROBE_DEPENDENCIES_PENDING')
except Exception as exc:
 publish('FAILED_RETAIN_ENV',error=type(exc).__name__+': '+str(exc));raise
