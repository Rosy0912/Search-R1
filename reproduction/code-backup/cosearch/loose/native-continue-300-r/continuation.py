"""Explicitly authorized P step5 continuation; read-only resume evidence gates."""
import json,os,re,hashlib
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')

def target():
 value=os.environ.get('COSEARCH_NATIVE_CONTINUE_TO')
 if value is None:return None
 assert value=='300','This reviewed continuation is exactly total global iteration300'
 assert os.environ.get('COSEARCH_NATIVE_RELOAD_ONLY')!='1','Training continuation and pureload are mutually exclusive'
 return 300

def source_checkpoint():
 source=os.environ['COSEARCH_NATIVE_SOURCE_RUN_ID']
 assert re.fullmatch(r'author-native-[a-z0-9-]+',source)
 assert source!=os.environ['COSEARCH_NATIVE_RUN_ID']
 step=int(os.environ.get('COSEARCH_NATIVE_RESUME_STEP','5'))
 assert 0<step<300
 checkpoint=ROOT/'runs'/source/'checkpoints'/f'global_step_{step}'
 assert checkpoint.is_dir() and (checkpoint/'data.pt').is_file()
 return source,step,checkpoint

def equal_state(actual,expected):
 import torch
 import numpy as np
 if isinstance(expected,dict):return isinstance(actual,dict) and actual.keys()==expected.keys() and all(equal_state(actual[k],v) for k,v in expected.items())
 if isinstance(expected,(list,tuple)):return type(actual)==type(expected) and len(actual)==len(expected) and all(equal_state(a,b) for a,b in zip(actual,expected))
 if torch.is_tensor(expected):return torch.is_tensor(actual) and actual.dtype==expected.dtype and actual.shape==expected.shape and torch.equal(actual,expected)
 if isinstance(expected,np.ndarray):return isinstance(actual,np.ndarray) and actual.dtype==expected.dtype and np.array_equal(actual,expected)
 if expected is None or isinstance(expected,(str,int,float,bool)):return type(actual)==type(expected) and actual==expected
 raise TypeError('Unsupported original dataloader state type '+type(expected).__name__)

def after_original_load(trainer,write):
 import torch
 source,step,checkpoint=source_checkpoint()
 assert trainer.global_steps==step
 assert trainer.total_training_steps==300
 expected=torch.load(checkpoint/'data.pt',map_location='cpu',weights_only=False)
 import random,numpy as np
 cpu_rng=torch.get_rng_state();python_rng=random.getstate();numpy_rng=np.random.get_state()
 try:actual=trainer.train_dataloader.state_dict()
 finally:
  torch.set_rng_state(cpu_rng);random.setstate(python_rng);np.random.set_state(numpy_rng)
 assert equal_state(actual,expected),'Native resumed dataloader state differs from saved state'
 records=[]
 for p in (Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'])/'workers').glob('*.jsonl'):
  for line in p.read_text().splitlines():
   x=json.loads(line)
   if x.get('kind')=='checkpoint_loaded_full_digest':records.append(x)
 pairs={(role,rank) for role in ['main','reranker'] for rank in range(4)}
 assert len(records)==8 and {(x['role'],x['rank']) for x in records}==pairs
 for x in records:
  assert str(x['job'])==os.environ['SLURM_JOB_ID']
  assert Path(x['path']).parent==checkpoint.resolve()
  saved=json.loads((Path(x['path'])/f"native-integrity-rank-{x['rank']}.json").read_text())
  assert saved['state']==x['state'] and saved['source_job']==x['source_job']
 write('continuation_resume_verified',dict(source_run=source,source_step=step,next_training_step=step+1,target_total_step=300,
  dataloader_state_equal=True,dataloader_sha256=hashlib.sha256((checkpoint/'data.pt').read_bytes()).hexdigest(),
  full_state_load_ranks=8,source_job=records[0]['source_job'],original_load_called_once=True,natural_scheduler_horizon_unchanged=True))
 return records
