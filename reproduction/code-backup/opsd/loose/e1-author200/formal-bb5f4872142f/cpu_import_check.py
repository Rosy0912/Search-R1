"""Current native CPU imports validated with external inventory; no GPU contexts."""
import hashlib,json,os,sys,types
from pathlib import Path
repo=Path(os.environ['SEARCH_R1_REPO']);run=Path(os.environ['E1_FORMAL_RUN'])
sys.path[:]=[str(repo)]+[p for p in sys.path if p and Path(p).resolve()!=repo]
from owned_runtime import pinned,physical
binding=pinned(os.environ['OPSD_E1_BINDING_MANIFEST'],os.environ['OPSD_E1_BINDING_MANIFEST_SHA256'])
record=binding['files']['import_closure'];closure=pinned(record['path'],record['sha256'])
name='reproduction.baselines.e1_import_bootstrap';r=closure['modules'][name];path=physical(repo/'reproduction/baselines/e1_import_bootstrap.py')
if r['path']!=str(path) or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Checker inventory pin differs')
module=types.ModuleType(name);module.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True),module.__dict__);sys.modules[name]=module
import torch
if torch.cuda.is_initialized():raise ValueError('Unexpected pre-approval CUDA context')
actual,count=module.verify_imports(repo,closure)
from search_r1.baselines.checkpoint_gate import verify_import_closure
verify_import_closure(record,run,repo)
if torch.cuda.is_initialized():raise ValueError('CPU import checker initialized CUDA before approval')
print(json.dumps({'source_commit':'188f4e5e7b63cf479290d0888a4a075499886750','pid':os.getpid(),'uid':os.getuid(),'job_id':os.environ['SLURM_JOB_ID'],'repo':str(repo),'module_count':len(actual['modules']),'function_code_count':count,'actual_loaded_modules':actual,'closure_sha256':record['sha256'],'cuda_is_initialized':False,'scope':'Actual loaded CPU subset; future Ray driver/ranks separately checked by frozen source'},sort_keys=True,allow_nan=False))
