"""Use process-private Triton artifacts on remote shared storage before model init."""
import ast,subprocess
from finalize_topd200_draft import ROOT,main as refresh
if 'topd' in subprocess.check_output(['squeue','-u','mou176','-h','-o','%j'],universal_newlines=True).lower():raise RuntimeError('Active TOP-D job')
p=ROOT/'tools/topd-instruct300-launch-v6/runtime/observed_worker.py';s=p.read_text()
needle='  output=original_init(self,*args,**kwargs);self._resource_updates=[]'
replacement="  cache=pathlib.Path(os.environ['R1_RESOURCE_AUDIT_DIRECTORY']).parent/'cache/triton-workers'/str(os.getpid());cache.mkdir(parents=True,exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(cache)\n"+needle
if replacement not in s:
 if needle not in s:raise ValueError('Original wrapper shape differs')
 s=s.replace(needle,replacement)
ast.parse(s);p.write_text(s);refresh();print('PROCESS_PRIVATE_SHARED_STORAGE_TRITON_CACHE_SET_GPU_VALIDATION_PENDING')
