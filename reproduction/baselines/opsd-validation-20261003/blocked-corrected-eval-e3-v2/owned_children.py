"""Normal work may run to completion; every exceptional own-child cleanup is bounded."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'strict-tools'))
from hf_launcher import stop_own_child

def run_owned(command,**kwargs):
 check=kwargs.pop('check',True);process=subprocess.Popen(command,**kwargs)
 primary=None
 try:
  code=process.wait()
  if check and code:raise subprocess.CalledProcessError(code,command)
  return subprocess.CompletedProcess(command,code)
 except BaseException as error:
  primary=error;raise
 finally:
  if process.poll() is None:
   try:stop_own_child(process,timeout=5)
   except Exception as cleanup:
    print('OWN_CHILD_BOUNDED_CLEANUP_ERROR='+json.dumps({'type':type(cleanup).__name__,'error':str(cleanup),'primary_type':None if primary is None else type(primary).__name__}),file=sys.stderr,flush=True)
    if primary is None:raise
