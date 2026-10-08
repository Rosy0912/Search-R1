"""Restart failed migration workers from preserved partials; never starts training."""
import argparse, fcntl, json, subprocess, sys, time
from pathlib import Path

def lock_is_held(path):
 with path.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(f,fcntl.LOCK_UN);return False

def main():
 base=Path(__file__).resolve().parents[1]
 p=argparse.ArgumentParser()
 p.add_argument('--evidence-dir',type=Path,default=base/'outputs/evidence')
 p.add_argument('--primary-script',type=Path,default=base/'work/migrate_cosearch_hpc.py')
 p.add_argument('--supplement-script',type=Path,default=base/'work/migrate_cosearch_hpc_retriever_supplement.py')
 p.add_argument('--poll-seconds',type=float,default=5)
 p.add_argument('--retry-base',type=float,default=15)
 p.add_argument('--retry-max',type=float,default=120)
 a=p.parse_args();e=a.evidence_dir;e.mkdir(parents=True,exist_ok=True)
 lock=(e/'native-x-hpc-transfer-supervisor.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state={'status':'supervising','training_started':False,'retry_count':0}
 def save(**kw):
  state.update(kw,updated_epoch=time.time());q=e/'native-x-hpc-transfer-supervisor.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(q)
 try:
  for name,script in [('migration',a.primary_script),('retriever-supplement',a.supplement_script)]:
   child=None;failures=0;last_progress=None
   while True:
    q=e/f'native-x-hpc-{name}-progress.json'
    v=json.loads(q.read_text()) if q.exists() else {}
    if v.get('status')=='asset_transfer_complete_all_sha_verified':break
    if lock_is_held(e/f'native-x-hpc-{name}.lock') or (child is not None and child.poll() is None):
     progress=(v.get('verified_bytes'),v.get('current_file'),v.get('current_stream_bytes'))
     if progress!=last_progress:failures=0;last_progress=progress
     save(stage=name,status='supervising_active_transfer',worker_pid=v.get('pid'),last_worker_status=v.get('status'))
     time.sleep(a.poll_seconds);continue
    error=v.get('error','')
    if 'AssertionError' in error or 'SHA mismatch' in error:
     raise RuntimeError('Integrity failure needs review; partials preserved: '+error)
    if v.get('status')=='transfer_failed_partial_preserved':
     stamp=str(time.time_ns());(e/f'native-x-hpc-{name}-failure-{stamp}.json').write_text(json.dumps(v,indent=2)+'\n')
     failures+=1;delay=min(a.retry_max,a.retry_base*(2**min(failures-1,4)))
     save(stage=name,status='retry_backoff',last_error=error,retry_count=state['retry_count']+1,next_retry_epoch=time.time()+delay)
     time.sleep(delay)
    with (e/f'native-x-hpc-{name}-process.log').open('ab') as log:
     child=subprocess.Popen([sys.executable,str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    save(stage=name,status='worker_restarted',worker_pid=child.pid)
    time.sleep(a.poll_seconds)
  save(status='both_asset_transfers_complete_all_sha_verified',runtime_relocation_pending=True,pure_load_performed=False)
 except Exception as exc:
  save(status='supervisor_needs_review',error=repr(exc));raise

if __name__=='__main__':main()
