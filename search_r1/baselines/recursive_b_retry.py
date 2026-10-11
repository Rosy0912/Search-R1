"""CPU-only exact failed-run terminal gate for the no-write beta compatibility retry."""
import hashlib,json,os
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_retry_terminal(path, expected_sha):
 p=Path(path);assert sha(p)==expected_sha
 r=json.loads(p.read_bytes())
 assert r['protocol']=='recursive-b-zero-write-speed-retry-terminal-v1'
 assert r['failed_source']=='17fa8c1fc24b2399161b0a3179b3d2d300f597a0'
 assert r['failed_run']=='9e3e5f5adc174d3bba2b6c5ccac939da'
 assert r['controller']==dict(pid=508701,uid=1001,tick=652619314)
 assert r['tmux']==dict(session_id='$5',pane_id='%5',dead=1,dead_status=1)
 assert r['all_known_owned_absent'] is True
 for identity in r['known_owned']:
  if Path('/proc/%d'%identity['pid']).exists():
   a=Path('/proc/%d/stat'%identity['pid']).read_text().rsplit(')',1)[1].split()
   assert int(a[19])!=identity['tick'], 'failed-run owner still exists'
 partial=Path(r['partial_manifest']);assert sha(partial)==r['partial_manifest_sha256']
 m=json.loads(partial.read_bytes());assert m['worker_states_verified'] and m['all_saved_worker_states_finite']
 assert m['identity']['run_id']==r['failed_run'] and m['identity']['code_commit']==r['failed_source']
 assert m['completed_outer_updates']==3 and m['attempted_outer']==4 and m['continuation_decision']=='FAULT_STOP'
 assert m['actual_optimizer_calls_per_rank']=={str(i):0 for i in range(4)}
 assert m['actual_scheduler_steps_per_rank']=={str(i):3 for i in range(4)}
 for w in m['worker_outcome']['worker_states']:
  assert w['total_committed_optimizer_calls']==60 and w['optimizer_opportunity_ledger']==[]
  assert 'actor_old_log_probs' in w['refusal'] and 'KeyError' in w['refusal']
 assert not (partial.parent.parent/'global_step_4'/'manifest.json').exists()
 return r
