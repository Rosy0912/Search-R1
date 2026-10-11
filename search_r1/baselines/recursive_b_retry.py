"""CPU-only exact failed-run terminal gate for the no-write beta compatibility retry."""
import hashlib,json,os
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_retry_terminal(path, expected_sha):
 p=Path(path);assert sha(p)==expected_sha
 r=json.loads(p.read_bytes())
 assert r['protocol']=='recursive-b-zero-write-speed-retry-terminal-v1'
 contexts = {('17fa8c1fc24b2399161b0a3179b3d2d300f597a0','9e3e5f5adc174d3bba2b6c5ccac939da'): (508701,652619314,'$5','%5'), ('7260c45fb4275651fd5ee52a104b1db80b59d5ec','f2771900501d4c8b825f2a7edaa6e7e9'): (543446,652913262,'$6','%6')}
 pid,tick,session,pane=contexts[(r['failed_source'],r['failed_run'])]
 assert r['controller']==dict(pid=pid,uid=1001,tick=tick)
 assert r['tmux']==dict(session_id=session,pane_id=pane,dead=1,dead_status=1)
 assert r['all_known_owned_absent'] is True
 for identity in r['known_owned']:
  if Path('/proc/%d'%identity['pid']).exists():
   a=Path('/proc/%d/stat'%identity['pid']).read_text().rsplit(')',1)[1].split()
   assert int(a[19])!=identity['tick'], 'failed-run owner still exists'
 partial=Path(r['partial_manifest']);assert sha(partial)==r['partial_manifest_sha256']
 m=json.loads(partial.read_bytes())
 assert m['identity']['run_id']==r['failed_run'] and m['identity']['code_commit']==r['failed_source']
 assert m['completed_outer_updates']==3 and m['attempted_outer']==4 and m['continuation_decision']=='FAULT_STOP'
 if r['failed_source'].startswith('17fa'):
  assert m['worker_states_verified'] and m['all_saved_worker_states_finite']
  assert m['actual_optimizer_calls_per_rank']=={str(i):0 for i in range(4)}
  assert m['actual_scheduler_steps_per_rank']=={str(i):3 for i in range(4)}
  for w in m['worker_outcome']['worker_states']:
   assert w['total_committed_optimizer_calls']==60 and w['optimizer_opportunity_ledger']==[]
   assert 'actor_old_log_probs' in w['refusal'] and 'KeyError' in w['refusal']
 else:
  # Actual matched refusal occurs before update_policy is invoked (source726).
  # Worker tensor snapshots are absent: explicitly retain that evidence limit.
  assert m['worker_states_verified'] is False and m['all_saved_worker_states_finite'] is None
  w=m['worker_outcome']['worker_states'];assert len(w)==4 and {x['rank'] for x in w}==set(range(4))
  assert m['worker_outcome']['scheduler_advanced_actual']==[False]*4
  for x in w:
   assert x['committed_optimizer_calls']==0 and x['scheduler_step_before_actual']==x['scheduler_step_after_actual']==3
   assert x['matched_all_ranks'] and not x['safety_guard'] and x['path'] is None and x['bytes']==0
   assert 'Global partial worker context preparation refusal' in x['refusal'] and 'Partial state directory must remain in this own run and match attempted outer' in x['refusal']
  restore=Path(r['restore_proof']);assert sha(restore)==r['restore_proof_sha256']
  proof=json.loads(restore.read_bytes());assert proof['driver_actual_state_equal'] and proof['next_outer']==4
  assert len(proof['workers'])==4
  for x in proof['workers']:assert x['actual_cold_native_state_equal'] and x['actual_total_optimizer_calls']==60 and x['scheduler_step']==3 and x['adam_steps']==[60]
 assert not (partial.parent.parent/'global_step_4'/'manifest.json').exists()
 return r
