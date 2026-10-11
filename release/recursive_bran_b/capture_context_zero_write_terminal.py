"""Read-only capture of the exact no-write 726 B context failure; no signals or GPU calls."""
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
known=[dict(pid=p,uid=1001,tick=t) for p,t in [(543446,652913262),(554737,652935678),(555148,652936406),(555149,652936407),(555150,652936407)]]
for r in known:assert not Path('/proc/%d'%r['pid']).exists(), 'previous PID still exists; inspect identity before retry'
proc=subprocess.run(['tmux','display-message','-p','-t','%6','#{session_id}|#{pane_id}|#{pane_dead}|#{pane_dead_status}'],capture_output=True,text=True,check=True)
assert proc.stdout.strip()=='$6|%6|1|1'
partial=root/'checkpoints/training_state/partial_attempt_000004_outer_000004_run_f2771900501d4c8b825f2a7edaa6e7e9/manifest.json'
record=dict(protocol='recursive-b-zero-write-speed-retry-terminal-v1',time=time.time(),failed_source='7260c45fb4275651fd5ee52a104b1db80b59d5ec',failed_run='f2771900501d4c8b825f2a7edaa6e7e9',controller=known[0],tmux=dict(session_id='$5',pane_id='%6',dead=1,dead_status=1),all_known_owned_absent=True,known_owned=known,partial_manifest=str(partial),partial_manifest_sha256=hashlib.sha256(partial.read_bytes()).hexdigest(),failure='Actual all-rank partial context directory refusal before update_policy; no new optimizer write, absent worker snapshots explicitly retained',training_rc=1,no_new_completed_outer=True)
restore=root/'output-speed-resume3-f2771900501d4c8b825f2a7edaa6e7e9/sparse-schedule-migrated.json'
record.update(restore_proof=str(restore),restore_proof_sha256=hashlib.sha256(restore.read_bytes()).hexdigest(),worker_tensor_snapshot_accepted=False)
p=root/'receipts/speed726-context-zero-write-failure-terminal.json'
raw=json.dumps(record,sort_keys=True).encode()
with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
from search_r1.baselines.recursive_b_retry import validate_retry_terminal
sha=hashlib.sha256(raw).hexdigest();validate_retry_terminal(p,sha)
print(json.dumps(dict(path=str(p),sha256=sha,record=record)))
