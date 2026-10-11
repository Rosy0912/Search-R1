"""Read-only capture of the exact no-write 17fa B failure; no signals or GPU calls."""
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
known=[dict(pid=p,uid=1001,tick=t) for p,t in [(508701,652619314),(518921,652635522),(519768,652638930),(520068,652639536),(520069,652639537),(520070,652639538),(510255,652635301)]]
for r in known:assert not Path('/proc/%d'%r['pid']).exists(), 'previous PID still exists; inspect identity before retry'
proc=subprocess.run(['tmux','display-message','-p','-t','%5','#{session_id}|#{pane_id}|#{pane_dead}|#{pane_dead_status}'],capture_output=True,text=True,check=True)
assert proc.stdout.strip()=='$5|%5|1|1'
partial=root/'checkpoints/training_state/partial_attempt_000004_outer_000004/manifest.json'
record=dict(protocol='recursive-b-zero-write-speed-retry-terminal-v1',time=time.time(),failed_source='17fa8c1fc24b2399161b0a3179b3d2d300f597a0',failed_run='9e3e5f5adc174d3bba2b6c5ccac939da',controller=known[0],tmux=dict(session_id='$5',pane_id='%5',dead=1,dead_status=1),all_known_owned_absent=True,known_owned=known,partial_manifest=str(partial),partial_manifest_sha256=hashlib.sha256(partial.read_bytes()).hexdigest(),failure='Actual TensorDict missing optional actor_old_log_probs raised KeyError before any optimizer write',training_rc=1,no_new_completed_outer=True)
p=root/'receipts/speed17fa-zero-write-failure-terminal.json'
raw=json.dumps(record,sort_keys=True).encode()
with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
from search_r1.baselines.recursive_b_retry import validate_retry_terminal
sha=hashlib.sha256(raw).hexdigest();validate_retry_terminal(p,sha)
print(json.dumps(dict(path=str(p),sha256=sha,record=record)))
