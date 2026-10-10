"""Finish only this accepted five-round task's independently bound CPU observer."""
import json,os,signal,sys,time
from pathlib import Path
B=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010');cfg=json.loads((B/'receipts/training-config.json').read_text());status=json.loads((B/'runs/status.json').read_text());assert status['phase']=='TRAIN_COMPLETE' and status['outer_iterations']==5 and status['actual_optimizer_updates']==50 and status['run_id']==cfg['run_id']
accept=json.loads((B/'receipts/five-outers-accepted-cpu.json').read_text());assert accept['accepted'] and accept['run_id']==cfg['run_id'] and accept['outers']==5 and accept['real_Adam']==50
sys.path.insert(0,cfg['paths']['helper_root']);import gates
registry=json.loads((B/'receipts/scoped-cpu-processes.json').read_text());expected=registry['roles'][0]['identity'];actual=gates.process(expected['pid']);assert gates.same_process(actual,expected) and actual['run_id'].startswith('j-bf16-memory-independent-') and actual['uid']==1005
assert all(a['pid']==2239529 for a in gates.inventory()[1]);assert gates.process(2239529)['starttick']==914664099
fd=os.pidfd_open(actual['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM)
proof=dict(accepted=True,task_completed=True,only_independent_cpu_observer_signaled=actual,training_already_complete=True,training_run_id=cfg['run_id'],observed_unix=time.time());(B/'receipts/observer-finished.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
