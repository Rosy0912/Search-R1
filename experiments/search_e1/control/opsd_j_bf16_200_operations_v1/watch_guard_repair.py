import json,subprocess,time
from pathlib import Path
b=Path(__file__).parent;code=(b/'read_snapshot_guard_repair.py').read_bytes();last=None
for n in range(60):
 r=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=8','hkust1','python3 -B -'],input=code,capture_output=True,timeout=30)
 if r.returncode:print('SSH_UNKNOWN '+r.stderr.decode()[-200:],flush=True);time.sleep(30);continue
 x=json.loads(r.stdout);(b/'latest-guard-repair.json').write_bytes(r.stdout);status=x.get('status') or {};live=max([v['optimizer_updates'] for v in x.get('live_updates',[]) ]+[100]);rows=sum(a['complete_lines'] for a in x.get('sampling',[]));key=(status.get('phase'),status.get('outer_index'),live,rows)
 if key!=last:print(json.dumps(dict(unix=x['remote_unix'],phase=key[0],outer_index=key[1],actual_Adam=live,sampled_rows=rows,closed_new_outers=x.get('completed_outers'),error=status.get('error'),tmux=x.get('tmux',{}).get('stdout'))),flush=True);last=key
 if x.get('completed_outers',0)>=6:print('ELEVENTH_ACTUAL_OUTER_COMPLETE_TRAINING_CONTINUES',flush=True);break
 if ' 1 ' in x.get('tmux',{}).get('stdout',''):print('INSPECT_ACTUAL_EXIT',flush=True);break
 time.sleep(30)
