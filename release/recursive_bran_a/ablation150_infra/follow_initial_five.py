"""Read-only local follower: five native boundaries plus the sixth real action."""
import argparse,json,re,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--management-socket',required=True);p.add_argument('--output',required=True)
p.add_argument('--group',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
transport=root/'h100_migration/task_transport.py';reader=Path(__file__).with_name('live_read.py')
previous=None;native_five=None
for index in range(240):
 path=out/f'snapshot-{index:04d}.json'
 r=subprocess.run([sys.executable,str(transport),'--management-socket',a.management_socket,
    '--target','h100-11','--python-file',str(reader),'--output',str(path),'--timeout','45'],capture_output=True,text=True)
 if r.returncode:
  print(json.dumps(dict(time=time.time(),event='CONNECTION_READ_UNKNOWN',rc=r.returncode,training_action='NONE',stderr=r.stderr[-500:])),flush=True)
  time.sleep(20);continue
 x=json.loads(path.read_text());entry=x.get('entry');progress=x.get('progress')
 if not entry or entry['group']!=a.group or entry['run_id']!=a.run_id:
  raise ValueError('Current driver is not this explicit ablation condition')
 process=x.get('entry_process');terminal=x.get('entry_exit');done=progress.get('completed_outer',0) if progress else 0
 if not process or process['uid']!=entry['uid'] or process['tick']!=entry['tick'] or process['state']=='Z' or terminal:
  print(json.dumps(dict(time=x['time'],event='ACTUAL_OWNER_TERMINAL_OR_IDENTITY_CHANGE',proof=str(path),exit=terminal)),flush=True);raise SystemExit(2)
 action6=False;stage=None
 for line in x.get('log',{}).get('evidence',[]):
  if 'TREE_TIMING=' in line:
   try:v=json.JSONDecoder().raw_decode(line.split('TREE_TIMING=',1)[1])[0]
   except ValueError:continue
   stage=(v.get('outer'),v.get('event'));action6|=v.get('outer')==6 and v.get('event') in ('trajectory_action','trajectory_complete')
 key=(done,stage)
 if key!=previous:
  print(json.dumps(dict(time=x['time'],event='OBSERVED_NATIVE_PROGRESS',completed_outer=done,
    adam_each_rank=progress.get('optimizer_calls_per_rank') if progress else None,
    stage=stage,log_bytes=x.get('log',{}).get('bytes'),proof=str(path))),flush=True);previous=key
 if done>=5:native_five=progress
 if native_five and action6:
  result=dict(time=x['time'],status='FIVE_NATIVE_COMPLETE_AND_SIXTH_ACTUAL_ACTION_OBSERVED',
    group=a.group,run_id=a.run_id,completed_outer=done,five_or_later_boundary=native_five,
    sixth_real_generation_action=True,live_owner=process,proof=str(path),goal_complete=False)
  (out/'first-five-accepted.json').write_text(json.dumps(result,indent=2))
  print(json.dumps(result),flush=True);raise SystemExit(0)
 time.sleep(60)
print(json.dumps(dict(event='FOLLOW_WINDOW_ENDED',goal_complete=False,training_action='NONE')),flush=True)
raise SystemExit(3)
