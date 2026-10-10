"""Keep the current turn observing actual progress; no training actions, raw downloads or GPU work."""
import json,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parents[1];source=(B/'control/watch_snapshot.py').read_bytes();start=time.time();last=None;n=0
while time.time()-start<7200:
 n+=1;r=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=8','hkust1','python3 -B -'],input=source,capture_output=True,timeout=30)
 if r.returncode:
  print(json.dumps({'observation':n,'ssh_rc':r.returncode,'error':r.stderr.decode()[-500:]}),flush=True);time.sleep(20);continue
 x=json.loads(r.stdout);(B/'latest-snapshot.json').write_text(json.dumps(x,indent=2)+'\n');(B/'snapshots-recovery4').mkdir(exist_ok=True);(B/'snapshots-recovery4'/f'{n:04d}.json').write_bytes(r.stdout+b'\n')
 live=max([v['optimizer_updates'] for v in x['live_updates']]+[0]);sample=sum(v['complete_lines'] for v in x['sampling']);key=(x['status']['phase'],x['status']['outer_index'],x['completed_outers'],live,sample,x['tmux']['stdout'])
 if key!=last:
  print(json.dumps({'remote_unix':x['remote_unix'],'phase':key[0],'outer_index':key[1],'completed_outers':key[2],'real_Adam':live,'sample_complete_rows':sample,'observer_age':x.get('memory',{}).get('age_s'),'memory_below':x.get('memory',{}).get('samples_below_limit'),'tmux':x['tmux']['stdout'],'error':x['status'].get('error')}),flush=True);last=key
 if x['status']['phase'] in ('FAILED','FAILURE_PRESERVED') or ' 1 ' in x['tmux']['stdout']:
  print('TERMINAL_INSPECT_REQUIRED',flush=True);break
 if x['completed_outers']==5:
  print('FIVE_CLOSED_OUTERS_REQUIRE_FINAL_ACCEPTANCE',flush=True);break
 time.sleep(40)
