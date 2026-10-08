import sys,json,shlex,subprocess,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');p=json.loads((prep/'attempt2-native-launch.json').read_text());base=p['base'];job=p['job_id']
code='import json,subprocess,time,os;from pathlib import Path\nb=Path('+repr(base)+');job='+repr(job)+'\n'+r'''
s=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
out=dict(checked_unix=time.time(),job_stdout=s.stdout,job_stderr=s.stderr,job_returncode=s.returncode)
for n in ('run-manifest.json','pipeline-status.json','process-ledger.json','cleanup-status.json'):
 p=b/'logs'/n
 if p.exists():
  v=json.loads(p.read_text())
  if n=='run-manifest.json':v={k:v[k] for k in ('run_id','completed_updates','source_commit','train_log','checkpoint_dir')}
  out[n]=v
p=b/'logs'/('native-full-'+job+'.log');out['native_tail']=p.read_text(errors='replace')[-10000:] if p.exists() else None
live={}
for j in out.get('process-ledger.json',{}).get('jobs',{}).values():
 for e in j['ledger'].values():
  p=Path('/proc')/str(e['pid']);v=dict(expected_birth=e['start_ticks'],present=p.exists())
  if p.exists():
   st=(p/'stat').read_text().rsplit(')',1)[1].split();v.update(actual_birth=int(st[19]),actual_state=st[0],birth_matches=int(st[19])==e['start_ticks'])
  live[str(e['pid'])]=v
out['actual_processes']=live;print(json.dumps(out))
'''
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);(prep/'attempt2-latest-observation.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
