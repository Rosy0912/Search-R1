import datetime, hashlib, json, subprocess
from pathlib import Path

script = r"""/usr/bin/python3 - <<'REMOTE'
import datetime, hashlib, json, pathlib, subprocess
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
run=root/'runs/gigpo-e1-policy-safety-calibration25-job3281-1791243032869507143'
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':3281,'run':str(run)}
for name,cmd in [('scheduler',['scontrol','show','job','3281','-o']),('queue',['squeue','-w','h100-8','-h','-o','%i %T %j %N'])]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
 out[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
names=['restore25-before-baseline.json','restore25-before-rollout.json','baseline-evaluation.json','updated-evaluation.json','probability-material-manifest.json','probability-identity-ready.json','run-result.json','ray-audit-environment-verification.json','actual-probability-alignment.json','full-old-versus-sampled-prefix.json']
out['records']={}
for name in names:
 p=run/name
 if p.is_file():
  raw=p.read_bytes();st=p.stat();rec={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}
  if len(raw)<=300000:rec['raw']=raw.decode();rec['data']=json.loads(raw)
  out['records'][name]=rec
 else:out['records'][name]={'path':str(p),'exists':False}
out['logs']={}
for p in [root/'logs/policy-safety-calibration-3281.log',run/'calibration.log',run/'guard.log']:
 if p.is_file():
  raw=p.read_bytes();lines=raw.decode(errors='replace').splitlines()
  keys=('PASS','ERROR','Traceback','ValueError','RuntimeError','baseline','restore','WAITING','refuse','REJECT','FAIL','ACTIVE_TRAJ_NUM','scheduler','probability','pause','checkpoint')
  selected=[line for line in lines if any(k in line for k in keys) and len(line)<1600][-35:]
  out['logs'][str(p)]={'bytes':len(raw),'sha256_snapshot':hashlib.sha256(raw).hexdigest(),'selected_lines':selected}
if run.is_dir():out['run_entries']=sorted(p.name for p in run.iterdir())
terminal=root/'logs/policy-safety-calibration-terminal-3281-raw-v1.json'
if terminal.is_file():
 raw=terminal.read_bytes();out['terminal']={'path':str(terminal),'sha256':hashlib.sha256(raw).hexdigest(),'raw':raw.decode(),'data':json.loads(raw)}
else:out['terminal']={'path':str(terminal),'exists':False}
print(json.dumps(out))
REMOTE
"""
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,capture_output=True,text=True,timeout=100)
record={'transport_returncode':p.returncode,'transport_stderr':p.stderr}
if p.returncode==0:record['actual']=json.loads(p.stdout)
else:record['raw_stdout']=p.stdout
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
path=Path('outputs/gigpo-e1-policy-safety')/('CALIBRATION3281-LIVE-'+stamp+'-v2.json')
with path.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
actual=record.get('actual',{})
summary={k:v for k,v in actual.items() if k not in ('records','logs','terminal')}
summary['records']={k:{a:b for a,b in v.items() if a not in ('raw','data')} for k,v in actual.get('records',{}).items()}
summary['logs']=actual.get('logs',{})
summary['terminal']={k:v for k,v in actual.get('terminal',{}).items() if k not in ('raw','data')}
print(json.dumps({'output':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'transport_returncode':p.returncode,'summary':summary},ensure_ascii=False))
