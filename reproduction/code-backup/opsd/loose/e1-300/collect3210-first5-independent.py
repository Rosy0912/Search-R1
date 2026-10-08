from pathlib import Path
import subprocess,shlex,json,hashlib,base64,datetime
W=Path(__file__).resolve().parent
code="""from pathlib import Path
import json,hashlib,base64,subprocess
r=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210');repo=r.parents[1]/'repo-a0968c2d24be';v={'scope':'READ_ONLY_ACTUAL3210_FIRST5_COLLECTION','files':{},'source_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'source_status':subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True),'slurm':subprocess.check_output(['scontrol','show','job','3210','-o'],text=True)}
paths=[r/'train.log',r/'resolved-profile.json',r/'full_state/global_step_5/manifest.json']
mp=paths[-1]
if mp.is_file():
 m=json.loads(mp.read_text());paths += [Path(z['path']) for n,z in m['files'].items() if n.endswith('.json')]
for p in paths:
 if p.is_file():
  b=p.read_bytes();v['files'][str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'base64':base64.b64encode(b).decode()}
print(json.dumps(v))
"""
p=subprocess.run(['python3',str(W.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=60)
out=W/('actual3210-independent-first5-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir();(out/'transport.stdout-original.txt').write_bytes(p.stdout);(out/'transport.stderr-original.txt').write_bytes(p.stderr);(out/'transport-exit-original.json').write_text(json.dumps({'returncode':p.returncode}))
if p.returncode:raise SystemExit(p.returncode)
v=json.loads(p.stdout);root='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210/'
for n,record in v['files'].items():
 b=base64.b64decode(record['base64']);assert hashlib.sha256(b).hexdigest()==record['sha256'];dest=out/n.removeprefix(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
(out/'collection-summary.json').write_text(json.dumps({k:z for k,z in v.items() if k!='files'}|{'files':{n:{k:z for k,z in rec.items() if k!='base64'} for n,rec in v['files'].items()}},indent=2)+'\n');print(out)
