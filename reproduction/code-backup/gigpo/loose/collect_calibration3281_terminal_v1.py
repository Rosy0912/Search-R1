import hashlib,json,subprocess
from pathlib import Path
script=r"""/usr/bin/python3 - <<'REMOTE'
import datetime,hashlib,json,os,pathlib,subprocess
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
run=root/'runs/gigpo-e1-policy-safety-calibration25-job3281-1791243032869507143'
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':'h100-8','job_id':'3281','run':str(run),'originals':[],'queries':{},'audit_paths':[]}
paths=[root/'logs/policy-safety-calibration-terminal-3281-raw-v1.json',run/'run-result.json',run/'resolved-config.yaml',run/'ray-audit-environment-verification.json',run/'calibration.log',run/'guard.log',run/'restore25-before-baseline.json',run/'restore25-before-rollout.json',run/'baseline-evaluation.json']
logs=run/'ray/session_2026-10-06_07-32-40_067943_1882479/logs'
for pid in (1886219,1886592,1886915,1886916,1886917,1886918,1886919,1886920,1886921):
 paths.extend(sorted(logs.glob(f'worker-*-{pid}.out')));paths.extend(sorted(logs.glob(f'worker-*-{pid}.err')))
for p in paths:
 if p.is_file():
  raw=p.read_bytes();st=p.stat();rec={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}
  if len(raw)<=2000000:rec['text']=raw.decode(errors='replace')
  r['originals'].append(rec)
 else:r['originals'].append({'path':str(p),'exists':False})
for name,args in [('scheduler',['scontrol','show','job','3281','-o']),('queue',['squeue','-w','h100-8','-h','-o','%i %T %j %N']),('compute',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);r['queries'][name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for rank in range(8):
 for sub in ('runtime','policy-safety','micro'):
  p=run/f'audit/{sub}/rank-{rank}.jsonl'
  try:
   raw=p.read_bytes();r['audit_paths'].append({'path':str(p),'status':'PRESENT','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(raw.splitlines()),'text':raw.decode() if len(raw)<=1000000 else None})
  except OSError as e:
   ancestor=p.parent
   while not ancestor.exists() and ancestor!=ancestor.parent:ancestor=ancestor.parent
   r['audit_paths'].append({'path':str(p),'status':'ABSENT' if e.errno==2 else 'UNREADABLE','errno':e.errno,'error':str(e),'readable_ancestor':str(ancestor),'ancestor_readable':os.access(ancestor,os.R_OK|os.X_OK)})
result=json.loads((run/'run-result.json').read_text());cg=pathlib.Path(result['cgroup']);r['cgroup']=[]
for p in (cg,cg.parent):
 try:r['cgroup'].append({'path':str(p),'entries':sorted(x.name for x in p.iterdir()),'exists':True})
 except OSError as e:
  ancestor=p.parent
  while not ancestor.exists() and ancestor!=ancestor.parent:ancestor=ancestor.parent
  r['cgroup'].append({'path':str(p),'exists':False,'errno':e.errno,'error':str(e),'readable_ancestor':str(ancestor),'ancestor_readable':os.access(ancestor,os.R_OK|os.X_OK)})
r['material_exists']={name:(run/name).exists() for name in ('probability-material-manifest.json','probability-identity-ready.json','actual-rollout-2560.pt','actual-probability-alignment.json','full-old-versus-sampled-prefix.json')}
print(json.dumps(r))
REMOTE
"""
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=120)
assert p.returncode==0,(p.returncode,p.stderr,p.stdout[-2000:])
record=json.loads(p.stdout)
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION3281-FAILURE-PAYLOAD-v1.json')
with out.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'output':str(out.resolve()),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'utc':record['utc'],'originals':len(record['originals']),'audit_paths':record['audit_paths'],'queries':record['queries'],'cgroup':record['cgroup'],'material_exists':record['material_exists']},ensure_ascii=False))
