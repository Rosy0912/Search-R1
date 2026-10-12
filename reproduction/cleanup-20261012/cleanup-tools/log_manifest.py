import pathlib,json,hashlib,time
b=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl');o={'records':[],'epoch':time.time()}
for name,runname in [('tree-grpo-retained18-20261008','tree-branpo-fresh200-3360'),('tree-grpo-retained18-20261008','tree-current200-continue-3341'),('opsd-e1-unclipped-multitraj200-20261006','formal-3304')]:
 run=b/name/'runs'/runname
 files=list(run.glob('*.log'))+list(run.glob('*.json'))+list(run.glob('*.yaml'))+list((run/'output/validation-cost').glob('*.json'))
 for f in files:
  if f.stat().st_size>100*1024**2:continue
  data=f.read_bytes();sha=hashlib.sha256(data).hexdigest();snap=run/'cleanup-backup-20261012'/f.name;snap.parent.mkdir(exist_ok=True)
  if snap.exists():assert snap.read_bytes()==data
  else:snap.write_bytes(data)
  group='logs-'+runname
  o['records'].append({'group':group,'archive':str(snap),'archive_bytes':len(data),'archive_sha256':sha,'repo_path':'h100-cleanup-20261012/logs/'+runname+'/'+f.name})
print(json.dumps(o))
