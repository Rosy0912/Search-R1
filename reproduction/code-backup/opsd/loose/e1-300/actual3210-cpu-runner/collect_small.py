"""Collect only small original stored-state/CPU evidence; binary tensors stay remote."""
from pathlib import Path
import base64,datetime,hashlib,json,shlex,subprocess
H=Path(__file__).resolve().parent;E=H.parent
ns={'__file__':str(E/'deploy_formal_reviewed.py')};s=Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns)
root='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210'
code='''import pathlib,hashlib,json,base64,os
root=pathlib.Path(ROOT);marker=root/'evidence/first5-cpu-inspection-launch.json';launch=json.loads(marker.read_bytes());out=pathlib.Path(launch['output']);assert out.resolve(strict=True)==out and out.is_relative_to(root/'evidence')
mp=root/'full_state/global_step_5/manifest.json';manifest=json.loads(mp.read_bytes());assert hashlib.sha256(mp.read_bytes()).hexdigest()==launch['manifest_sha256']
paths=[mp,marker,root/'train.log']
for name,record in manifest['files'].items():
 p=pathlib.Path(record['path'])
 if p.suffix in ('.json','.yaml','.yml'):paths.append(p)
paths.extend(p for p in out.iterdir() if p.is_file())
records=[]
for p in sorted(set(paths)):
 assert p.resolve(strict=True)==p and p.is_relative_to(root) and p.stat().st_uid==0
 size=p.stat().st_size
 if size>8*1024*1024:records.append({'path':str(p),'bytes':size,'retained_remote_only':True});continue
 data=p.read_bytes();records.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()})
print(json.dumps({'job_id':'3210','output':str(out),'manifest_sha256':launch['manifest_sha256'],'records':records,'binary_tensors_retained_remote':True},sort_keys=True))
'''.replace('ROOT',repr(root))
p=subprocess.run(ns['route']('python3 -c '+shlex.quote(code)),capture_output=True,timeout=60)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=H/('actual-small-'+stamp);dest.mkdir()
(dest/'remote-stderr-original.txt').write_bytes(p.stderr);(dest/'remote-exit-original.json').write_text(json.dumps({'returncode':p.returncode})+'\n')
if p.returncode:print(p.stderr.decode());raise SystemExit(p.returncode)
v=json.loads(p.stdout);index=[]
for n,record in enumerate(v['records']):
 record=dict(record);b=record.pop('base64',None)
 if b is not None:
  data=base64.b64decode(b);assert hashlib.sha256(data).hexdigest()==record['sha256'];file=dest/(str(n).zfill(3)+'-'+Path(record['path']).name);file.write_bytes(data);record['local_path']=str(file.resolve())
 index.append(record)
v['records']=index;(dest/'index.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');print(json.dumps({'local':str(dest),'output':v['output'],'manifest_sha256':v['manifest_sha256'],'small_records':len(index)}))
