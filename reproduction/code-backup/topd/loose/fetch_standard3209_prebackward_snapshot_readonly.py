import sys,json,hashlib,base64,pathlib,time
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
code=r'''
import pathlib,json,hashlib,base64,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v14/evidence/actor-resource-audit');rows=[]
for rank in range(8):
 p=root/('actor-pre-backward-memory-rank%d.jsonl'%rank)
 for attempt in range(3):
  a=p.stat();b=p.read_bytes();z=p.stat()
  if (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns):break
 else:raise RuntimeError('Cannot capture stable current journal '+str(p))
 rows.append(dict(path=str(p),size=len(b),sha256=hashlib.sha256(b).hexdigest(),capture_epoch=time.time(),scope='RUNNING_JOB_SNAPSHOT_NOT_FINAL_NOT_UPDATE_PROOF',data=base64.b64encode(b).decode()))
print(json.dumps(rows))
'''
r=send(code,attempts=3,retry_safe=True,timeout=60);assert r.returncode==0,r.stderr
rows=json.loads(r.stdout);out=pathlib.Path('outputs/remote-evidence/standard3209-prebackward-snapshot-'+str(int(rows[0]['capture_epoch'])));out.mkdir(exist_ok=False)
for row in rows:
 b=base64.b64decode(row.pop('data'));assert hashlib.sha256(b).hexdigest()==row['sha256'];(out/pathlib.Path(row['path']).name).write_bytes(b)
(out/'original-snapshot-refs.json').write_text(json.dumps(rows,indent=2)+'\n');print(str(out));print(json.dumps(rows))
