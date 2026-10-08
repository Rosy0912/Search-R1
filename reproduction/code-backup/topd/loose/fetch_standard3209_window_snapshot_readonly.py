import sys,pathlib,json,base64,hashlib,time
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
code=r"""import pathlib,json,hashlib,base64,time
p=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v14/evidence/window-actor-rollout-init.jsonl');assert not p.is_symlink()
for attempt in range(3):
 a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  f.seek(max(0,a.st_size-1024*1024));tail=f.read()
 z=p.stat()
 if (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns):break
 time.sleep(.1)
else:raise RuntimeError('Live file changed during exact snapshot; no accepted snapshot')
print(json.dumps(dict(scope='RUNNING_READ_ONLY_SNAPSHOT_NOT_FINAL_FILE',path=str(p),size=a.st_size,sha256=h.hexdigest(),offset=max(0,a.st_size-1024*1024),slice_size=len(tail),slice_sha256=hashlib.sha256(tail).hexdigest(),capture_epoch=time.time(),mtime_ns=a.st_mtime_ns,data=base64.b64encode(tail).decode())))"""
r=send(code,attempts=3,retry_safe=True,timeout=90);assert r.returncode==0,(r.stdout,r.stderr);v=json.loads(r.stdout);b=base64.b64decode(v.pop('data'));assert hashlib.sha256(b).hexdigest()==v['slice_sha256'];out=pathlib.Path('outputs/remote-evidence/standard3209-window-running-snapshot-'+str(int(v['capture_epoch'])));out.mkdir(exist_ok=False);(out/'window-tail1m.raw').write_bytes(b);(out/'original-ref.json').write_text(json.dumps(v,indent=2)+'\n');print(str(out));print(json.dumps(v))
