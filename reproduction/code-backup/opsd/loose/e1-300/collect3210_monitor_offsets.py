import pathlib,subprocess,sys,shlex,datetime,json
H=pathlib.Path(__file__).resolve().parent
code=r'''
import pathlib,os,json,base64,hashlib,datetime
x={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fds':{},'files':{}}
p=pathlib.Path('/proc/584849')
for f in (p/'fd').iterdir():
 try:
  dest=os.readlink(f)
  if '591546' in dest or f.name in ('1','2'):x['fds'][f.name]={'target':dest,'fdinfo':(p/'fdinfo'/f.name).read_text(),'target_size':pathlib.Path(dest).stat().st_size if pathlib.Path(dest).is_file() else None}
 except Exception as e:x['fds'][f.name]={'error':str(e)}
for name in ('log_monitor.py','worker.py'):
 f=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/ray/_private')/name;b=f.read_bytes();x['files'][str(f)]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
print(json.dumps(x))
'''
r=subprocess.run([sys.executable,str(H.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=75);p=H/('formal3210-monitor-offset-independent-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));p.mkdir();(p/'stdout').write_bytes(r.stdout);(p/'stderr').write_bytes(r.stderr);assert r.returncode==0,r.stderr.decode();x=json.loads(r.stdout);(p/'original.json').write_bytes(r.stdout)
import base64
for path,v in x['files'].items():(p/pathlib.Path(path).name).write_bytes(base64.b64decode(v['base64']))
print(p);print(x['fds'])
