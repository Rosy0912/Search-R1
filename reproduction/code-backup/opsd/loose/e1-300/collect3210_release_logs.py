import pathlib,json,subprocess,shlex,datetime,sys
H=pathlib.Path(__file__).resolve().parent
code=r'''
import pathlib,json,hashlib,subprocess,base64,datetime,os
run=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210');pid=591546;p=pathlib.Path('/proc')/str(pid);out={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'3210','files':{}}
for n in ('stat','status','cgroup','wchan','io','syscall'):
 try:out[n]=(p/n).read_text()
 except Exception as e:out[n]={'error':str(e)}
out['cmdline']=(p/'cmdline').read_bytes().decode().replace('\0',' ');out['fds']={str(f.name):str(f.resolve()) for f in (p/'fd').iterdir() if f.name in ('1','2')};out['slurm']=subprocess.check_output(['scontrol','show','job','3210','-o'],text=True)
paths=[run/'train.log',run/'formal-wrapper.log',run/'acceptance-5-receipt.json']
for root in [run/'raytmp',run/'ray',pathlib.Path('/tmp')]:
 if root.exists():
  if root==pathlib.Path('/tmp'):
   candidates=list(root.glob('*3210*'))
  else:candidates=[root]
  for d in candidates:
   for f in d.rglob('*591546*'):
    if f.is_file() and f.suffix in ('.out','.err','.log'):paths.append(f)
for f in p.joinpath('fd').iterdir():
 if f.name in ('1','2'):
  target=f.resolve()
  if target.is_file() and str(target).startswith(('/tmp/','/mnt/data/')):paths.append(target)
for f in dict.fromkeys(paths):
 if not f.is_file():continue
 b=f.read_bytes();out['files'][str(f)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'base64':base64.b64encode(b).decode(),'tail':b[-12000:].decode(errors='replace')}
print(json.dumps(out))
'''
r=subprocess.run([sys.executable,str(H.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=75)
p=H/('formal3210-release-log-independent-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));p.mkdir();(p/'stdout').write_bytes(r.stdout);(p/'stderr').write_bytes(r.stderr)
assert r.returncode==0,r.stderr.decode();x=json.loads(r.stdout);(p/'original.json').write_bytes(r.stdout)
import base64,hashlib
for i,(path,v) in enumerate(x['files'].items()):
 b=base64.b64decode(v['base64']);assert hashlib.sha256(b).hexdigest()==v['sha256'];(p/(str(i)+'-'+pathlib.Path(path).name)).write_bytes(b)
print(p);print(json.dumps({k:v for k,v in x.items() if k!='files'},indent=2));print(json.dumps({k:{z:v[z] for z in ('sha256','bytes','tail')} for k,v in x['files'].items()},indent=2))
