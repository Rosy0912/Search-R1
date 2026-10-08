import pathlib,subprocess,sys,shlex,datetime,json,base64,hashlib
H=pathlib.Path(__file__).resolve().parent
code=r'''
import pathlib,os,subprocess,json,hashlib,base64,datetime
run=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210');session=pathlib.Path('/tmp/o4e-3210/ray/session_2026-10-05_00-25-10_112134_584559');out={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'processes':{},'files':{}}
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cg=(p/'cgroup').read_text()
  if '/job_3210/' not in cg:continue
  cmd=(p/'cmdline').read_bytes().decode().replace('\0',' ');name=(p/'comm').read_text().strip()
  if any(k in cmd for k in ('log_monitor','e1_import_bootstrap','formal_entry','formal_supervisor','tee','default_worker')) or name=='tee':
   rec={'cmdline':cmd,'comm':name,'stat':(p/'stat').read_text(),'status':(p/'status').read_text(),'wchan':(p/'wchan').read_text(),'cgroup':cg,'fds':{}}
   for f in (p/'fd').iterdir():
    if f.name in ('0','1','2') or name=='tee':
     try:rec['fds'][f.name]=os.readlink(f)
     except Exception as e:rec['fds'][f.name]={'error':str(e)}
   out['processes'][p.name]=rec
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
paths=[run/'train.log',run/'formal-wrapper.log',session/'logs/log_monitor.log',session/'logs/log_monitor.err',session/'logs/log_monitor.out']
# actualdriver fd1 workerfile and all python-driver log files for forwarding errors
paths+=[pathlib.Path(os.readlink('/proc/591546/fd/1'))]
paths+=list((session/'logs').glob('python-core-driver*'))
for f in paths:
 if not f.is_file():continue
 b=f.read_bytes();out['files'][str(f)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mtime_ns':f.stat().st_mtime_ns,'base64':base64.b64encode(b).decode(),'tail':b[-15000:].decode(errors='replace')}
print(json.dumps(out))
'''
r=subprocess.run([sys.executable,str(H.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=75);p=H/('formal3210-forwarding-independent-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));p.mkdir();(p/'stdout').write_bytes(r.stdout);(p/'stderr').write_bytes(r.stderr);assert r.returncode==0,r.stderr.decode();x=json.loads(r.stdout);(p/'original.json').write_bytes(r.stdout)
for i,(path,v) in enumerate(x['files'].items()):
 b=base64.b64decode(v['base64']);assert hashlib.sha256(b).hexdigest()==v['sha256'];(p/(str(i)+'-'+pathlib.Path(path).name)).write_bytes(b)
print(p);print(json.dumps({k:v for k,v in x.items() if k!='files'},indent=2));print(json.dumps({k:{z:v[z] for z in ('sha256','bytes','tail')} for k,v in x['files'].items()},indent=2))
