import argparse,base64,datetime,hashlib,json,os,pathlib,shlex,subprocess,sys,time,types
P=argparse.ArgumentParser();P.add_argument('--peer-report',required=True);P.add_argument('--peer-sha',required=True);A=P.parse_args()
H=pathlib.Path(__file__).resolve().parent
C=H/'checkpoint50-observation-candidate/collect_checkpoints50_100.py'
PIN='e459fb4961183b48226e20ebe7ae4595a92ec27ac8b569a76eb5befc58bc2b9f'
B=C.read_bytes();assert hashlib.sha256(B).hexdigest()==PIN
Q=pathlib.Path(A.peer_report);peer=Q.read_bytes();assert hashlib.sha256(peer).hexdigest()==A.peer_sha
json.loads(peer)
T=H/'transport.py';t=T.read_bytes();assert hashlib.sha256(t).hexdigest()=='a64bf52a97cbef6e3b0cc7948e095546687c8f1690094469c4937180f9449a85'
D=H/'checkpoint50_100-collection-once-v1-original';D.mkdir(mode=0o700)
def save(n,b):
 fd=os.open(D/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def js(n,x):save(n,json.dumps(x,sort_keys=True,allow_nan=False).encode())
def syncdir():
 fd=os.open(D,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
save('caller-source.original.py',pathlib.Path(__file__).read_bytes());save('collector-source.original.py',B);save('transport-source.original.py',t);save('software-peer.original.json',peer)
bootstrap='import sys,hashlib; b=sys.stdin.buffer.read(); assert hashlib.sha256(b).hexdigest()=='+repr(PIN)+'; sys.argv=["collect_checkpoints50_100.py","--steps","50","100"]; exec(compile(b,"<root_fixed_readonly_collector>","exec"),{"__name__":"__main__"})'
save('bootstrap-source.original.py',bootstrap.encode())
m=types.ModuleType('fixedtransport');exec(compile(t,str(T),'exec'),m.__dict__)
argv=m.route('python3 -B -c '+shlex.quote(bootstrap))
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
js('intent.original.json',{'utc':start,'argv':argv,'collector_sha256':PIN,'caller_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'peer_report':str(Q),'peer_sha256':A.peer_sha,'requested_steps':[50,100],'timeout_seconds':2400,'wait_slice_seconds':30,'scope':'Once readonly sameh10010 metadata and wholeSHA; no PT/HF load, no remote files/newjob/GPU/sign/source modification'})
syncdir();p=None;waited=False;rc=None;complete=False;error=None
try:
 with (D/'stdout.original').open('xb') as out,(D/'stderr.original').open('xb') as err:
  p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=out,stderr=err)
  js('popen.original.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'argv':argv});syncdir()
  p.stdin.write(B);p.stdin.flush();p.stdin.close();complete=True
  js('stdin.original.json',{'complete':True,'bytes':len(B),'sha256':PIN,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});syncdir()
  deadline=time.monotonic()+2400
  while time.monotonic()<deadline:
   try:rc=p.wait(timeout=min(30,max(.01,deadline-time.monotonic())));waited=True;break
   except subprocess.TimeoutExpired:pass
  out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
js('wait.original.json',{'returncode':rc,'waited':waited,'pid':p.pid if p else None,'stdin_complete':complete,'error':error,'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'True own SSH wait only; timeout or error remains UNKNOWN, no signal/retry'})
syncdir()
print(json.dumps({'directory':str(D),'returncode':rc,'waited':waited,'stdin_complete':complete,'error':error,'stdout_bytes':(D/'stdout.original').stat().st_size if (D/'stdout.original').exists() else None,'stderr_bytes':(D/'stderr.original').stat().st_size if (D/'stderr.original').exists() else None},sort_keys=True),flush=True)
sys.exit(0 if waited and rc==0 and error is None else 1)
