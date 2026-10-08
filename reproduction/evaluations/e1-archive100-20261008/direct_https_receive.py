"""Pinned-TLS concurrent same-cluster transfer, SHA-prefix resume, single worker launch."""
import concurrent.futures, fcntl, hashlib, json, os, ssl, subprocess, threading, time, urllib.parse, urllib.request
from pathlib import Path
BASE=Path(__file__).resolve().parent
CFG=json.loads((BASE/'direct-private-config.json').read_text())
SPEC=json.loads((BASE/'model-manifest.json').read_text())
CTX=ssl.create_default_context(cafile=str(BASE/'direct-source.crt'))
CTX.minimum_version=ssl.TLSVersion.TLSv1_2
LOCK=threading.Lock();PROGRESS={};START=time.time();NEW_BYTES=0

def status(phase,**fields):
 with LOCK:
  elapsed=time.time()-START
  value={'phase':phase,'pid':os.getpid(),'observed_epoch':time.time(),'elapsed_seconds':elapsed,'new_bytes':NEW_BYTES,'real_MiB_per_s':NEW_BYTES/max(elapsed,1)/1024**2,'files':dict(PROGRESS),**fields}
  p=BASE/'direct-transfer-status.json.tmp';p.write_text(json.dumps(value,indent=2)+'\n');p.replace(BASE/'direct-transfer-status.json')

def request(path,headers=None):
 h={'Authorization':'Bearer '+CFG['token'],**(headers or {})}
 req=urllib.request.Request(CFG['url']+path,headers=h)
 return urllib.request.urlopen(req,context=CTX,timeout=120)

def digest(path,length=None):
 h=hashlib.sha256()
 with path.open('rb') as f:
  remaining=length
  while remaining is None or remaining>0:
   chunk=f.read(8*1024*1024 if remaining is None else min(8*1024*1024,remaining))
   if not chunk:break
   h.update(chunk)
   if remaining is not None:remaining-=len(chunk)
 return h.hexdigest()

def receive(row):
 global NEW_BYTES
 name=row['name'];path=BASE/'model.partial'/name
 assert '/' not in name and not path.is_symlink()
 for attempt in range(3):
  size=path.stat().st_size if path.exists() else 0
  assert size<=row['bytes'],'Existing partial larger than immutable source'
  if size:
   with request('/prefix/'+urllib.parse.quote(name)+'?length='+str(size)) as r:prefix=json.load(r)
   assert prefix['length']==size and prefix['name']==name
   if digest(path)!=prefix['sha256']:
    rejected=path.with_name(name+'.rejected-'+str(time.time_ns()));path.rename(rejected);size=0
  with LOCK:PROGRESS[name]={'bytes':size,'expected_bytes':row['bytes'],'resumed_bytes':size,'phase':'receiving'}
  status('transferring')
  if size<row['bytes']:
   try:
    with request('/file/'+urllib.parse.quote(name),{'Range':f'bytes={size}-'}) as r,path.open('ab') as f:
     assert r.status==206 and r.headers['Content-Range']==f"bytes {size}-{row['bytes']-1}/{row['bytes']}"
     previous=time.time()
     while True:
      chunk=r.read(1024*1024)
      if not chunk:break
      f.write(chunk)
      with LOCK:NEW_BYTES+=len(chunk);PROGRESS[name]['bytes']+=len(chunk)
      if time.time()-previous>=2:f.flush();status('transferring');previous=time.time()
     f.flush();os.fsync(f.fileno())
    assert path.stat().st_size==row['bytes']
   except Exception:
    status('retrying',retry_file=name,attempt=attempt+1)
    if attempt==2:raise
    continue
  assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Full file SHA256 mismatch'
  with LOCK:PROGRESS[name]['phase']='verified';PROGRESS[name]['sha256']=row['sha256']
  status('transferring');return {'name':name,'bytes':row['bytes'],'sha256':row['sha256']}
 raise RuntimeError('Transfer attempts exhausted')

def main():
 os.umask(0o077)
 lock=(BASE/'direct-transfer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (BASE/'worker-launch.json').exists(),'An inference worker is already registered'
 assert not (BASE/'model').exists(),'Published model already exists'
 with request('/health') as r:assert json.load(r)=={'ready':True,'files':12}
 status('transferring',concurrency=3)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:verified=list(ex.map(receive,SPEC['files']))
 receipt={'checked_at':time.time(),'files':verified,'checkpoint_bytes':sum(x['bytes'] for x in verified),'route':'HKUST1 -> authenticated pinned-TLS -> HKUST5','concurrency':3,'elapsed_seconds':time.time()-START,'new_bytes':NEW_BYTES}
 (BASE/'direct-transfer-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
 assert not (BASE/'worker-launch.json').exists()
 log=(BASE/'worker.log').open('ab')
 p=subprocess.Popen(['python3','-B',str(BASE/'evaluate_worker.py')],cwd=str(BASE),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch={'pid':p.pid,'launched_epoch':time.time(),'script':str(BASE/'evaluate_worker.py'),'launched_after':'12 file size and SHA256 verified via direct same-cluster TLS'}
 (BASE/'worker-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 status('remote_verification_started',remote_worker=launch,verified_files=12)
 # Remove only this transfer's random HTTP authentication material after all transfers finish.
 (BASE/'direct-private-config.json').unlink()
 print(json.dumps({'phase':'remote_verification_started','worker':launch,'elapsed_seconds':receipt['elapsed_seconds'],'new_bytes':NEW_BYTES}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  status('failed',error_type=type(e).__name__,error=str(e)[:600]);raise
