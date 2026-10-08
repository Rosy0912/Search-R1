"""Resume public-wheel relay after target direct public connection timed out."""
import concurrent.futures,json,os,shlex,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
PREP=Path('/Users/shaoshitong/Documents/Search-G1/preparation')
LOCAL=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/g1-common8-h100-extra-wheels')
META=json.loads((PREP/'probe-extra-wheels-public.json').read_text())
REMOTE='/dataset/shaoshitong/Search-G1-common8/bootstrap-assets/probe-extra-wheels'
STATUS=PREP/'probe-wheels-relay-status.json'
def publish(stage,**kwargs):
 tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),unix=time.time(),**kwargs)));tmp.replace(STATUS)
def remote(code,**kwargs):
 r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,timeout=240,**kwargs)
 if r.returncode:raise RuntimeError(r.stderr[-500:].decode(errors='replace'))
 return r.stdout.decode()
def transfer(item):
 target=REMOTE+'/'+item['file'];partial=target+'.relay-partial'
 for attempt in range(4):
  try:
   code='from pathlib import Path; import hashlib; p=Path('+repr(target)+'); q=Path('+repr(partial)+'); print("DONE" if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(item['sha256'])+' else str(q.stat().st_size if q.exists() else 0))'
   value=remote(code).strip()
   if value=='DONE':return item['file']
   offset=int(value);assert 0<=offset<=item['bytes']
   code='''import sys,fcntl,hashlib,os
from pathlib import Path
p=Path(TARGET);q=Path(PARTIAL);lock=open(str(q)+'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (q.stat().st_size if q.exists() else 0)==OFFSET
with q.open('ab') as out:
 while True:
  c=sys.stdin.buffer.read(1024*1024)
  if not c:break
  out.write(c)
 out.flush();os.fsync(out.fileno())
assert q.stat().st_size==SIZE
assert hashlib.sha256(q.read_bytes()).hexdigest()==EXPECTED
q.replace(p);print('VERIFIED')
'''
   for key,value in dict(TARGET=target,PARTIAL=partial,OFFSET=offset,SIZE=item['bytes'],EXPECTED=item['sha256']).items():code=code.replace(key,repr(value))
   with (LOCAL/item['file']).open('rb') as f:
    f.seek(offset);assert remote(code,stdin=f).strip()=='VERIFIED'
   return item['file']
  except Exception:
   if attempt==3:raise
   time.sleep(3)
try:
 publish('UPLOADING_VERIFIED_PUBLIC_WHEELS',total=len(META['files']))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(transfer,item) for item in META['files']]
  for count,future in enumerate(concurrent.futures.as_completed(futures),1):
   future.result();print('VERIFIED',count,flush=True)
 publish('INSTALLING_PRIVATE_ENV')
 code='''import subprocess,json,hashlib
from pathlib import Path
base=Path('/dataset/shaoshitong/Search-G1-common8');items=ITEMS;root=base/'bootstrap-assets/probe-extra-wheels'
assert all(hashlib.sha256((root/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in items)
subprocess.run(['/root/miniconda3/bin/python','-m','pip','--python',str(base/'env/bin/python'),'install','--no-index','--find-links',str(root),'--find-links',str(base/'bootstrap-assets/wheelhouse')]+[str(root/x['file']) for x in items],check=True)
subprocess.run([str(base/'env/bin/python'),'-c','import sklearn,scipy,joblib,threadpoolctl; print(sklearn.__version__,scipy.__version__,joblib.__version__,threadpoolctl.__version__)'],check=True)
print('ALL_TARGETS_VERIFIED_AND_IMPORTED')
'''.replace('ITEMS',repr(META['files']))
 print(remote(code),flush=True)
 for item in META['files']:(LOCAL/item['file']).unlink()
 publish('COMPLETE',accepted=True,file_count=len(META['files']))
except Exception as exc:
 publish('FAILED_RETAIN_LOCAL_AND_TARGET_PARTIAL',error=type(exc).__name__+': '+str(exc));raise
