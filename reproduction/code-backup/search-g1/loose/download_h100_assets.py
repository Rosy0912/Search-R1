"""Resume allowlisted files over the H100 private network and verify all hashes."""
import concurrent.futures,hashlib,json,os,time,urllib.request,urllib.parse
from pathlib import Path
BASE=Path('/dataset/shaoshitong/Search-G1-common8')
PREP=BASE/'preparation'; ROOT=BASE/'bootstrap-assets'
EXPECTED=json.loads((PREP/'source-hashes.json').read_text())
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def transfer(item):
 name,expected=item; dest=ROOT/name; dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if digest(dest)==expected: return name
  raise RuntimeError('Existing completed asset has wrong hash: '+name)
 partial=dest.with_name(dest.name+'.g1-partial')
 url='http://10.33.0.13:19873/'+urllib.parse.quote(name)
 for attempt in range(5):
  try:
   size=partial.stat().st_size if partial.exists() else 0
   headers={'Range':f'bytes={size}-'} if size else {}
   with OPENER.open(urllib.request.Request(url,headers=headers),timeout=60) as response:
    assert response.status==(206 if size else 200)
    with partial.open('ab' if size else 'wb') as f:
     while True:
      chunk=response.read(1024*1024)
      if not chunk: break
      f.write(chunk)
   assert digest(partial)==expected, 'Downloaded hash mismatch: '+name
   partial.replace(dest); return name
  except Exception:
   # A complete file may remain after a transient final-response failure.
   if partial.exists() and digest(partial)==expected:
    partial.replace(dest); return name
   if attempt==4: raise
   time.sleep(2)
pid=os.getpid(); ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
status=PREP/'assets-direct-status.json'
def publish(stage,**kwargs):
 temp=status.with_suffix('.tmp'); temp.write_text(json.dumps(dict(stage=stage,pid=pid,start_ticks=ticks,unix=time.time(),**kwargs)));temp.replace(status)
try:
 publish('DOWNLOADING',total=len(EXPECTED))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(transfer,item) for item in EXPECTED.items()]
  for count,future in enumerate(concurrent.futures.as_completed(futures),1):
   name=future.result(); publish('DOWNLOADING',completed=count,total=len(EXPECTED),last=name)
 assert all(digest(ROOT/name)==sha for name,sha in EXPECTED.items())
 publish('COMPLETE',accepted=True,file_count=len(EXPECTED))
except Exception as exc:
 publish('FAILED_RETAIN_PARTIAL',error=type(exc).__name__+': '+str(exc));raise
