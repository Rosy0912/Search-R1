"""Same-cluster original E5 encoder transfer; no index/GPU work."""
import hashlib,json,os,time,traceback,urllib.request
from pathlib import Path
root=Path(os.environ['SEARCH_G1_E5_TRANSFER_DIR']);files=json.loads((root/'files.json').read_text())
target=Path('/dataset/shaoshitong/Search-G1-common8/assets/e5-base-v2');target.mkdir(parents=True,exist_ok=True)
state={'stage':'RUNNING','pid':os.getpid(),'started_unix':time.time(),'verified':{},'accepted':False,'source_cluster':'wuhan-h100','source_node':'h100-9','target_node':'h100-8'}
def save():
 p=root/'status.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(root/'status.json')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
try:
 save();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for key,record in files.items():
  dest=target/Path(key).name
  if not dest.exists():
   partial=root/(dest.name+'.partial')
   request=urllib.request.Request('http://10.33.0.13:19875/'+key)
   with opener.open(request,timeout=30) as response,partial.open('xb') as out:
    assert response.status==200 and int(response.headers['Content-Length'])==record['size']
    while True:
     b=response.read(1024*1024)
     if not b:break
     out.write(b)
    out.flush();os.fsync(out.fileno())
   assert partial.stat().st_size==record['size'] and sha(partial)==record['sha256']
   assert not dest.exists();partial.rename(dest)
  assert dest.stat().st_size==record['size'] and sha(dest)==record['sha256']
  state['verified'][dest.name]=record;save()
 assert len(state['verified'])==8
 state.update(stage='COMPLETE',accepted=True,finished_unix=time.time());save()
except BaseException:
 state.update(stage='FAILED',error=traceback.format_exc(),finished_unix=time.time());save();raise
