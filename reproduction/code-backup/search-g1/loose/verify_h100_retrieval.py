"""Read-only source asset verification; no GPU work or cross-cluster data."""
import gzip,hashlib,json,os,time,traceback
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/assets-root')
OUT=Path(os.environ['SEARCH_G1_RETRIEVAL_VERIFY_DIR'])
state={'stage':'RUNNING','pid':os.getpid(),'started_unix':time.time(),'assets':{},'accepted':False}
def save():
 t=OUT/'status.json.tmp';t.write_text(json.dumps(state,indent=2));t.replace(OUT/'status.json')
def digest(path,start=0,limit=None):
 h=hashlib.sha256();n=0
 with path.open('rb') as f:
  f.seek(start)
  while limit is None or n<limit:
   b=f.read(8*1024*1024 if limit is None else min(8*1024*1024,limit-n))
   if not b:break
   h.update(b);n+=len(b)
 if limit is not None and n!=limit:raise ValueError('short file')
 return h.hexdigest(),n
try:
 save()
 wiki=ROOT/'assets/wiki';idx=wiki/'e5_Flat.index'
 assert idx.stat().st_size==64559075373
 whole=hashlib.sha256()
 for name,start,size,expected in [('index-part-aa',0,42949672960,'a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023'),('index-part-ab',42949672960,21609402413,'b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11')]:
  part=hashlib.sha256();n=0
  with idx.open('rb') as f:
   f.seek(start)
   while n<size:
    b=f.read(min(8*1024*1024,size-n))
    if not b:raise ValueError('short index')
    part.update(b);whole.update(b);n+=len(b)
  actual=part.hexdigest();assert actual==expected,(name,actual)
  state['assets'][name]={'sha256':actual,'size':n,'accepted':True};save()
 state['assets']['index']={'path':str(idx),'sha256':whole.hexdigest(),'size':idx.stat().st_size,'accepted':True};save()
 gz=wiki/'wiki-18.jsonl.gz';actual,n=digest(gz)
 assert actual=='7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db'
 state['assets']['corpus-gzip']={'sha256':actual,'size':n,'accepted':True};save()
 raw=wiki/'wiki-18.jsonl';actual,n=digest(raw)
 h=hashlib.sha256();decoded=0
 with gzip.open(gz,'rb') as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b);decoded+=len(b)
 assert decoded==n and h.hexdigest()==actual
 state['assets']['corpus']={'path':str(raw),'sha256':actual,'size':n,'gzip_decoding_matched':True,'accepted':True};save()
 meta=json.loads((ROOT/'assets.json').read_text())
 items=meta if isinstance(meta,list) else meta.get('assets',[])
 expected={}
 for item in items:
  if 'e5-base-v2' not in json.dumps(item):continue
  name=Path(item.get('path',item.get('filename',''))).name
  expected[name]=item
 model=ROOT/'assets/e5-base-v2/model.safetensors';actual,n=digest(model)
 assert actual=='d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693'
 state['assets']['encoder-model']={'path':str(model),'sha256':actual,'size':n,'accepted':True}
 state['encoder_metadata_candidates']=len(expected)
 state['stage']='COMPLETE';state['accepted']=True;state['finished_unix']=time.time();save()
except BaseException:
 state['stage']='FAILED';state['error']=traceback.format_exc();state['finished_unix']=time.time();save();raise
