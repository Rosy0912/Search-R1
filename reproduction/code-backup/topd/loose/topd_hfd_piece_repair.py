"""Recover missing aria2 v1 pieces with requests; preserve control until full SHA."""
import hashlib, json, os, pathlib, struct, sys, time
import requests

ROOT=pathlib.Path(__file__).resolve().parent
token=sys.stdin.readline().strip();assert token
p=ROOT/'official/topd-migration-20261009/archives/wiki_corpus.tar.zst'
control=p.with_suffix(p.suffix+'.aria2')
s=requests.Session();s.trust_env=False
rpc='http://127.0.0.1:18609/jsonrpc'
active=s.post(rpc,json={'jsonrpc':'2.0','id':'check','method':'aria2.tellActive','params':[['gid']]},timeout=4).json()['result']
assert not active, 'Another writer still active'
s.post(rpc,json={'jsonrpc':'2.0','id':'close','method':'aria2.forceShutdown','params':[]},timeout=4)
time.sleep(2)
raw=control.read_bytes();assert raw[:2]==b'\0\1'
hashlen=struct.unpack_from('!I',raw,6)[0];off=10+hashlen
piece,total,upload,n=struct.unpack_from('!IQQI',raw,off);off+=24
assert total==5564130143 and n==((total+piece-1)//piece+7)//8
bits=raw[off:off+n];missing=[i for i in range((total+piece-1)//piece) if not bits[i//8]&(128>>(i%8))]
progress=ROOT/'wiki-piece-repair-progress.json';done=set()
if progress.exists():done=set(json.loads(progress.read_text()).get('repaired_pieces',[]))
(ROOT/'piece-repair-controller.json').write_text(json.dumps({'pid':os.getpid(),'epoch':time.time(),'mac_used':False,'proxy':'HKUST2_local17894','missing_pieces':len(missing)}))
s.proxies={'https':'http://127.0.0.1:17894'}
url='https://huggingface.co/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/wiki_corpus.tar.zst'
with p.open('r+b') as f:
 for i in missing:
  if i in done:continue
  start=i*piece;end=min(total,start+piece)-1
  for retry in range(1000):
   try:
    r=s.get(url,headers={'Authorization':'Bearer '+token,'Range':f'bytes={start}-{end}'},timeout=(10,30))
    assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
    assert len(r.content)==end-start+1
    f.seek(start);f.write(r.content);f.flush();os.fsync(f.fileno());done.add(i)
    progress.write_text(json.dumps({'epoch':time.time(),'repaired_pieces':sorted(done),'remaining_pieces':len(set(missing)-done),'piece_bytes':piece,'mac_used':False}));break
   except Exception as e:
    (ROOT/'wiki-piece-repair-failure.json').write_text(json.dumps({'epoch':time.time(),'piece':i,'error_type':type(e).__name__,'retry':retry+1}));time.sleep(min(30,2+retry))
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
sha=h.hexdigest();assert sha=='63756238c05cf00f5531795b965677cbea8b3cd5be6f29ff3e4b431401b24d74', 'Full archive SHA failed'
receipt={'epoch':time.time(),'group':'wiki_corpus','bytes':total,'sha256':sha,'status':'SOURCE_HF_DOWNLOAD_ARCHIVE_SHA256_VERIFIED','mac_used_for_repair':False}
(ROOT/'wiki_corpus-download-verified.json').write_text(json.dumps(receipt));control.rename(control.with_suffix('.aria2.accepted-backup'));print(json.dumps(receipt),flush=True)
