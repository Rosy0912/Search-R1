import concurrent.futures,hashlib,json,subprocess
from pathlib import Path
BASE=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/bapo-assets-20261003')
ASSETS=json.loads((Path(__file__).resolve().parent/'bapo/reproduction/h20/assets.json').read_text())
def fetch(a):
 dst=BASE/a['path'];dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.partial')
 if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()==a['sha256']:return True
 prefix='/datasets/' if a['kind']=='dataset' else '/'
 url='https://huggingface.co'+prefix+a['repo']+'/resolve/'+a['revision']+'/'+a['file']
 print('LOCAL_RELAY_START',a['path'],a['size'],flush=True)
 code=subprocess.call(['curl','--proxy','http://127.0.0.1:7892','--http1.1','-fLsS','--connect-timeout','15','--max-time','3600','--speed-time','60','--speed-limit','1024','--retry','2','-C','-','-o',str(tmp),url])
 if code:return False
 h=hashlib.sha256()
 with tmp.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if tmp.stat().st_size!=a['size'] or h.hexdigest()!=a['sha256']:raise RuntimeError('Relay asset content mismatch: '+a['path'])
 tmp.replace(dst);print('LOCAL_RELAY_HASH_VERIFIED',a['path'],flush=True);return True
if __name__=='__main__':
 wanted={'wiki/wiki-18.jsonl.gz','hotpotqa/dev.jsonl'}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  result=list(pool.map(fetch,[a for a in ASSETS if a['path'] in wanted]))
 (BASE/'status.json').write_text(json.dumps({'complete':all(result)}))
 raise SystemExit(0 if all(result) else 1)
