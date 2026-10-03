"""Copy official-source wheels/model/data within HKUST, checking recorded hashes."""
import urllib.request,urllib.parse,time,hashlib,json,subprocess,os
from pathlib import Path
base=Path(os.environ['E1_BASE']);url=os.environ.get('E1_PEER_URL','http://10.123.4.11:18281')
def fetch(path):
 return urllib.request.urlopen(url+'/'+path,timeout=30).read()
def transfer(remote,dest,sha):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==sha:return
 temp=dest.with_name(dest.name+'.part');h=hashlib.sha256()
 with urllib.request.urlopen(url+'/'+urllib.parse.quote(remote,safe='/'),timeout=60) as src,temp.open('wb') as out:
  for block in iter(lambda:src.read(8*1024*1024),b''):out.write(block);h.update(block)
 if h.hexdigest()!=sha:raise RuntimeError('Peer hash mismatch: '+remote)
 temp.replace(dest);print('COPIED',remote,flush=True)
for i in range(4320):
 try:
  raw=fetch('wheels.sha256').decode();break
 except Exception:time.sleep(10)
else:raise SystemExit('Wheel source not ready after twelve hours; see source log')
for line in raw.splitlines():
 sha,path=line.split(maxsplit=1);name=Path(path).name
 transfer('wheels/'+name,base/'wheels'/name,sha)
py=base/'env/bin/python';req=base/'source/experiments/search_e1/control/requirements-hkust.txt'
subprocess.run([str(py),'-m','pip','install','--no-index','--find-links',str(base/'wheels'),'-r',str(req)],check=True)
flash=list((base/'wheels').glob('flash_attn*.whl'));assert len(flash)==1
subprocess.run([str(py),'-m','pip','install','--no-deps',str(flash[0])],check=True)
subprocess.run([str(py),'-m','pip','check'],check=True)
(base/'receipts').mkdir(exist_ok=True)
(base/'receipts/environment-freeze.txt').write_bytes(subprocess.check_output([str(py),'-m','pip','freeze']))
(base/'receipts/env.ready').touch()
for i in range(4320):
 try:
  manifest=json.loads(fetch('assets/manifest.json'))
  if all(k in manifest for k in ['model','encoder','datasets']):break
 except Exception:pass
 time.sleep(10)
else:raise SystemExit('Model/data source not ready after twelve hours')
for name in ['model','encoder','datasets']:
 for entry in manifest[name]['files']:
  transfer('assets/'+name+'/'+entry['path'],base/'assets'/name/entry['path'],entry['sha256'])
(base/'assets/manifest.json').write_text(json.dumps({k:manifest[k] for k in ['model','encoder','datasets']},indent=2))
(base/'receipts/local-assets.ready').touch();print('PEER_READY',flush=True)
