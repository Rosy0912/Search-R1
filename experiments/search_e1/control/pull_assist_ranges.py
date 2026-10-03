"""Transfer public asset shards inside HKUST; verify each before final LFS checks."""
import os,json,time,urllib.request,urllib.parse,urllib.error,hashlib,shutil,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
base=Path(os.environ['E1_BASE']);url='http://10.120.17.101:18282'
for _ in range(1080):
 try:
  shards=json.loads(urllib.request.urlopen(url+'/range-shards.json',timeout=15).read());break
 except (urllib.error.HTTPError,urllib.error.URLError):time.sleep(10)
else:raise RuntimeError('Assist shard manifest not ready after three hours')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def transfer(s):
 rel=Path(s['path'])
 if rel.is_absolute() or '..' in rel.parts or not str(rel).startswith('segmented/'):raise RuntimeError('Invalid shard path')
 dest=base/'assets'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and dest.stat().st_size==s['bytes'] and digest(dest)==s['sha256']:return
 temp=dest.with_suffix('.lan-transfer')
 with urllib.request.urlopen(url+'/assets/'+urllib.parse.quote(str(rel),safe='/'),timeout=60) as src,temp.open('wb') as out:shutil.copyfileobj(src,out,8*1024**2)
 if temp.stat().st_size!=s['bytes'] or digest(temp)!=s['sha256']:raise RuntimeError('LAN shard hash mismatch')
 temp.replace(dest);print('LAN_SHARD_VERIFIED',str(rel),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(transfer,shards))
(base/'receipts/assist-shards-verified.json').write_text(json.dumps(shards,indent=2))
# All ranges are now local: assembled files must still match published LFS hashes.
subprocess.run(['/home/shaoshitong/miniconda3/envs/EditVerse/bin/python',str(base/'source/experiments/search_e1/control/fast_asset_download.py')],env=dict(os.environ,E1_DOWNLOAD_WORKERS='8'),check=True)
