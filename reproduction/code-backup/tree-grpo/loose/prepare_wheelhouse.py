import concurrent.futures, hashlib, json, pathlib, re, subprocess, urllib.request
from packaging.tags import cpython_tags, compatible_tags
from packaging.utils import parse_wheel_filename
base=pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/tree-grpo-deps-20261002')
base.mkdir(parents=True,exist_ok=True)
packages=json.loads(pathlib.Path('work/linux-packages.json').read_text())
platforms=[f'manylinux_2_{i}_x86_64' for i in range(35,4,-1)]+['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']
tags=list(cpython_tags((3,10),['cp310'],platforms))+list(compatible_tags((3,10),'cp310',platforms)); ranking={t:i for i,t in enumerate(tags)}
def resolve(pair):
 name,version=pair
 with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as response: metadata=json.load(response)
 candidates=[]
 for f in metadata['urls']:
  if f['packagetype']=='bdist_wheel':
   rank=min((ranking[t] for t in parse_wheel_filename(f['filename'])[3] if t in ranking),default=None)
   if rank is not None: candidates.append((rank,f))
 f=min(candidates,key=lambda x:x[0])[1] if candidates else next(f for f in metadata['urls'] if f['packagetype']=='sdist')
 return dict(package=name,version=version,filename=f['filename'],url=f['url'],sha256=f['digests']['sha256'],size=f['size'],kind=f['packagetype'])
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: manifest=list(pool.map(resolve,packages))
# Include the deliberate final cuBLAS override as a separate no-deps install.
manifest.append(resolve(('nvidia-cublas-cu12','12.4.5.8')))
(base/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('SELECTED',len(manifest),'SIZE_GIB',round(sum(f['size'] for f in manifest)/1024**3,2),flush=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def fetch(f):
 p=base/f['filename']
 if p.exists() and p.stat().st_size==f['size'] and digest(p)==f['sha256']:return
 partial=p.with_suffix(p.suffix+'.partial')
 print('DOWNLOAD',f['filename'],f['size'],flush=True)
 r=subprocess.run(['curl','-fLsS','--connect-timeout','15','--speed-time','60','--speed-limit','1024','--retry','1','-C','-','-o',str(partial),f['url']])
 r.check_returncode()
 assert partial.stat().st_size==f['size'] and digest(partial)==f['sha256'],f['filename']
 partial.rename(p);print('VERIFIED',f['filename'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,manifest))
(base/'download.complete').write_text('All PyPI source SHA256 verified. Not yet transferred.\n')
print('WHEELHOUSE_DOWNLOAD_COMPLETE',flush=True)
