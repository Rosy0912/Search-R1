"""Verify own copies against fixed assets and source package-byte manifest."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(os.environ['AGENT_ROOT'])
ready=ROOT/'logs/full-copy-verification.ready'
if ready.exists():ready.rename(ready.with_name(ready.name+'.previous.'+str(time.time_ns())))
os.environ['SEARCH_R1_ASSETS']=str(ROOT/'assets-root')
sys.path.insert(0,str(ROOT/'repo/reproduction/h20'))
from download_assets import valid
assets=json.loads((ROOT/'assets-root/assets.json').read_text())
failed=[a['path'] for a in assets if not valid(ROOT/'assets-root/assets'/a['path'],a)]
assert not failed,failed
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((ROOT/'logs/source-environment-file-hashes.json').read_text())
source=ROOT/'cache/source-site-packages'
installed=ROOT/'env/lib/python3.10/site-packages'
changed=[]
for name,expected in manifest['hashes'].items():
 assert digest(source/name)==expected,('copied_source_package',name)
 first=Path(name).parts[0]
 replace=('verl' in first and (first.startswith('__editable__') or first=='verl.egg-link' or first.endswith('.dist-info')))
 replace=replace or first=='click' or first.startswith('click-')
 if replace:changed.append(name)
 else:assert digest(installed/name)==expected,('installed_package',name)
derived={
 'wiki/e5_Flat.index':'69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166',
 'wiki/wiki-18.jsonl':'85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9'}
for name,expected in derived.items():assert digest(ROOT/'assets-root/assets'/name)==expected,name
result={'original_assets_verified':len(assets),'source_environment_files_verified':len(manifest['hashes']),
 'source_manifest_sha256':manifest['manifest_sha256'],'source':'h100-16 read-only CoSearch cache',
 'intentional_own_installed_replacements':changed,'replacement_reason':'own verl editable link; Ray-compatible Click8.2.1',
 'derived_sha256':derived,'runtime_retrieval_policy':'all on h100-8; no A800 or cross-node RPC'}
(ROOT/'logs/cluster-copy-verification.json').write_text(json.dumps(result,indent=2))
(ROOT/'logs/full-copy-verification.ready').touch()
print('ALL_CLUSTER_COPY_CONTENT_HASHES_PASS',len(assets),len(manifest['hashes']),flush=True)
