import os,sys,json,hashlib,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/topd')
os.environ['SEARCH_R1_ASSETS']=str(root/'assets-root')
sys.path.insert(0,str(root/'repo/reproduction/h20'))
import download_assets as d
manifest=json.loads((d.BASE/'assets.json').read_text())
failed=[a['path'] for a in manifest if not d.valid(d.BASE/'assets'/a['path'],a)]
record={'asset_count':len(manifest),'verified':len(manifest)-len(failed),'failed':failed,'source':'h100-16:/dataset/shaoshitong/search-r1-rl/cosearch/assets-root/assets','target':str(d.BASE/'assets')}
(root/'logs/assets-verification.json').write_text(json.dumps(record,indent=2));print('TARGET_ASSETS_HASH_GATE',record,flush=True)
assert not failed
source=root/'logs/source-environment-file-hashes.json'
while True:
 try:env_manifest=json.loads(source.read_text());break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(5)
base=root/'cache/source-site-packages';bad=[]
for name,expected in env_manifest['hashes'].items():
 p=base/name
 if not p.is_file():bad.append(name);continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 if h.hexdigest()!=expected:bad.append(name)
result={'source':env_manifest['source'],'target':str(base),'manifest_sha256':env_manifest['manifest_sha256'],'files_checked':len(env_manifest['hashes']),'failed':bad,'installed_editable_project':'target repo; source verl editable hooks removed before install'}
(root/'logs/environment-file-hash-verification.json').write_text(json.dumps(result,indent=2));print('TARGET_ENVIRONMENT_HASH_GATE',result,flush=True)
assert not bad
(root/'logs/full-copy-verification.ready').touch()
