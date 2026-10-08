from pathlib import Path
import os,sys,json,time
root=Path('/dataset/shaoshitong/search-r1-rl/bapo')
os.environ['SEARCH_R1_ASSETS']=str(root/'assets-root')
sys.path.insert(0,str(root/'repo/reproduction/h20'))
from download_assets import BASE,valid
manifest=json.loads((BASE/'assets.json').read_text())
while True:
 missing=[a['path'] for a in manifest if not (BASE/'assets'/a['path']).is_file() or (a['size'] and (BASE/'assets'/a['path']).stat().st_size!=a['size'])]
 if not missing:break
 print('WAIT_MISSING_CANONICAL',missing,flush=True);time.sleep(20)
failed=[a['path'] for a in manifest if not valid(BASE/'assets'/a['path'],a)]
assert not failed, failed
status={'complete':True,'asset_count':len(manifest),'content_hashes_verified':len(manifest),'failed':[],'verification':'all official SHA256/git blob SHA1 verified independently on target'}
(root/'logs/all-assets-official-verification.json').write_text(json.dumps(status,indent=2)+'\n')
p=BASE/'logs/assets-status.json';tmp=p.with_suffix('.verified.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(p)
print('ALL_ASSETS_OFFICIAL_VERIFIED',flush=True)
