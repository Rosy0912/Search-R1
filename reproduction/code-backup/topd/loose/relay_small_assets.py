import json,hashlib,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('work/topd/reproduction/h20').resolve()))
import download_assets as d
root=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/topd-assets-20261003')
root.mkdir(parents=True,exist_ok=True)
manifest=json.loads(Path('work/topd/reproduction/h20/assets.json').read_text())
verified=set(json.loads(Path('work/target-verified-assets.json').read_text()))
records=json.loads((root/'relay-manifest.json').read_text()) if (root/'relay-manifest.json').exists() else []
verified.update(r['path'] for r in records if r['verified'])
for a in sorted(manifest,key=lambda a:a['size'] or 0):
 if a['path'] in verified or a['size']>8*1024**3: continue
 p=root/a['path'];p.parent.mkdir(parents=True,exist_ok=True)
 url='https://hf-mirror.com'+('/datasets/' if a['kind']=='dataset' else '/')+a['repo']+'/resolve/'+a['revision']+'/'+a['file']
 if not d.valid(p,a):
  print('LOCAL_DOWNLOAD',a['path'],a['size'],flush=True)
  r=subprocess.run(['curl','--proxy','http://127.0.0.1:7892','--http1.1','-fLsS','--connect-timeout','15','--speed-time','60','--speed-limit','1024','--retry','3','--retry-all-errors','--retry-delay','3','-C','-','-o',str(p),url])
  if r.returncode or not d.valid(p,a):print('RETAINED_FAILED',a['path'],flush=True);continue
 print('UPSTREAM_VERIFIED',a['path'],flush=True)
 target='/dataset/shaoshitong/search-r1-rl/topd/assets-root/assets/'+a['path']
 r=subprocess.run([sys.executable,'work/transfer_file.py',str(p),target])
 if r.returncode:print('TARGET_PENDING',a['path'],flush=True);continue
 records.append({'path':a['path'],'size':p.stat().st_size,'sha256':d.digest(p),'target':target,'verified':True})
 (root/'relay-manifest.json').write_text(json.dumps(records,indent=2))
 p.unlink()
 print('TARGET_VERIFIED_LOCAL_RELAY_CLEANED',a['path'],flush=True)
