import concurrent.futures, importlib.util, json, os, pathlib, subprocess, time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/assets-root')
spec=importlib.util.spec_from_file_location('download_assets','/dataset/shaoshitong/search-r1-rl/tree-grpo/repo/reproduction/h20/download_assets.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
assets=[a for a in json.loads((root/'assets.json').read_text()) if (a['size'] or 0)<=32*1024**2]
def fetch(a):
 dest=root/'assets'/a['path'];dest.parent.mkdir(parents=True,exist_ok=True)
 if helper.valid(dest,a):print('VERIFIED_EXISTING',a['path'],flush=True);return True
 part=dest.with_name(dest.name+'.partial')
 if part.exists() and a['size'] and part.stat().st_size>=a['size']:
  if helper.valid(part,a):part.replace(dest);print('VERIFIED_PARTIAL',a['path'],flush=True);return True
  part.rename(part.with_name(part.name+'.rejected.'+str(time.time_ns())))
 url='http://127.0.0.1:19993/'+a['path']
 print('COPY_START',a['path'],a['size'],flush=True)
 if (a['size'] or 0)>32*1024**2:
  # A separate partial prevents aria2 from mistaking an old sequential HF
  # partial for its own segmented download state; the old copy is retained.
  part=dest.with_name(dest.name+'.source.partial')
  cmd=['aria2c','--continue=true','--file-allocation=none','--split=8','--max-connection-per-server=8','--min-split-size=8M','--connect-timeout=10','--timeout=60','--lowest-speed-limit=1024','--max-tries=2','--retry-wait=3','--summary-interval=60','--console-log-level=warn','--auto-file-renaming=false','--all-proxy=','--http-proxy=','--dir='+str(part.parent),'--out='+part.name,url]
 else:
  cmd=['curl','--noproxy','*','-fLsS','--connect-timeout','10','--speed-time','60','--speed-limit','1024','--retry','1','-C','-','-o',str(part),url]
 r=subprocess.run(cmd)
 if r.returncode or not helper.valid(part,a):print('COPY_FAILED_RETAINED',a['path'],r.returncode,flush=True);return False
 part.replace(dest);print('COPY_SHA_VERIFIED',a['path'],a['size'],flush=True);return True
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(fetch,assets))
(root/'logs/source-small-status.json').write_text(json.dumps({'complete':all(results),'source':'read-only fixed H20 assets via existing SSH routing','failed':[a['path'] for a,ok in zip(assets,results) if not ok]},indent=2))
raise SystemExit(0 if all(results) else 1)
