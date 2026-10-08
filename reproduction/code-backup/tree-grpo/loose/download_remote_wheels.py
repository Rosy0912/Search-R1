import concurrent.futures,hashlib,json,pathlib,subprocess
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');dest=root/'wheelhouse';dest.mkdir(exist_ok=True)
manifest=json.loads((root/'tools/wheelhouse-manifest.json').read_text())
def valid(p,a):
 if not p.is_file() or p.stat().st_size!=a['size']:return False
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()==a['sha256']
def fetch(a):
 p=dest/a['filename']
 if valid(p,a):print('VERIFIED_EXISTING',a['filename'],flush=True);return True
 for old in root.glob('wheelhouse-incoming-*/'+a['filename']):
  if valid(old,a):old.replace(p);print('VERIFIED_RELAY',a['filename'],flush=True);return True
 part=p.with_name(p.name+'.direct.partial');url=a['url']
 print('DOWNLOAD',a['filename'],a['size'],flush=True)
 if a['size']>32*1024**2:
  cmd=['aria2c','--continue=true','--file-allocation=none','--split=8','--max-connection-per-server=8','--min-split-size=8M','--connect-timeout=10','--timeout=60','--lowest-speed-limit=1024','--max-tries=3','--retry-wait=3','--summary-interval=60','--console-log-level=warn','--auto-file-renaming=false','--all-proxy=http://10.33.0.16:3128','--https-proxy=http://10.33.0.16:3128','--dir='+str(dest),'--out='+part.name,url]
 else:cmd=['curl','--proxy','http://10.33.0.16:3128','--http1.1','-fLsS','--connect-timeout','10','--max-time','600','--retry','2','-C','-','-o',str(part),url]
 r=subprocess.run(cmd)
 if r.returncode or not valid(part,a):print('FAILED_RETAINED',a['filename'],r.returncode,flush=True);return False
 part.replace(p);print('SOURCE_SHA_VERIFIED',a['filename'],flush=True);return True
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:ok=list(pool.map(fetch,manifest))
(root/'logs/wheelhouse-status.json').write_text(json.dumps({'complete':all(ok),'count':len(ok),'failed':[a['filename'] for a,b in zip(manifest,ok) if not b]},indent=2))
raise SystemExit(0 if all(ok) else 1)
