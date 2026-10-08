import hashlib,json,re,subprocess
from pathlib import Path
BASE=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/bapo-assets-20261003')
a=next(a for a in json.load(open(Path(__file__).resolve().parent/'bapo/reproduction/h20/assets.json')) if a['path']=='wiki/wiki-18.jsonl.gz')
dest=BASE/a['path'];temp=dest.with_name(dest.name+'.partial');dest.parent.mkdir(parents=True,exist_ok=True)
url='https://hf-mirror.com/datasets/'+a['repo']+'/resolve/'+a['revision']+'/'+a['file']
cmd=['aria2c','--all-proxy=http://127.0.0.1:7892','--continue=true','--file-allocation=none','--max-connection-per-server=8','--split=8','--min-split-size=64M','--connect-timeout=15','--timeout=60','--lowest-speed-limit=1024','--max-tries=3','--retry-wait=5','--summary-interval=60','--console-log-level=warn','--auto-file-renaming=false','--dir='+str(temp.parent),'--out='+temp.name,url]
print('LOCAL_WIKI_RESUME_MIRROR_ARIA_START',flush=True)
p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
for line in p.stdout:
 if re.search(r'http|token|signature|credential|authorization',line,re.I):
  codes=re.findall(r'(?:errorCode|status)=(\d+)',line)
  if codes:print('LOCAL_WIKI_TRANSFER_ERROR_CODES',codes,flush=True)
  continue
 if line.startswith('[#') or 'Download Results' in line or 'errorCode=' in line:print(line.strip(),flush=True)
code=p.wait()
if code:raise SystemExit(code)
h=hashlib.sha256()
with temp.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert temp.stat().st_size==a['size'] and h.hexdigest()==a['sha256'],'Official corpus hash mismatch'
temp.replace(dest)
print('LOCAL_WIKI_OFFICIAL_CONTENT_HASH_VERIFIED',h.hexdigest(),flush=True)
