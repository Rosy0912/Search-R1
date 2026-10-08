import concurrent.futures,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo')
os.environ['SEARCH_R1_ASSETS']=str(ROOT/'assets-root')
sys.path.insert(0,str(ROOT/'repo/reproduction/h20'))
from download_assets import BASE,valid
assets=json.loads((BASE/'assets.json').read_text())
def fetch(a):
 dest=BASE/'assets'/a['path'];temp=dest.with_name(dest.name+'.partial')
 if valid(dest,a):print('EXISTS_VERIFIED',a['path'],flush=True);return True
 url='https://huggingface.co/datasets/'+a['repo']+'/resolve/'+a['revision']+'/'+a['file']
 print('CANONICAL_RESUME_START',a['path'],'retained_bytes',temp.stat().st_blocks*512 if temp.exists() else 0,flush=True)
 command=['aria2c','--continue=true','--file-allocation=none','--max-connection-per-server=16','--split=16','--min-split-size=64M','--connect-timeout=15','--timeout=60','--lowest-speed-limit=1024','--max-tries=5','--retry-wait=5','--summary-interval=60','--console-log-level=warn','--auto-file-renaming=false','--dir='+str(temp.parent),'--out='+temp.name,url]
 proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in proc.stdout:
  # Never retain redirect URLs or signed request fields in task evidence.
  if re.search(r'http|token|signature|credential|authorization',line,re.I):
   codes=re.findall(r'(?:errorCode|status)=(\d+)',line)
   if codes:print('TRANSFER_ERROR_CODES',a['path'],codes,flush=True)
   continue
  if line.startswith('[#') or 'Download Results' in line or re.search(r'errorCode=\d+',line):print(a['path'],line.strip(),flush=True)
 code=proc.wait()
 if code:print('CANONICAL_RESUME_FAILED',a['path'],code,flush=True);return False
 assert valid(temp,a),'Full canonical resumed asset SHA mismatch: '+a['path']
 temp.replace(dest);print('CANONICAL_ASSET_HASH_VERIFIED',a['path'],flush=True);return True
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 results=list(pool.map(fetch,[a for a in assets if a['path'] in {'wiki/part_aa','wiki/part_ab','wiki/wiki-18.jsonl.gz'}]))
(ROOT/'logs/canonical-index-status.json').write_text(json.dumps({'complete':all(results)},indent=2))
raise SystemExit(0 if all(results) else 1)
