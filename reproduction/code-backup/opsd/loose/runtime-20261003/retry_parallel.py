"""Reuse reviewed downloader validation; resume large immutable files with 8 HTTP streams."""
import json,os,re,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['SEARCH_R1_REPO']+'/reproduction/h20')
import download_assets as d
original=d.subprocess.run
def run(command,**kwargs):
    if command[0]=='curl':
        dest=command[command.index('-o')+1];url=command[-1]
        command=['aria2c','--continue=true','--file-allocation=none','--max-connection-per-server=8','--split=8','--min-split-size=64M','--connect-timeout=10','--timeout=60','--lowest-speed-limit=1024','--max-tries=3','--retry-wait=3','--summary-interval=60','--console-log-level=warn','--auto-file-renaming=false','--dir='+str(Path(dest).parent),'--out='+Path(dest).name,url]
    r=original(command,**kwargs)
    if r.returncode: print('TRANSFER_DIAGNOSTIC',re.sub(r'https?://[^\s\]\)]+','[source URL]',(r.stdout+'\n'+r.stderr)[-4000:]),flush=True)
    return r
d.subprocess.run=run
entry=next(a for a in json.loads((d.BASE/'assets.json').read_text()) if a['path']==sys.argv[1])
for attempt in range(8):
    path=d.BASE/'assets'/entry['path'];partial=path.with_name(path.name+'.partial')
    before=partial.stat().st_blocks*512 if partial.exists() else 0
    if d.fetch(entry): break
    after=partial.stat().st_blocks*512 if partial.exists() else 0
    if after<=before: raise RuntimeError('No progress; retain partial and use verified cache relay')
    print('RESUME_VALIDATED_PARTIAL_PROGRESS',entry['path'],after-before,flush=True)
else: raise RuntimeError('Bounded transfer attempts exhausted; use verified cache relay')
