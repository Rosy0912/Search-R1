import json,os,re,sys
from pathlib import Path
sys.path.insert(0,os.environ['SEARCH_R1_REPO']+'/reproduction/h20')
import download_assets as d
original=d.subprocess.run
def run(command,**kwargs):
    if command[0]=='curl': command=command[:1]+['--http1.1']+command[1:]
    if command[0]=='aria2c': command=command[:1]+['--stop=600']+command[1:]
    r=original(command,**kwargs)
    if r.returncode:
        print('TRANSFER_DIAGNOSTIC',re.sub(r'https?://[^\s\]\)]+','[source URL]',(r.stdout+'\n'+r.stderr)[-8000:]),flush=True)
    return r
d.subprocess.run=run
manifest=json.loads((d.BASE/'assets.json').read_text())
assert d.fetch(next(a for a in manifest if a['path']==sys.argv[1]))
