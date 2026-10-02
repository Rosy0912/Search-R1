import json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['SEARCH_R1_REPO']+'/reproduction/h20')
import download_assets as d
original=d.subprocess.run
def run(command,**kwargs):
    if command[0]=='curl': command=command[:1]+['--http1.1']+command[1:]
    return original(command,**kwargs)
d.subprocess.run=run
manifest=json.loads((d.BASE/'assets.json').read_text())
entry=next(a for a in manifest if a['path']=='wiki/wiki-18.jsonl.gz')
assert d.fetch(entry), 'HTTP1.1 retry failed; use verified local relay'
