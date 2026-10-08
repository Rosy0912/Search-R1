from pathlib import Path
import os,sys,json
root=Path('/dataset/shaoshitong/search-r1-rl/bapo')
os.environ['SEARCH_R1_ASSETS']=str(root/'assets-root')
sys.path.insert(0,str(root/'repo/reproduction/h20'))
from download_assets import BASE,valid
manifest=json.loads((BASE/'assets.json').read_text())
for a in manifest:
 if a['path'] not in {'wiki/part_aa','wiki/part_ab','wiki/wiki-18.jsonl.gz'}: continue
 dest=BASE/'assets'/a['path']; incoming=dest.parent/'cluster-incoming'/dest.name
 assert valid(incoming,a), a['path']
 incoming.replace(dest)
 print('CLUSTER_COPY_OFFICIAL_SHA_VERIFIED',a['path'],flush=True)
