"""Full target-node hashes against the shared experiment's accepted Wiki/E5."""
import json,hashlib,time,os
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
assets=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets')
reference=json.loads((root/'tools/common-retrieval-identity.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 return h.hexdigest()
records=[]
for filename,key in [('wiki/e5_Flat.index','index'),('wiki/wiki-18.jsonl','corpus')]:
 p=assets/filename;r=reference[key]
 assert p.stat().st_size==r['size'] and digest(p)==r['sha256'],str(p)
 records.append(dict(path=str(p),sha256=r['sha256'],bytes=r['size']))
for name,expected in reference['model'].items():
 if '.partial' in name:continue
 p=assets/'e5-base-v2'/name
 assert digest(p)==expected,str(p)
 records.append(dict(path=str(p),sha256=expected,bytes=p.stat().st_size))
value=dict(status='PASS',node=os.uname().nodename,records=records,reference_identity=reference,time=time.time())
(root/'logs/retrieval-assets-full-sha-common-identity.json').write_text(json.dumps(value,indent=2))
print(json.dumps(dict(status='PASS',files=len(records),scope='Same Wiki2018/E5 actual full SHA as shared Tree common experiment')))
