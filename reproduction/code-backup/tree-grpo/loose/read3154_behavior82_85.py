import json,re,time,hashlib
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154/stage.log')
rows=[]
for raw in p.open('rb'):
 if not raw.endswith(b'\n'):break
 line=raw.decode(errors='replace');m=re.search(r'(?<![\w/])step:(\d+) - ',line)
 if not m or not 82<=int(m[1])<=85:continue
 fields={}
 for field in line[m.end():].split(' - '):
  k,sep,v=field.partition(':')
  try:fields[k]=float(v)
  except ValueError:pass
 if 'training/actual_outer_updates' not in fields:continue
 rows.append({'step':int(m[1]),'line_sha256':hashlib.sha256(raw).hexdigest(),'fields':fields})
print(json.dumps({'epoch':time.time(),'scope':'Logged training-batch aggregate metrics; different question batches, not fixed dev comparison or raw answer audit','rows':rows}))
