import json,re,time
from pathlib import Path
import sys
run=Path('/dataset/shaoshitong/search-r1-rl/bapo/runs')/sys.argv[1]
print('STATUS_TIME',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
for name in ('train.log','retriever.log','driver-time.log'):
 p=run/name
 if not p.exists():continue
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-12000));s=f.read().decode(errors='replace')
 lines=s.replace('\r','\n').splitlines()
 print(name,'\n'.join(lines[-5:]))
p=run/'cpu-process-monitor.jsonl'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-100000));raw=f.read().splitlines()
 rows=[]
 for line in raw:
  try:rows.append(json.loads(line))
  except ValueError:pass
 if rows:
  d=rows[-1]; ps=d['processes']; print('RETRIEVER_RESOURCE_LAST',d['time'],[v for v in ps if v['role']=='retriever']);print('OWN_PROCESS_COUNT',len(ps))
p=run/'resource-watch.jsonl'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-60000));raw=f.read().splitlines()
 for line in reversed(raw):
  try:d=json.loads(line);break
  except ValueError:continue
 else:d={}
 print('RESOURCE_WATCH', {k:d.get(k) for k in ('time','pause','reason','exceeded')})
print('CLEANUP_EXISTS',(run/'own-process-cleanup.json').exists(),'SUCCESS',(run/'TRAIN_RETURNED_SUCCESSFULLY').exists())
