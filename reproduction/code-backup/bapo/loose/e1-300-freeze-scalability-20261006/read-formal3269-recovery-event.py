from pathlib import Path
import json,base64,hashlib,time
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006/runs/bapo-instruct300-formal300-3269');f={}
for n in ('resource-watch.jsonl','source-commit.txt','own-process-cleanup.json'):
 p=r/n
 if not p.exists():continue
 b=p.read_bytes();f[n]={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
t=(r/'train.log').read_text(errors='replace');metrics=[x for x in t.splitlines()if 'BAPO_TRAIN_METRICS_JSON='in x];f['metrics']={'lines':metrics,'train_tail':t[-12000:]}
print(json.dumps({'status':'ACTUAL_READONLY_PRODUCTION_RECOVERY_JOURNAL_SNAPSHOT','job_id':3269,'run_root':str(r),'time_unix':time.time(),'files':f}))
