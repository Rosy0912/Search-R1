"""Read and rehash completed training trajectory artifacts on h100-9, no GPU."""
import gzip,hashlib,json,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output/trajectories')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
result=[]
for d in sorted(root.glob('outer-00010[1-5]')):
 manifest=d/'manifest.json'
 if not manifest.exists():continue
 m=json.loads(manifest.read_bytes());assert m['identity']['job_id']=='3234'
 assert m['identity']['source']=='502e9fafa8e3aba722dbfcb493abc2b9d3ece326' and m['rows']==3840
 payloads={}
 for name,expected in m['payloads'].items():
  p=d/name;actual=sha(p);assert actual==expected
  payloads[name]=dict(sha256=actual,bytes=p.stat().st_size)
 rows=0;reward_sum=0.;metadata_keys=None
 with gzip.open(d/'trajectories.jsonl.gz','rt') as stream:
  for line in stream:
   x=json.loads(line);assert x['row']==rows and x['identity']==m['identity'];rows+=1;reward_sum+=x['reward'];assert 'token_ids' in x and 'response_token_ids' in x and 'metadata' in x
   if metadata_keys is None:metadata_keys=sorted(x['metadata'])
 assert rows==3840
 result.append(dict(identity=m['identity'],payloads=payloads,actual_rows=rows,mean_recorded_reward=reward_sum/rows,metadata_keys=metadata_keys,scope='Both payload raw SHA and all gzip rows inspected; tensor payload not loaded; Tree inheritance requires separate audit'))
print(json.dumps(dict(epoch=time.time(),records=result)),flush=True)
