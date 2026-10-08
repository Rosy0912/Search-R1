import json,hashlib,time
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154/resource-samples.jsonl')
h=hashlib.sha256();count=0;peaks={};events=[];first=None;last=None;used_bytes=0
with p.open('rb') as stream:
 for line in stream:
  if not line.endswith(b'\n'):break
  row=json.loads(line);h.update(line);used_bytes+=len(line);count+=1
  if first is None:first=row['epoch']
  last=row['epoch']
  if row.get('event'):events.append({'epoch':last,'event':row['event']})
  for gpu in row.get('gpus',[]):
   key=gpu['uuid'];used=gpu['total_used_memory_bytes'];total=gpu['total_memory_bytes']
   if key not in peaks or used>peaks[key]['used_bytes']:
    peaks[key]={'used_bytes':used,'total_bytes':total,'fraction':used/total,'epoch':last,'gib':used/1024**3}
print(json.dumps({'status':'ACTUAL_RESOURCE_PREFIX_SAMPLED_NOT_FINAL','read_epoch':time.time(),'complete_sample_records':count,'first_epoch':first,'last_epoch':last,'observed_span_seconds':last-first if first is not None else None,'prefix_bytes':used_bytes,'prefix_sha256':h.hexdigest(),'whole_card_sampled_peak_by_uuid':peaks,'guard_events':events,'limits':'Includes retriever and foreign GPU usage; 5-second samples may miss spikes; not Torch peak, not process allocation, not whole-job final peak.'}))
