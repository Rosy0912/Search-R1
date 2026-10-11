from pathlib import Path
import json,time
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');run=Path(json.loads((r/'scheduler-head/eval150-submit-receipt.json').read_text())['run']);rows=[]
p=run/'qa-rows.jsonl'
if p.exists():
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except ValueError:pass
watch=json.loads((run/'resource-watch.jsonl').read_text().splitlines()[-1]);done=json.loads((run/'progress.jsonl').read_text().splitlines()[-1]) if(run/'progress.jsonl').exists() else None
print(json.dumps(dict(time=time.time(),run=str(run),progress=done,persisted_qa_rows=len(rows),partial_em=sum(x['em'] for x in rows)/max(1,len(rows)),partial_f1=sum(x['f1'] for x in rows)/max(1,len(rows)),resource=watch)))
