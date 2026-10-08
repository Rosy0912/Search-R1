import json,time,hashlib
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154/output')
reports=[];ledgers=[]
for p in (r/'validation-cost').glob('*.json'):
 d=json.loads(p.read_bytes())
 if d.get('global_step')==100:reports.append({'path':str(p),'report':d})
for p in (r/'validation-cost').glob('*.jsonl'):
 rows=[];h=hashlib.sha256()
 for b in p.open('rb'):
  if not b.endswith(b'\n'):break
  h.update(b);rows.append(json.loads(b))
 if rows and rows[0].get('global_step')==100:
  ledgers.append({'path':str(p),'prefix_sha256':h.hexdigest(),'events':len(rows),'real_rows_submitted':sum(x.get('rows',0) for x in rows if x.get('kind')=='real_sample_batch'),'last_event':rows[-1],'failures':[x for x in rows if str(x.get('kind','')).endswith('_failed')]})
p=r/'checkpoints/tree-instruct300/training_state/global_step_100/manifest.json'
print(json.dumps({'epoch':time.time(),'reports':reports,'ledgers':ledgers,'state100_manifest_exists':p.is_file(),'scope':'Read-only completed reports and complete-line ledger prefix; submitted rows are not completed scored rows, checkpoint manifest existence alone is not acceptance'}))
