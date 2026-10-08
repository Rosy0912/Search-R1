import json,hashlib
from pathlib import Path
p=Path(__file__).parent;records=[]
for step in range(121,131):
 f=p/f'ROUND{step}_RECEIPT.json'
 if not f.exists():break
 prev=p/f'ROUND{step-1}_RECEIPT.json' if step>121 else p.parent/'incremental111120/ROUND120_RECEIPT.json'
 old=json.loads(prev.read_text());new=json.loads(f.read_text());om={x['path']:x['sha256'] for x in old['evidence']};nm={x['path']:x['sha256'] for x in new['evidence']}
 selected=[path for path in nm if path.endswith('.json') and '/tree-rank-audit/' in path and '/memory-profile/' not in path and '/ratio-tail/' not in path and f'step-{step-1:06d}-rank-' in path]
 assert len(selected)==8 and all(nm[path]==om[path] for path in selected)
 records.append(dict(step=step,previous_receipt_sha256=hashlib.sha256(prev.read_bytes()).hexdigest(),current_receipt_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),previous_original_eight_rank_bytes_unchanged=True))
print(json.dumps(dict(status='PASS_ACCEPTED_PREVIOUS_RAW_SHA_CHAIN_ONLY',records=records)))
