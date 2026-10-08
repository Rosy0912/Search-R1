from pathlib import Path
import json,hashlib
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
p=R/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315/logs/first5-evidence-v4-20261007-a2/collection.json'
b=p.read_bytes();v=json.loads(b);assert v['source_commit']=='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'and v['job_id']==3315 and v['status']=='ACTUAL_FIRST5_COLLECTION_REQUIRES_INDEPENDENT_REVIEW'
print(json.dumps(dict(remote_path=str(p),sha256=hashlib.sha256(b).hexdigest(),collection=v)))
