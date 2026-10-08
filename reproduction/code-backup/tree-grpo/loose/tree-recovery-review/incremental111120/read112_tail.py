import json,hashlib,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');records=[]
for step in [111,112]:
 for p in sorted((root/'output/tree-rank-audit/ratio-tail').glob(f'step-{step:06d}-rank-*-optimizer-*.json')):
  raw=p.read_bytes();x=json.loads(raw);records.append(dict(step=step,path=str(p),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode()))
log=(root/'stage.log').read_bytes();lines=[s for s in log.decode().splitlines() if 'step:111 ' in s or 'step:112 ' in s]
print(json.dumps(dict(status='READONLY_RAW_TAIL111112',epoch=time.time(),records=records,console=lines,stage_sha256=hashlib.sha256(log).hexdigest())))
