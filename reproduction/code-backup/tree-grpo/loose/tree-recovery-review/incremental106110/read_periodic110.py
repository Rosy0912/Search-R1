from pathlib import Path
import json,hashlib,time
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');files={}
for p in (root/'output/validation-cost').glob('*.json'):
 raw=p.read_bytes();x=json.loads(raw)
 if x.get('global_step')==110:
  for f in [p,p.with_suffix('.jsonl')]:
   data=f.read_bytes();files[str(f)]=dict(sha256=hashlib.sha256(data).hexdigest(),text=data.decode(),bytes=len(data))
for p in [root/'output/checkpoints/tree-recovery300/training_state/global_step_110/manifest.json',root/'output/entry-audit/entry.jsonl',root/'output/entry-audit/entry-config.yaml',root/'resolved.yaml']:
 if p.exists():
  raw=p.read_bytes();files[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode(),bytes=len(raw))
log=(root/'stage.log').read_bytes();lines=[s for s in log.decode().splitlines() if 'step:110 ' in s]
print(json.dumps(dict(epoch=time.time(),files=files,step110_console_lines=lines,stage_log_sha256=hashlib.sha256(log).hexdigest())),flush=True)
