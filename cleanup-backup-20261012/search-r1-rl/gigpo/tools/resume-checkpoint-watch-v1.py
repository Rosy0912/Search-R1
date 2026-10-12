"""Observe only the existing resume job; never submit or change training."""
from pathlib import Path
import json,time,subprocess,datetime,sys
r=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo/runs/gigpo-resume200to300-job3130-1791086878335123421');target=int(sys.argv[1]);assert target in (250,275,300)
last=None;deadline=time.monotonic()+7200
while time.monotonic()<deadline:
 counts={}
 for rank in range(8):
  rows=[]
  for line in (r/'audit'/f'{rank}.jsonl').read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
  counts[rank]=rows[-1]['outer_step']
 state=subprocess.check_output(['squeue','-h','-j','3130','-o','%T'],text=True).strip();cp=r/'checkpoints/actor'/f'global_step_{target}'/'training_state/manifest.json'
 result=None
 if (r/'run-result.json').exists():
  try:result=json.loads((r/'run-result.json').read_text())
  except json.JSONDecodeError:pass
 snapshot=dict(time=datetime.datetime.now().isoformat(),job='3130',state=state,logical_step=min(counts.values()),new_updates=min(counts.values())-200,rank_steps=counts,checkpoint_target=target,checkpoint_manifest=cp.exists(),result_status=None if result is None else result['status'])
 if snapshot['logical_step']!=last or cp.exists() or result is not None or state!='RUNNING':print(json.dumps(snapshot),flush=True);last=snapshot['logical_step']
 if cp.exists() or result is not None or state!='RUNNING':break
 time.sleep(45)
