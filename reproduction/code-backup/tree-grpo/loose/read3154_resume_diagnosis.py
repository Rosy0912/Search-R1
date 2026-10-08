import hashlib,json,re,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154')
raw=(root/'stage.log').read_bytes()
rows=[]
for line in raw.decode(errors='replace').splitlines():
    m=re.search(r'(?<![\w/])step:(\d+) - ',line)
    if not m: continue
    d={}
    for field in line[m.end():].split(' - '):
        k,_,v=field.partition(':')
        try:d[k]=float(v)
        except ValueError:pass
    if 'training/actual_outer_updates' not in d:continue
    rows.append({'step':int(m[1]),'metrics':d})
scores=[r for r in rows if 'val/f1/all' in r['metrics']]
best=max(scores,key=lambda r:(r['metrics']['val/f1/all'],r['step']))
directory=root/'output/checkpoints/tree-instruct300'
state=directory/'training_state'/f'global_step_{best["step"]}'/'manifest.json'
print(json.dumps(dict(epoch=time.time(),log_sha256=hashlib.sha256(raw).hexdigest(),actual=len(rows),
 scores=[dict(step=r['step'],f1=r['metrics']['val/f1/all'],em=r['metrics']['val/em/all'],reward=r['metrics']['critic/rewards/mean']) for r in scores],
 best_evaluated_checkpoint=best['step'],best_state_manifest=json.loads(state.read_text()),
 collapse_first_zero_reward=next((r['step'] for r in rows if r['metrics']['critic/rewards/mean']==0),None),
 tail=rows[-3:],logged_metrics=rows),allow_nan=False))
