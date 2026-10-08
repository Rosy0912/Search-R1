import json,sys,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric/runs/bapo-formal500-'+sys.argv[1])
result={'run':str(r),'time':time.time(),'success_marker':(r/'TRAIN_RETURNED_SUCCESSFULLY').exists()}
p=r/'train.log'; rows=[]
if p.exists():
 for line in p.read_text(errors='replace').splitlines():
  mark='BAPO_TRAIN_METRICS_JSON='
  if mark in line:
   try: rows.append(json.loads(line.split(mark,1)[1]))
   except json.JSONDecodeError: pass
 result['complete_outer_summaries']=[{'step':x['step'],'metrics':{k:v for k,v in x['metrics'].items() if k.startswith(('timing_s/','bapo/stage','bapo/actual_outer','bapo/grad','bapo/optimizer','bapo/mask','actor/grad','actor/pg_loss'))}} for x in rows[-3:]]
 steps=[x['step'] for x in rows]
 result['observed_steps']=steps
 result['observed_unique_steps']=len(set(steps))
 result['completed_outer_count']=len(rows) if steps==list(range(1,len(rows)+1)) else None
 result['contiguous_unique']=steps==list(range(1,len(rows)+1))
 result['train_tail'] = [line[-800:] for line in p.read_text(errors='replace').splitlines()[-3:]]
w=r/'resource-watch.jsonl'
if w.exists():
 lines=w.read_text().splitlines()
 result['watch_count']=len(lines)
 if lines:
  value=json.loads(lines[-1]);result['last_watch']={k:v for k,v in value.items() if k in ('time','pause','reason','event')}
print(json.dumps(result,ensure_ascii=False))
