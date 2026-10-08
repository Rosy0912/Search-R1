import json,time,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154')
out={'epoch':time.time(),'slurm':subprocess.run(['scontrol','show','job','3154'],text=True,capture_output=True).__dict__.get('stdout'),'files':{}}
for name in ['stage.log','retriever.log','ray.log','run-result.json','guard.log','resource-samples.jsonl','output/acceptance-step5/ready.json','output/acceptance-step5/hf5-full-reload.json','output/acceptance-step5/state5-files-cpu.json']:
 p=r/name
 if p.exists():
  raw=p.read_bytes();out['files'][name]={'bytes':len(raw),'tail':raw[-8000:].decode(errors='replace')}
  if p.suffix=='.json':out['files'][name]['json']=json.loads(raw)

out['rank_records']=[]
for p in sorted((r/'output/tree-rank-audit').glob('*.json')):
    if 'profile' not in p.name: out['rank_records'].append(dict(path=str(p),record=json.loads(p.read_bytes())))
out['ranks']=len(out['rank_records'])

out['validation_files']=[str(p.relative_to(r)) for p in (r/'output/validation-cost').glob('*')]
out['validation_ledgers']=[]
for p in sorted((r/'output/validation-cost').glob('*.jsonl')):
    events=[]
    for line in p.read_text().splitlines():
        try: events.append(json.loads(line))
        except json.JSONDecodeError: pass
    out['validation_ledgers'].append(dict(path=str(p),last_event=events[-1] if events else None,
        started_real_question_rows=sum(e.get('rows',0) for e in events if e.get('kind')=='real_sample_batch'),
        generation_completed=sum(e.get('kind')=='generation_completed' for e in events),
        http_completed=sum(e.get('kind')=='http_completed' for e in events),
        failed_events=[e for e in events if e.get('kind') in ['generation_failed','http_failed']] ))

print(json.dumps(out))
