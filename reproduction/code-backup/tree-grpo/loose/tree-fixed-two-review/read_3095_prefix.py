import pathlib,json,hashlib,time,subprocess
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-fixed-two-3095';files={}
for relative in ['launch-manifest.json','resolved.yaml','run-result.json','cost.json','termination-cost.json','stage.log','guard.log','output/entry-audit/entry-config.yaml']:
 p=run/relative
 if p.is_file():
  data=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'text':data.decode()}
for p in list((run/'output/tree-rank-audit').rglob('*.json'))+[run/'output/entry-audit/entry.jsonl']:
 if p.is_file():
  data=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'text':data.decode()}
counts={}
for name in ['tree-rank-audit','tree-audit']:
 d=run/'output'/name;counts[name]=[str(p.relative_to(run)) for p in d.rglob('*.json') if p.is_file()] if d.exists() else []
guard=run/'resource-samples.jsonl';lines=guard.read_text().splitlines() if guard.exists() else []
slurm={}
for job in ['3095','3096','3097']:
 p=subprocess.run(['scontrol','show','job',job],capture_output=True,text=True);slurm[job]={'exit':p.returncode,'text':p.stdout+p.stderr}
print(json.dumps({'epoch':time.time(),'files':files,'counts':counts,'guard_count':len(lines),'guard_last':json.loads(lines[-1]) if lines else None,'guard_events':[json.loads(x) for x in lines if json.loads(x).get('event')],'slurm':slurm}))
