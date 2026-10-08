import subprocess,shlex,json
from pathlib import Path
code=r'''
from pathlib import Path
import json,re,datetime
p=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/runs/opsd-new-acceptance-2994')
f=p/'training.log'; raw=f.read_text(errors='replace') if f.exists() else ''
clean=re.sub(r'\x1b\[[0-9;]*m','',raw)
metrics=[l[l.index('step:'):] for l in clean.splitlines() if re.search(r'step:\d+\s+-\s',l)]
e=p/'evidence'
optional={n:(e/n).read_text() for n in ['micro-capacity.json','actual-fresh5-audit.json'] if (e/n).exists()}
resource={}
r=e/'resources.ndjson'
if r.exists():
 with r.open('rb') as h:
  h.seek(max(0,r.stat().st_size-300000));tail=h.read().decode()
 for line in reversed(tail.splitlines()):
  try:resource=json.loads(line);break
  except json.JSONDecodeError:continue
 resource={k:resource.get(k) for k in ['time','job_id','gpus','query_error']}

print(json.dumps(dict(sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='2994',metric_lines=metrics,training_tail=clean[-1800:],training_finished=(p/'training-finished.txt').exists(),exit=(p/'exit.txt').read_text() if (p/'exit.txt').exists() else None,optional_evidence=optional,resource=resource)))
'''
p=subprocess.run(['python3','work/remote.py','python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True)
x=json.loads(p.stdout);Path('work/job2994-progress.json').write_text(json.dumps(x,indent=2));print('metric_steps',[l[:20] for l in x['metric_lines']]);print(x['training_tail']);print('training_finished',x['training_finished'],'exit',x['exit'])
if 'micro-capacity.json' in x['optional_evidence']:
 c=json.loads(x['optional_evidence']['micro-capacity.json']);Path('work/job2994-micro-capacity.json').write_text(json.dumps(c,indent=2));print('CAPACITY',c['passed'],c['max_safe_local_micro'])

print('GPU',x.get('resource',{}).get('gpus'))
