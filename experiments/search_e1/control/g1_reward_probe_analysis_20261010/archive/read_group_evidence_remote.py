"""Read-only grouped-reward availability audit for original Archive4 steps0–150."""
import datetime,hashlib,json,os,re
from collections import Counter
from pathlib import Path
B=Path('/weka_3/shaoshitong/search-e1-config-study-20261004');D={'readonly':True,'new_eval':False,'raw_questions_or_rollouts_exported':False,'observed_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'run_stage':'archive/round-1-grpo','scope_iterations':[0,150],'roots':[],'source':[],'full_metric_key_union_1_150':[],'any_named_group_metrics':[]}
for rel in ['runs/archive/round-1-grpo','outputs/archive','rollouts/archive','outputs','rollouts']:
 top=B/rel;e={'path':str(top),'exists':top.exists(),'noncheckpoint_output_files':[]}
 if top.is_dir():
  for root,dirs,files in os.walk(top):
   dirs[:]=[z for z in dirs if z!='actor'and'global_step_'not in z and not z.startswith('.')]
   for f in files:
    p=Path(root)/f;st=p.stat()
    if f.startswith('._'):continue
    e['noncheckpoint_output_files'].append({'path':str(p),'bytes':st.st_size})
 D['roots'].append(e)
p=B/'runs/archive/round-1-grpo/train.log';D['log']={'path':str(p),'bytes':p.stat().st_size};h=hashlib.sha256();keys=set();D['scalar_count']=0;example_schema=set();matches=0
with p.open('rb') as f:
 for n,raw in enumerate(f,1):
  h.update(raw);s=re.sub(r'\x1b\[[0-9;]*m','',raw.decode(errors='replace'))
  m=re.search(r'\bstep:(\d+)\s*-',s)
  if not m or not 1<=int(m[1])<=150:continue
  values={a:float(v)for a,v in re.findall(r'([\w/]+):(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)',s)}
  if 'critic/rewards/mean'in values:D['scalar_count']+=1
  keys.update(values)
  group={k:v for k,v in values.items() if re.search(r'group|std|var|all.?correct|all.?wrong|mixed',k,re.I)}
  if group:D['any_named_group_metrics'].append({'iteration':int(m[1]),'line':n,'metrics':group})
D['log']['sha256']=h.hexdigest();D['full_metric_key_union_1_150']=sorted(keys)
for rel in ['runtime/archive/upstream/Search-R1/verl/trainer/ppo/core_algos.py','runtime/archive/upstream/Search-R1/verl/trainer/main_ppo.py']:
 p=B/rel;raw=p.read_bytes();ls=raw.decode(errors='replace').splitlines();e={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'lines':[]}
 if p.name=='core_algos.py':
  start=next(i for i,s in enumerate(ls,1) if 'def compute_grpo_outcome_advantage' in s)
  e['lines']=[{'line':i,'text':ls[i-1]}for i in range(start,min(start+85,len(ls))+1)]
 else:e['lines']=[{'line':i,'text':ls[i-1]}for i in range(30,114)]
 D['source'].append(e)
print(json.dumps(D,ensure_ascii=False,indent=2))
