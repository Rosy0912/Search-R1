import pathlib,json,time,os,re,subprocess
b=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl');t=b/'tree-grpo-retained18-20261008';op=b/'opsd-e1-unclipped-multitraj200-20261006';o={'epoch':time.time(),'df':subprocess.check_output(['df','-B1','/mnt/data'],text=True),'candidate':[],'references':[]}
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(t) in cmd or str(op) in cmd:o['references'].append({'pid':int(proc.name),'cmd':cmd[:2000]})
 except:pass
for run in (t/'runs').iterdir():
 if run.name.startswith('tree-current200-continue-'):
  for cat in ['actor','training_state']:
   d=run/'checkpoints'/cat
   if d.exists():
    for c in d.iterdir():
     if run.name.endswith('3341') and (c.name=='global_step_200' or c.name=='rolling-completed'):continue
     o['candidate'].append(str(c))
  for cat in ['trajectories','tree-audit']:
   d=run/'output'/cat
   if d.exists():o['candidate'].append(str(d))
for c in (op/'runs/formal-3304/checkpoints/actor').iterdir():
 if c.name not in ['global_step_200','global_step_150']:o['candidate'].append(str(c))
cur=t/'runs/tree-branpo-fresh200-3360'
for cat in ['actor','training_state']:
 d=cur/'checkpoints'/cat
 for c in d.iterdir():
  if c.name in ['global_step_5','global_step_50']:o['candidate'].append(str(c))
for p in [cur,op/'runs/formal-3304']:
 o[str(p)]=subprocess.check_output(['find',str(p/'checkpoints'),'-maxdepth','4','-type','f','-name','*.json'],text=True)[:20000]
print(json.dumps(o))
