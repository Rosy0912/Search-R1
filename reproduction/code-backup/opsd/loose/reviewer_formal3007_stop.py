import subprocess,shlex,json,sys,re,hashlib,datetime
from pathlib import Path
root='/mnt/data/shaoshitong/search-r1-rl/opsd'
bundle='formal-8e60976d364a-pilot-2994-hf-2996-reviewed-c85d38bd29f6'
code=f'''from pathlib import Path
import json,sys,hashlib,re,datetime,subprocess
root=Path({root!r});run=root/'runs/opsd-formal-3007'
sys.path.insert(0,str(root/'tools'/{bundle!r}))
from verify_formal_prerequisites import verify_outers
b=(run/'training.log').read_bytes();s=b.decode()
steps=[int(m.group(1)) for line in s.splitlines() if 'actor/optimizer_steps:' in line for m in [re.search(r'(?:^|\\s)step:(\\d+) - ',line)] if m]
v=verify_outers(s,steps[-1])
ckpt=run/'output/checkpoints/opsd-gpu-formal500/actor'
checkpoints=[]
for p in sorted(ckpt.glob('global_step_*')):
 index=p/'model.safetensors.index.json'
 checkpoints.append(dict(path=str(p),files=[dict(name=q.name,bytes=q.stat().st_size) for q in sorted(p.iterdir()) if q.is_file()],index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None,index_keys=len(json.loads(index.read_text())['weight_map']) if index.exists() else None))
markers={{str(p.relative_to(run)):p.read_text() for p in run.glob('*.txt') if p.stat().st_size<5000}}
procs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  c=(p/'cgroup').read_text()
  if re.search(r'job_3007(?:/|$)',c):procs.append(dict(pid=int(p.name),cgroup=c,comm=(p/'comm').read_text().strip()))
 except (PermissionError,FileNotFoundError,ProcessLookupError):pass
print(json.dumps(dict(sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='3007',source_commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),original_log_bytes=len(b),original_log_sha256=hashlib.sha256(b).hexdigest(),training_log=s,native_numeric_result=v,checkpoints=checkpoints,run_text_markers=markers,finished_marker_present='FORMAL_500_OUTERS_AND_FINAL_EVALUATION_EXITED' in s,final51713_marker_present='SEARCH_R1_FINAL_METRICS_JSON=' in s,current_job_cgroup_processes=procs)))
'''
r=subprocess.run(['python3','work/remote.py','env CUDA_VISIBLE_DEVICES= '+root+'/env/bin/python -c '+shlex.quote(code)],capture_output=True,text=True)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
x=json.loads(r.stdout);Path('work/job3007-stop-independent-raw.json').write_text(r.stdout)
sched=subprocess.run(['python3','work/head.py','scontrol show job 3007'],capture_output=True,text=True)
if sched.returncode:print(sched.stderr);raise SystemExit(sched.returncode)
Path('work/job3007-stop-independent-scontrol.txt').write_text(sched.stdout)
s=sched.stdout
assert re.search(r'\bJobId=3007\b',s)
assert re.findall(r'\bJobState=([^\s]+)',s)==['CANCELLED']
assert re.findall(r'\bExitCode=([^\s]+)',s)==['0:15']
assert 'NodeList=h100-10' in s and 'gres:gpu:h100:8' in s
m=re.search(r'RunTime=(?:(\d+)-)?(\d+):(\d+):(\d+)',s)
seconds=(int(m[1] or 0)*24+int(m[2]))*3600+int(m[3])*60+int(m[4])
x['actual_terminal_scontrol']=s;x['actual_terminal_sha256']=hashlib.sha256(s.encode()).hexdigest();x['runtime_seconds']=seconds;x['allocated_gpu_hours']=seconds*8/3600
sys.path.insert(0,str(Path('work/formal-preparation/hf500-formal3007-strict-two-raywarnings-preview').resolve()))
from periodic_validation import verify_periodic_validations
x['historical_periodic_pair_result']=verify_periodic_validations(x['training_log'],expected_steps=(0,50),require_final=False)
x['formal_500_completed']=False;x['corrected_efficacy_completed']=False
Path('work/job3007-stop-independent-analysis.json').write_text(json.dumps({k:v for k,v in x.items() if k!='training_log'},indent=2))
print(json.dumps({k:x[k] for k in ['sample_utc','original_log_bytes','original_log_sha256','native_numeric_result','runtime_seconds','allocated_gpu_hours','checkpoints','run_text_markers','current_job_cgroup_processes','historical_periodic_pair_result']},indent=2))
