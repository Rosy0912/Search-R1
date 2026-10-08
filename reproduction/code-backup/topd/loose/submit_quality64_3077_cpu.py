import subprocess,pathlib
remote=r"""python3 - <<'INNER'
import pathlib,json,hashlib,subprocess,fcntl,time,os
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');ev=r/'logs/qwen25-32b-terminal-fixed413-eval64-3077/evaluation';d=json.loads((ev/'result.json').read_text());assert d['status']=='COMPLETE64_UNREVIEWED_NO_TRAINING'
assert len((ev/'qa.jsonl').read_text().splitlines())==64 and len(json.loads((ev/'questions.json').read_text()))==64
for name,key in [('final.safetensors','final_sha256'),('qa.jsonl','qa_sha256')]:assert hashlib.sha256((ev/name).read_bytes()).hexdigest()==d[key]
entries=[('source','quality64-source-recompute.sbatch','37f8c8ce0038119ffab800b3be49b9263a44e530355de57e6bbd00313e5491d2',[str(ev),str(r/'logs/quality64-3077-source-recomputed.json')]),('independent','review-eval64-raw-native.sbatch','7c26ee7080275ad370a80d3e11f3eebd10aa62deded9e853449830ab5c403326',[str(ev),str(r/'logs/qwen25-32b-terminal-fixed413-binding.json'),str(r/'logs/quality64-3077-independent-raw.json')])]
records=[]
for tag,name,sha,args in entries:
 entry=r/'tools'/name;assert hashlib.sha256(entry.read_bytes()).hexdigest()==sha
 record=r/'logs'/('quality64-3077-'+tag+'-cpu-submission.json')
 def persist(obj):
  tmp=record.with_suffix('.tmp')
  with tmp.open('w') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
  os.replace(tmp,record);fd=os.open(str(record.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
 with record.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  if record.exists():
   x=json.loads(record.read_text());assert x['status']=='SUBMITTED',x
  else:
   x=dict(status='UNKNOWN_BEFORE_SUBMISSION',scope='COMPLETE64_CPU_RECOMPUTATION_ONLY_NO_GPU_NO_TRAINING',evaluation_job=3077,entry=str(entry),entry_sha256=sha,args=args,epoch=time.time());persist(x)
   res=subprocess.run(['sbatch','--parsable',str(entry)]+args,capture_output=True,text=True);x.update(returncode=res.returncode,stdout=res.stdout,stderr=res.stderr);persist(x);assert res.returncode==0,x
   jid=res.stdout.strip().split(';')[0];assert jid.isdigit(),x;x.update(status='SUBMITTED',job_id=int(jid));persist(x)
   ctl=subprocess.run(['scontrol','show','job','-dd',jid],capture_output=True,text=True);(r/'logs'/('quality64-3077-'+tag+'-'+jid+'-initial-scontrol.txt')).write_text(ctl.stdout+ctl.stderr)
 records.append(x)
print(json.dumps(records))
INNER"""
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True);print(p.stdout,end='');print(p.stderr,end='')
if not p.returncode:pathlib.Path('outputs/quality64-3077-cpu-submissions.json').write_text(p.stdout)
raise SystemExit(p.returncode)
