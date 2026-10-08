"""Bounded same-call scontrol capture; real terminal only, never synthetic approval."""
import hashlib,json,os,pathlib,re,subprocess,time,types
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def atomic(p,r):
 p=pathlib.Path(p);tmp=p.with_name('.'+p.name+'.'+str(time.time_ns()))
 with open(tmp,'x') as f:json.dump(r,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.link(tmp,p);tmp.unlink();fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def parse(job,r):
 text=r.stdout;fields=dict(re.findall(r'(\w+)=([^\s]+)',text))
 same=r.returncode==0 and fields.get('JobId')==str(job) and fields.get('NodeList')=='h100-11'
 terminal=same and fields.get('JobState') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','PREEMPTED','OUT_OF_MEMORY','BOOT_FAIL','DEADLINE')
 complete=terminal and fields.get('JobState')=='COMPLETED' and fields.get('ExitCode')=='0:0'
 return {'same_job_node':same,'terminal_observed':terminal,'completed_zero':complete,'state':fields.get('JobState'),'exit_code':fields.get('ExitCode')}
def observe(job,helper,*,runner=subprocess.run,clock=time.monotonic,pause=time.sleep,budget=60):
 if not 0<budget<=60:raise ValueError('Bounded native observer maximum60s')
 directory=ROOT/'logs'/('g1-v13-known-cpu-observation-'+str(job));directory.mkdir(exist_ok=True)
 final=directory/'actual-terminal.json'
 if final.exists():
  r=json.loads(final.read_text());raw=r['raw']
  if r['job_id']!=job or sha(raw['path'])!=raw['sha256']:raise ValueError('Saved native terminal identity/bytes drift')
  actual=json.loads(pathlib.Path(raw['path']).read_text());parsed=parse(job,types.SimpleNamespace(returncode=actual['scontrol_returncode'],stdout=actual['scontrol_stdout'],stderr=actual['scontrol_stderr']))
  if not parsed['terminal_observed'] or any(r.get(k)!=v for k,v in parsed.items()):raise ValueError('Saved terminal raw does not prove recorded state')
  return r
 deadline=clock()+budget;records=[];last=None;failure=None
 while clock()<deadline:
  remaining=deadline-clock()
  try:r=runner(['scontrol','show','job','-dd',str(job)],text=True,capture_output=True,timeout=min(5,remaining))
  except Exception as exc:
   raw={'job_id':job,'captured_epoch':time.time(),'error_type':type(exc).__name__,'error':str(exc),'final_state_observed':False};parsed={'terminal_observed':False,'completed_zero':False};failure=str(exc)
  else:
   parsed=parse(job,r);raw={'job_id':job,'captured_epoch':time.time(),'scontrol_returncode':r.returncode,'scontrol_stdout':r.stdout,'scontrol_stderr':r.stderr,'final_state_observed':parsed['terminal_observed'],'completed_zero':parsed['completed_zero']}
  raw_path=directory/('scontrol-'+str(time.time_ns())+'.json');atomic(raw_path,raw);last={'path':str(raw_path),'sha256':sha(raw_path)};records.append(last)
  if parsed['terminal_observed']:
   result=dict(scope='ACTUAL_NATIVE_CPU_TERMINAL_NOT_GPU_APPROVAL',job_id=job,raw=last,observations=records,**parsed)
   # COMPLETE0 still needs actual exact native logs/source receipts; invoking the
   # frozen capture now avoids the Slurm terminal TTL after this call returns.
   if parsed['completed_zero']:
    remaining=deadline-clock()
    if remaining>0:
     try:
      c=runner([str(ROOT/'env/bin/python'),str(pathlib.Path(helper)/'capture_cpu_binding.py'),'--output',str(ROOT/'logs'/('g1-v13-known-cpu-'+str(job)))],text=True,capture_output=True,timeout=min(5,remaining))
      result['binding_capture']={'returncode':c.returncode,'stdout':c.stdout,'stderr':c.stderr}
     except Exception as exc:result['binding_capture']={'error_type':type(exc).__name__,'error':str(exc)}
   atomic(final,result);return result
  remaining=deadline-clock()
  if remaining<=0:break
  pause(min(2,remaining))
 return {'scope':'ACTUAL_NATIVE_CPU_OBSERVATION_PENDING_NOT_PROOF','job_id':job,'terminal_observed':False,'completed_zero':False,'raw':last,'observations':records,'last_query_error':failure}
