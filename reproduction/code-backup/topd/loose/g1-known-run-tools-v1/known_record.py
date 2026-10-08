"""Single immutable recipe admission; UNKNOWN can never automatically resubmit."""
import fcntl,json,os,pathlib,time

def atomic(path,data):
 p=pathlib.Path(path);temp=p.with_name(p.name+'.private')
 with temp.open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
 os.replace(temp,p);fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def prior(path,binding):
 if not pathlib.Path(path).exists():return None
 value=json.loads(pathlib.Path(path).read_text())
 if value.get('status')!='SUBMITTED' or value.get('binding')!=binding:raise RuntimeError('Existing UNKNOWN/mismatched record: reconcile actual Slurm job; no retry')
 return value

def submit_once(path,binding,command,env,runner,precheck):
 path=pathlib.Path(path)
 with (path.parent/'g1-known-run-admission.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  existing=prior(path,binding)
  if existing:return existing
  observations=precheck()
  result={'status':'UNKNOWN_BEFORE_SUBMISSION','binding':binding,'epoch':time.time(),'pre_submit':observations};atomic(path,result)
  process=runner(command,env=env,text=True,capture_output=True)
  result.update(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr);atomic(path,result)
  number=process.stdout.strip().split(';')[0]
  if process.returncode!=0 or not number.isdigit():raise RuntimeError('Submission result UNKNOWN; inspect record and Slurm, never retry automatically')
  result.update(status='SUBMITTED',job_id=int(number));atomic(path,result);return result
