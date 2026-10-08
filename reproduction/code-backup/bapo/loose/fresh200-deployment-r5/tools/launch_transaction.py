"""Durable single-allocation admission. Unknown outcomes never authorize a retry."""
import fcntl, json, os, re, subprocess, sys, time, uuid
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006')
PHASES=('diagnostic-discovery128','diagnostic-discovery256','diagnostic-confirmation128','diagnostic-confirmation256','formal200')
def require(ok, message):
 if not ok: raise RuntimeError(message)
def durable(path, value):
 path=Path(path); require(not path.exists(), 'record already exists: '+str(path))
 data=(json.dumps(value,sort_keys=True)+'\n').encode()
 tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
 fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 try:
  with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
  os.link(tmp,path); tmp.unlink()
  d=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try: os.fsync(d)
  finally: os.close(d)
 finally:
  if tmp.exists(): tmp.unlink()
def identity(intent, job):
 require(str(job.get('JobId'))==str(intent.get('job_id')), 'job ID mismatch')
 require(job.get('Comment')==intent['comment'], 'scheduler nonce/source/phase comment mismatch')
 require(job.get('WorkDir')==str(ROOT), 'job root mismatch')
 require(job.get('JobName')=='bapo-fresh200-'+intent['phase'], 'job phase/name mismatch')
 require(job.get('UserId','').split('(')[0]==intent['user'], 'job owner mismatch')
 require(job.get('ReqNodeList')=='h100-5', 'requested node scope mismatch')
 count=job.get('NumNodes');state=job.get('JobState')
 require(count=='1' or (count=='1-1' and state=='PENDING'), 'node count scope mismatch')
 if state=='RUNNING': require(job.get('NodeList')=='h100-5', 'allocated node scope mismatch')
 require(job.get('TresPerNode') in ('gres:gpu:h100:8','gres/gpu:h100=8'), 'GPU scope mismatch')
 require(job.get('Partition')=='gpu' and job.get('OverSubscribe')=='NO', 'exclusive partition scope mismatch')
 require(Path(job.get('Command',''))==ROOT/'tools'/(intent['phase']+'.sbatch'), 'job command mismatch')
def scheduler(job_id):
 require(re.fullmatch(r'[1-9][0-9]*',str(job_id)) is not None,'invalid job ID')
 raw=subprocess.check_output(['scontrol','show','job','-o',str(job_id)],text=True)
 return dict(re.findall(r'(\w+)=([^\s]+)',raw))
def source():
 value=(ROOT/'source-expected.txt').read_text().strip()
 require(re.fullmatch(r'[0-9a-f]{40}',value) is not None,'unfixed source')
 return value
def unresolved_barrier(d, allow=None):
 for item in d.glob('*-submit-intent.json'):
  receipt=item.with_name(item.name.replace('-submit-intent.json','-receipt.json'))
  intent=json.loads(item.read_text())
  if allow is not None and intent==allow and not receipt.exists(): continue
  require(receipt.exists(),'unresolved intent across phases: '+str(item))
  rec=json.loads(receipt.read_text())
  require(all(rec.get(k)==v for k,v in intent.items()),'receipt binding mismatch')
def validate_intent(value):
 require(set(value)=={'phase','source','nonce','user','time','comment'},'intent schema mismatch')
 require(value['phase'] in PHASES and value['source']==source(),'intent phase/source mismatch')
 require(isinstance(value['nonce'],str) and re.fullmatch(r'[0-9a-f]{32}',value['nonce']) is not None,'invalid nonce')
 require(value['user']==__import__('getpass').getuser(),'intent owner mismatch')
 require(type(value['time']) in (int,float) and 0<value['time']<float('inf'),'invalid intent timestamp')
 require(value['comment']=='BAPO_FRESH200:'+value['phase']+':'+value['source']+':'+value['nonce'],'intent comment mismatch')
def admit(kind, raw):
 require(kind in ('intent','receipt'),'unknown mirror kind')
 value=json.loads(raw); intent={k:v for k,v in value.items() if k!='job_id'}
 validate_intent(intent)
 require(raw==(json.dumps(value,sort_keys=True)+'\n').encode(),'mirror bytes must be exact canonical record')
 d=ROOT/'scheduler-head'
 # Distinct from head lock: head/worker need not share their dataset filesystem.
 with (d/'worker-admission.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  unresolved_barrier(d,allow=intent)
  target=d/(intent['phase']+('-submit-intent.json' if kind=='intent' else '-receipt.json'))
  if kind=='receipt':
   require(set(value)==set(intent)|{'job_id'},'receipt schema mismatch')
   existing=d/(intent['phase']+'-submit-intent.json')
   require(existing.exists() and json.loads(existing.read_text())==intent,'mirror receipt without exact admitted intent')
   identity(value,scheduler(value['job_id']))
  if target.exists():
   require(target.read_bytes()==raw,'conflicting existing worker mirror')
  else: durable(target,value)
def prepare(phase):

 require(phase in PHASES,'unknown phase'); d=ROOT/'scheduler-head'; src=source()
 unresolved_barrier(d)
 target=d/(phase+'-receipt.json')
 if target.exists():
  rec=json.loads(target.read_text()); require(rec['source']==src,'existing receipt source mismatch')
  identity(rec,scheduler(rec['job_id'])); print('EXISTING '+rec['job_id']); return
 nonce=uuid.uuid4().hex
 value=dict(phase=phase,source=src,nonce=nonce,user=__import__('getpass').getuser(),time=time.time(),comment='BAPO_FRESH200:'+phase+':'+src+':'+nonce)
 durable(d/(phase+'-submit-intent.json'),value); print(value['comment'])
def record(phase,job_id):
 d=ROOT/'scheduler-head'; value=json.loads((d/(phase+'-submit-intent.json')).read_text())
 value['job_id']=job_id; identity(value,scheduler(job_id)); durable(d/(phase+'-receipt.json'),value)
def worker(phase,job_id):
 require(os.environ.get('SLURM_RESTART_COUNT','0')=='0','restarted allocation forbidden')
 require(phase in PHASES,'unknown phase')
 d=ROOT/'scheduler-head'; value=json.loads((d/(phase+'-submit-intent.json')).read_text())
 validate_intent(value); unresolved_barrier(d,allow=value)
 value['job_id']=job_id; require(value['source']==source(),'worker source mismatch'); job=scheduler(job_id); identity(value,job)
 require(job.get('JobState')=='RUNNING' and job.get('NodeList')=='h100-5' and job.get('NumNodes')=='1','worker requires exact running allocation')
 run=ROOT/'runs'/(phase+'-'+value['nonce']+'-'+job_id)
 # O_EXCL intent precedes all runtime side effects; retained on every failure.
 value.update(run=str(run),pid=os.getpid(),started=time.time(),host=os.uname().nodename)
 durable(d/(phase+'-worker-intent.json'),value)
 run.mkdir(mode=0o700); durable(run/'worker-start.json',value)
 print(run)
if __name__=='__main__':
 require(sys.flags.optimize==0,'optimized Python is forbidden')
 action=sys.argv[1]
 if action=='prepare': prepare(sys.argv[2])
 elif action=='record': record(sys.argv[2],sys.argv[3])
 elif action=='worker': worker(sys.argv[2],sys.argv[3])
 elif action=='admit-intent': admit('intent',sys.stdin.buffer.read())
 elif action=='admit-receipt': admit('receipt',sys.stdin.buffer.read())
 else: raise RuntimeError('unknown action')
