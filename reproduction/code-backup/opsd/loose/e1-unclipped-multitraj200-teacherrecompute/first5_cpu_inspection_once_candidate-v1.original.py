"""Once-only actual3304 CPU inspector launch; no signing, release, GPU or retry."""
import argparse,datetime,hashlib,json,os,re,socket,stat,subprocess,sys,types
from pathlib import Path
SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
JOB='3304'
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
REPO=ROOT/('repo-'+SOURCE[:12]);RUN=ROOT/'runs'/('formal-'+JOB)
FIRST5=ROOT/'tools/first5-1b16c2227caf'
FIRST5_SHA='1b16c2227caf6aa6fae68d79dc02a5b997833e63f719ac393e759a7ca432a694'
INSPECT_SHA='211fd0fa3f30ba2a85aa2460ab34a2b9a34750463caa83cfc97353f1d57e52c6'
NATIVE_SHA='dc1905562f2f834dfcb7542f96d6d0c3cda4d3200c0a61a36ecb645320077af9'
PYTHON=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python')
HERE=Path(__file__).resolve()
OUTPUT=RUN/'evidence/first5-cpu-inspection-once-v1'
SCOPE='ACTUAL3304_CPU_FIRST5_ONCE_NOT_SIGNATURE_RELEASE_GPU_OR_EFFECT_ACCEPTANCE'
CONTRACT={'global_mini_batch':256,'optimizer_steps_per_rank_per_outer':10,'saved_adam_steps':50,'horizon':200,'warmup_steps':57,'base_learning_rate':1e-6}

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def unique(b):
 def pairs(xs):
  d={}
  for k,v in xs:need(k not in d,'Duplicate key');d[k]=v
  return d
 return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def pin(x):
 need(type(x) is str and re.fullmatch('[0-9a-f]{64}',x) is not None,'Exact observed SHA/nonce required');return x

def physical(p,directory=False):
 p=Path(p);need(p.is_absolute() and p.resolve(strict=True)==p and p.stat().st_uid==0,'Physical UID0 path required')
 need(p.is_dir() if directory else p.is_file(),'Wrong physical type');return p

def stable_raw(p,expected=None):
 p=physical(p);a=p.lstat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  z=os.fstat(fd);chunks=[]
  while True:
   b=os.read(fd,1048576)
   if not b:break
   chunks.append(b)
  b=b''.join(chunks);f=lambda v:(v.st_dev,v.st_ino,v.st_uid,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
  need(stat.S_ISREG(z.st_mode) and f(a)==f(z)==f(os.fstat(fd))==f(p.lstat()) and len(b)==z.st_size,'Stable same-FD bytes required')
  if expected is not None:need(sha(b)==pin(expected),'Exact file SHA changed')
  return b
 finally:os.close(fd)

def write_once(p,x):
 with p.open('xb') as f:f.write(canonical(x));f.flush();os.fsync(f.fileno())

def host():need(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','Own physical root/h100-10 required')
def process(pid,job=None,step=None):
 need(type(pid) is int and pid>0,'Actual positive PID required');p=Path('/proc')/str(pid)
 a=(p/'stat').read_text().rsplit(')',1)[1].split()[19];status=(p/'status').read_text();cg=(p/'cgroup').read_text();b=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
 need(a==b and re.fullmatch('[1-9][0-9]*',a) is not None,'Process start changed')
 need(set(map(int,re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split()))=={0},'Actual four UID values differ')
 state=re.search(r'^State:\s+(\S+)',status,re.M).group(1);need(state in ('R','S','D','I'),'Actual process not live')
 if job is not None:need(re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',cg,re.M) is not None,'Actual job cgroup differs')
 if step is not None:need(re.search(r'(?:^|/)step_'+re.escape(step)+r'(?:\.scope)?(?:/|$)',cg,re.M) is not None,'Actual step cgroup differs')
 return {'pid':pid,'start_ticks':a,'uid':0,'state':state,'cgroup':cg,'status_original':status,'hostname':socket.gethostname()}

def self_pin(expected):
 host();need(ROOT/'tools' in HERE.parents,'Isolated physical ROOT/tools controller required');stable_raw(HERE,expected)

def context(config):
 self_pin(config['controller_sha256']);tm=unique(stable_raw(FIRST5/'manifest.json',FIRST5_SHA))
 need(tm['source_commit']==SOURCE and tm['native_payload_tools_manifest_sha256']==NATIVE_SHA,'Current first5/native source binding required')
 for n,p in tm['tools'].items():stable_raw(FIRST5/n,p)
 raw=stable_raw(FIRST5/'inspect_first5_cpu.py',INSPECT_SHA);m=types.ModuleType('actual3304_frozen_first5_inspector');m.__file__=str(FIRST5/'inspect_first5_cpu.py');exec(compile(raw,m.__file__,'exec',dont_inherit=True),m.__dict__)
 m.JOB=JOB;m.RUN=RUN;m.TOOLS_MANIFEST_SHA=FIRST5_SHA
 manifest,mp,mpin,before=m.context()
 need(config['job_id']==JOB and config['source_commit']==SOURCE and config['scope']==SCOPE and config['contract']==CONTRACT,'Wrong fixed actual run/contract/scope')
 need(mpin==config['checkpoint_manifest_sha256'] and manifest['pause_nonce']==config['pause_nonce'] and manifest['driver_pid']==config['driver_pid'],'Actual pause manifest SHA/nonce/driver differs')
 driver=process(manifest['driver_pid'],JOB);need(driver['start_ticks']==config['driver_start_ticks'],'Original paused driver start changed')
 command=['scontrol','show','job',JOB,'-o'];r=subprocess.run(command,capture_output=True,text=True,timeout=20)
 need(r.returncode==0,'Actual scheduler query failed');fields=dict(x.split('=',1) for x in r.stdout.split() if '=' in x)
 need(fields.get('JobId')==JOB and fields.get('JobState')=='RUNNING' and fields.get('NodeList')=='h100-10' and fields.get('TresPerNode')=='gres:gpu:h100:8' and fields.get('OverSubscribe')=='NO' and fields.get('UserId')=='root(0)','Own running exclusive8H100 allocation lost')
 return {'manifest_sha256':mpin,'manifest_path':str(mp),'driver':driver,'inspector_context':before,'scontrol_command':command,'scontrol_original':{'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode}}

def role_command(role,launch,launch_sha):return [str(PYTHON),str(HERE),role,'--launch',str(launch),'--launch-sha256',launch_sha]
def srun_command(launch,launch_sha):
 return ['srun','--jobid='+JOB,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--nodelist=h100-10','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(REPO)]+role_command('cpu-step',launch,launch_sha)

def load_launch(p,expected):
 p=physical(p);need(p==OUTPUT/'launch-original.json','Only own actual once marker allowed');c=unique(stable_raw(p,expected))
 need(set(c)=={'version','job_id','source_commit','scope','contract','controller_sha256','checkpoint_manifest_sha256','pause_nonce','driver_pid','driver_start_ticks','output'} and type(c['version']) is int and c['version']==1 and c['job_id']==JOB and c['source_commit']==SOURCE and c['scope']==SCOPE and c['contract']==CONTRACT and c['output']==str(OUTPUT),'Launch fields/scope differ')
 for k in ('controller_sha256','checkpoint_manifest_sha256','pause_nonce'):pin(c[k])
 need(type(c['driver_pid']) is int and c['driver_pid']>0 and type(c['driver_start_ticks']) is str and re.fullmatch('[1-9][0-9]*',c['driver_start_ticks']) is not None,'Actual driver identity required');self_pin(c['controller_sha256']);physical(OUTPUT,True);return c

def waited_code(rc):return rc if rc>=0 else 128-rc

def wait_child(argv,out,prefix):
 """Always preserve true waited exit even if child-start metadata capture fails."""
 with (out/(prefix+'-stdout.original')).open('xb') as stdout,(out/(prefix+'-stderr.original')).open('xb') as stderr:
  p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
  try:
   write_once(out/(prefix+'-start.original.json'),{'argv':argv,'child_pid':p.pid,'parent':process(os.getpid()),'child':process(p.pid),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':SCOPE})
  finally:
   rc=p.wait()
   write_once(out/(prefix+'-exit.original.json'),{'returncode':rc,'waited':True,'waited_by_pid':os.getpid(),'child_pid':p.pid,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':SCOPE})
 return waited_code(rc)

def controller(c,launch,lsha):
 before=context(c);write_once(OUTPUT/'controller-start.original.json',{'process':process(os.getpid()),'before':before,'scope':SCOPE})
 return wait_child(srun_command(launch,lsha),OUTPUT,'srun')

def supervise(c,launch,lsha):
 before=context(c);write_once(OUTPUT/'supervisor-start.original.json',{'process':process(os.getpid()),'before':before,'scope':SCOPE})
 return wait_child(role_command('controller',launch,lsha),OUTPUT,'controller')

def cpu_step(c):
 host();need(os.environ.get('SLURM_JOB_ID')==JOB and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('OMP_NUM_THREADS')=='1','CPU-only step actual environment differs')
 step=os.environ.get('SLURM_STEP_ID');need(type(step) is str and re.fullmatch('[0-9]+',step) is not None,'Actual CPU Slurm step ID required')
 before=context(c);own=process(os.getpid(),JOB,step);write_once(OUTPUT/'cpu-step-before-exec.original.json',{'process':own,'job_id':JOB,'step_id':step,'cuda_visible_devices':'','omp_num_threads':'1','before':before,'scope':SCOPE})
 # Exec preserves the CPU process PID/start/cgroup for the pinned inspector.
 argv=[str(PYTHON),str(FIRST5/'inspect_first5_cpu.py'),'--job-id',JOB,'--tools-manifest-sha256',FIRST5_SHA,'--output',str(OUTPUT)]
 write_once(OUTPUT/'frozen-inspector-command.original.json',{'argv':argv,'inspector_sha256':INSPECT_SHA,'scope':SCOPE});os.execv(str(PYTHON),argv)

def start(a):
 self_pin(a.controller_sha256);physical(RUN,True);physical(RUN/'evidence',True)
 c={'version':1,'job_id':a.job_id,'source_commit':SOURCE,'scope':SCOPE,'contract':CONTRACT,'controller_sha256':pin(a.controller_sha256),'checkpoint_manifest_sha256':pin(a.checkpoint_manifest_sha256),'pause_nonce':pin(a.pause_nonce),'driver_pid':a.driver_pid,'driver_start_ticks':a.driver_start_ticks,'output':str(OUTPUT)}
 need(type(a.driver_pid) is int and a.driver_pid>0 and type(a.driver_start_ticks) is str and re.fullmatch('[1-9][0-9]*',a.driver_start_ticks) is not None,'Actual driver start required')
 before=context(c);need(not os.path.lexists(OUTPUT),'Existing/partial once directory never retried');OUTPUT.mkdir(mode=0o700)
 write_once(OUTPUT/'root-start-context.original.json',before);launch=OUTPUT/'launch-original.json';write_once(launch,c);lsha=sha(stable_raw(launch));argv=role_command('supervise',launch,lsha)
 with (OUTPUT/'supervisor-stdout.original').open('xb') as stdout,(OUTPUT/'supervisor-stderr.original').open('xb') as stderr:
  child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
  # Outcome here is launch only. On metadata/transport failure do not start again.
  write_once(OUTPUT/'durable-launch.original.json',{'argv':argv,'child_pid':child.pid,'child':process(child.pid),'launcher':process(os.getpid()),'scope':SCOPE,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps({'launched':True,'job_id':JOB,'output':str(OUTPUT),'supervisor_pid':child.pid,'launch_sha256':lsha,'scope':SCOPE},sort_keys=True))

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='role',required=True);a=sub.add_parser('start')
 for k in ('job-id','controller-sha256','checkpoint-manifest-sha256','pause-nonce','driver-start-ticks'):a.add_argument('--'+k,required=True)
 a.add_argument('--driver-pid',type=int,required=True)
 for role in ('supervise','controller','cpu-step'):
  q=sub.add_parser(role);q.add_argument('--launch',type=Path,required=True);q.add_argument('--launch-sha256',required=True)
 args=p.parse_args()
 if args.role=='start':start(args);return
 c=load_launch(args.launch,args.launch_sha256)
 if args.role=='supervise':sys.exit(supervise(c,args.launch,args.launch_sha256))
 if args.role=='controller':sys.exit(controller(c,args.launch,args.launch_sha256))
 cpu_step(c)
if __name__=='__main__':main()
