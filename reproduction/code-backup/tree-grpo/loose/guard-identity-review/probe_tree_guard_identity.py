"""Linux CPU-only exact guard-functions proof. Does not start guard/NVML or cancel jobs."""
import argparse,ast,hashlib,json,os,pathlib,resource,time
p=argparse.ArgumentParser();p.add_argument('--guard',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
raw=a.guard.read_bytes();assert hashlib.sha256(raw).hexdigest()=='284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88'
job=os.environ['SLURM_JOB_ID'];assert job.isdigit() and os.environ['SLURM_JOB_NODELIST']=='h100-9'
tree=ast.parse(raw);funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('has_job','process_identity','fd_pid','own')]
ns=dict(pathlib=pathlib,os=os,job=job,known_own={},zombie_proofs={});exec(compile(ast.Module(body=funcs,type_ignores=[]),str(a.guard),'exec'),ns)
assert ns['has_job'](pathlib.Path('/proc/self/cgroup').read_text())
r,w=os.pipe();pid=os.fork()
if pid==0:
 os.close(w);os.read(r,1);os._exit(0)
os.close(r)
try:
 assert ns['own'](pid) is True
 start=ns['known_own'][pid]['start'];fd=ns['known_own'][pid]['fd'];assert ns['fd_pid'](fd)==pid
 class P:
  def __init__(self,*parts):self.p=pathlib.Path(*parts)
  def read_text(self):
   if self.p==pathlib.Path('/proc',str(pid),'cgroup'):return '0::/outside_simulated'
   return self.p.read_text()
 ns['pathlib']=type('Proxy',(),dict(Path=P))
 assert ns['own'](pid) is False,'Living outside process must not be exempt'
 os.write(w,b'x');os.close(w);w=-1
 deadline=time.monotonic()+5
 while ns['process_identity'](pid)[0]!='Z':
  assert time.monotonic()<deadline;time.sleep(.01)
 assert ns['own'](pid) is True;proof=dict(ns['zombie_proofs'][pid]);assert proof['start_ticks']==start
 os.waitpid(pid,0);pid_waited=True
 assert ns['fd_pid'](fd)==-1;assert ns['own'](pid) is None
 out=dict(status='PASS',job_id=job,rlimit_nofile=resource.getrlimit(resource.RLIMIT_NOFILE),open_fd_count=len(os.listdir('/proc/self/fd')),probe_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),guard_sha256=hashlib.sha256(raw).hexdigest(),scope='Real Linux stat/pidfd/starttime/Z; registration actual own Slurm cgroup; outside membership simulated; no GPU/NVML timing proof',proof=proof,live_outside_refused=True,reaped_identity_not_exempt=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps(out))
finally:
 if w>=0:os.close(w)
 if not locals().get('pid_waited',False):os.waitpid(pid,0)
 for cached in ns['known_own'].values():os.close(cached['fd'])
