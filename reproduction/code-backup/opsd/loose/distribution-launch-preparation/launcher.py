"""Wait at most300s for root's actual allocation and retain fd9+native GPU leases."""
import argparse,hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
HISTORICAL_COMMIT='8e60976d364ab7d7ae51714415813c1359dafe9a'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loads(s):
 def unique(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate JSON key')
   d[k]=v
  return d
 return json.loads(s,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def owned_file(p,scope=ROOT):
 p=Path(p)
 if not p.is_absolute() or not p.is_file() or not p.resolve().is_relative_to(scope.resolve()):raise ValueError('File escapes own root or is missing')
 return p

def verify_wrapper(env):
 pin=env['OPSD_DISTRIBUTION_WRAPPER_MANIFEST_SHA256']
 if not re.fullmatch('[0-9a-f]{64}',pin):raise ValueError('External wrapper manifest pin required')
 p=owned_file(env['OPSD_DISTRIBUTION_WRAPPER_MANIFEST'])
 if digest(p)!=pin:raise ValueError('Wrapper manifest differs from external reviewed SHA')
 m=loads(p.read_text())
 if set(m['files'])!={'bootstrap.sh','launcher.py'}:raise ValueError('Exact two wrapper files required')
 for name,item in m['files'].items():
  path=owned_file(item['path'])
  if path.name!=name or digest(path)!=item['sha256']:raise ValueError('Wrapper byte pin mismatch')
 if owned_file(env['OPSD_DISTRIBUTION_LAUNCHER']).resolve()!=owned_file(m['files']['launcher.py']['path']).resolve():raise ValueError('Executing launcher path differs')
 if Path(__file__).resolve()!=owned_file(env['OPSD_DISTRIBUTION_LAUNCHER']).resolve():raise ValueError('Executing launcher bytes/path differ')
 if owned_file(env['OPSD_DISTRIBUTION_EXECUTING_BOOTSTRAP']).resolve()!=owned_file(m['files']['bootstrap.sh']['path']).resolve():raise ValueError('Actual bootstrap path differs')
 return m

def lease_fds(records):
 if len(records)!=1 or set(records[0])!={'uuid','fd'}:raise ValueError('One native lease required')
 fd=records[0]['fd']
 if type(fd) is not int or fd<3 or fd==9:raise ValueError('Invalid lease descriptor or root lock overwritten')
 if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',records[0]['uuid']):raise ValueError('Malformed GPU UUID')
 return (9,fd)
def validate_allocation(a,request):
 if set(a)!={'version','job_id','uid','node','selected_gpu_uuid'} or type(a['version']) is not int or a['version']!=1:raise ValueError('Exact external allocation schema required')
 if type(a['uid']) is not int or a['uid']<0 or not isinstance(a['job_id'],str) or not re.fullmatch('[0-9]+',a['job_id']):raise ValueError('Allocation UID/job types invalid')
 for key in ('job_id','uid','node','selected_gpu_uuid'):
  if a[key]!=request[key]:raise ValueError('Allocation differs from actual request: '+key)
 return a
def wait_allocation(run,request,timeout=300,clock=time.monotonic,pause=time.sleep):
 deadline=clock()+timeout;p=run/'allocation.json';pinpath=run/'allocation.sha256'
 while clock()<deadline:
  if p.exists() and pinpath.exists():
   p=owned_file(p,run);pinpath=owned_file(pinpath,run);pin=pinpath.read_text().strip()
   if not re.fullmatch('[0-9a-f]{64}',pin) or digest(p)!=pin:raise ValueError('Root-written allocation external SHA mismatch')
   return p,pin,validate_allocation(loads(p.read_text()),request)
  pause(min(1,max(0,deadline-clock())))
 raise TimeoutError('Root allocation/externally reviewed SHA not supplied within300s')
def collect_command(env,allocation_path,allocation_sha,output):
 binding=owned_file(env['OPSD_DISTRIBUTION_BINDING']);sha=env['OPSD_DISTRIBUTION_BINDING_SHA256'];probe=owned_file(env['OPSD_DISTRIBUTION_PROBE'])
 bundle=Path(env['OPSD_DISTRIBUTION_BUNDLE'])
 if not bundle.is_absolute() or not bundle.resolve().is_relative_to((ROOT/'tools').resolve()) or bundle.name=='initial-distribution-7ac-96cfe2326cfe':raise ValueError('New reviewed bundle required; failed96 bundle prohibited')
 if probe.parent.resolve()!=bundle.resolve() or binding.parent.resolve()!=bundle.resolve():raise ValueError('Probe/binding not in root-provided bundle')
 if not re.fullmatch('[0-9a-f]{64}',sha) or digest(binding)!=sha:raise ValueError('External distribution binding SHA mismatch')
 record=loads(binding.read_text())
 # The probe author may still change software. Pin only actual reviewed binding at launch.
 probe_sha=env['OPSD_DISTRIBUTION_PROBE_SHA256'];source_sha=env['OPSD_DISTRIBUTION_SOURCE_SHA256']
 if not re.fullmatch('[0-9a-f]{64}',probe_sha) or probe_sha!=record['probe_sha256'] or probe_sha!=digest(probe):raise ValueError('Actual probe differs from reviewed external SHA/binding')
 source=owned_file(record['source']['path'])
 if source.parent.resolve()!=bundle.resolve() or not re.fullmatch('[0-9a-f]{64}',source_sha) or source_sha!=record['source']['sha256'] or source_sha!=digest(source):raise ValueError('Actual source differs from externally reviewed source SHA/binding')
 if Path(output).exists():raise ValueError('Collect output must be newly created by probe')
 return ['python','-u',str(probe),'collect','--binding',str(binding),'--binding-sha256',sha,'--allocation',str(allocation_path),'--allocation-sha256',allocation_sha,'--output',str(output)]
def write_new(p,data):
 with Path(p).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--verify-wrapper-only',action='store_true');args=parser.parse_args();env=os.environ
 verify_wrapper(env)
 if args.verify_wrapper_only:return
 from search_r1.baselines.preflight import validate_inherited_gpu_leases
 validate_inherited_gpu_leases(1)
 job=env['SLURM_JOB_ID'];run=Path(env['OPSD_DISTRIBUTION_RUN'])
 if not re.fullmatch('[0-9]+',job) or job in {'2994','2996','2999','3007'}:raise ValueError('Actual fresh job required')
 if run!=ROOT/'runs'/('opsd-distribution-'+job) or not run.is_dir() or run.is_symlink():raise ValueError('Wrong own new run')
 if socket.gethostname().split('.')[0]!=env['SLURM_JOB_NODELIST'] or env['SLURM_JOB_NODELIST']!='h100-10':raise ValueError('Only h100-10 allocation allowed')
 if not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text()):raise ValueError('Outside own Slurm cgroup')
 records=loads(env['SEARCH_R1_GPU_LEASES']);fds=lease_fds(records)
 for fd in fds:os.set_inheritable(fd,True)
 # Root lock identity and exclusivity independently verified before writing requests.
 import fcntl
 target=ROOT/'gpu-job.lock';st=os.fstat(9);actual=target.stat()
 if (st.st_dev,st.st_ino)!=(actual.st_dev,actual.st_ino):raise ValueError('fd9 not production gpu-job.lock')
 fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
 probe_fd=os.open(target,os.O_RDWR)
 try:
  try:fcntl.flock(probe_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise ValueError('Production root job lock not held')
 finally:os.close(probe_fd)
 terminal=subprocess.check_output(['scontrol','show','job',job],text=True)
 for key,value in (('JobId',job),('JobState','RUNNING'),('NodeList','h100-10'),('NumNodes','1'),('TresPerNode','gres:gpu:h100:1')):
  if re.findall(r'\b'+key+r'=([^\s]+)',terminal)!=[value]:raise ValueError('Actual singleton Slurm allocation differs: '+key)
 if re.findall(r'\bUserId=[^\s(]+\(([0-9]+)\)',terminal)!=[str(os.getuid())]:raise ValueError('Slurm actual UID differs')
 write_new(run/'slurm-runtime.json',{'job_id':job,'actual_scontrol':terminal})
 request={'version':1,'job_id':job,'uid':os.getuid(),'node':'h100-10','selected_gpu_uuid':records[0]['uuid']}
 write_new(run/'allocation-request.json',request)
 write_new(run/'wrapper-runtime.json',{'job_id':job,'wrapper_manifest_sha256':env['OPSD_DISTRIBUTION_WRAPPER_MANIFEST_SHA256'],'launcher_sha256':digest(__file__),'bootstrap_sha256':digest(owned_file(env['OPSD_DISTRIBUTION_EXECUTING_BOOTSTRAP'])),'inherited_fds':list(fds),'actual_probe_sha256':digest(owned_file(env['OPSD_DISTRIBUTION_PROBE'])),'scope':'launcher identity only; no GPU collect accepted'})
 allocation_path,pin,allocation=wait_allocation(run,request)
 command=collect_command(env,allocation_path,pin,run/'collect')
 validate_inherited_gpu_leases(1)
 with (run/'collect.log').open('x') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=fds,env=dict(env))
 write_new(run/'collect-child-exit.json',{'job_id':job,'returncode':result.returncode,'scope':'read-only distribution probe only; not training or effect acceptance'})
 if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
