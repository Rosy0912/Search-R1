"""H10011 cohost guard: free-memory reserve, never foreign-load thresholds/signals."""
from pathlib import Path
import argparse,fcntl,json,math,os,signal,stat,subprocess,sys,time
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009')
SOURCE='7a501b38b7ed6b06bf39935b97fcb7bbd56ecc47'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def epoch(pid):
 f=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 require(f[0] not in ('Z','X'),'process no longer live')
 return int(f[19]),int(f[1])
def identity(run):
 require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),'optimized Python forbidden')
 require(sys.platform=='linux' and os.uname().nodename=='h100-11','only authorized H10011')
 require(sys.prefix==str(ROOT/'env'),'isolated cohost environment required')
 require(run.parent==ROOT/'runs' and run.is_dir() and run.resolve()==run,'canonical own RUN required')
 require(os.environ.get('BAPO_RUN_ID')==str(run) and os.environ.get('SLURM_JOB_ID','').isdigit(),'exact own run/job required')
 require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'source080 required')
 require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'dirty source080 forbidden')
def inventory(text,selected):
 rows={}
 for line in text.splitlines():
  if not line.strip():continue
  index,uuid,total,free,used,sm=[x.strip() for x in line.split(',')]
  values=[float(v) for v in (total,free,used,sm)]
  require(all(math.isfinite(v) for v in values),'nonfinite GPU inventory')
  total,free,used,sm=values
  require(total>0 and 0<=free<=total and 0<=used<=total and 0<=sm<=100 and free+used<=total+2,'invalid GPU inventory')
  require(index not in rows and uuid.startswith('GPU-'),'duplicate/invalid GPU')
  rows[index]=dict(uuid=uuid,total_mib=total,free_mib=free,used_mib=used,sm_percent=sm)
 chosen=[]
 for token in selected:
  match=[v for k,v in rows.items() if token in (k,v['uuid'])];require(len(match)==1,'cannot resolve selected GPU');chosen+=match
 require(len(chosen)==8 and len({r['uuid']for r in chosen})==8,'eight distinct GPUs required')
 return chosen
def snapshot(selected):
 p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
 return inventory(p.stdout,selected)
def admission(records,minimum):
 require(math.isfinite(minimum) and minimum>=1024,'finite reserve >=1GiB required')
 return all(r['free_mib']>=minimum for r in records)
def validate_leases(records):
 leases=json.loads(os.environ.get('SEARCH_R1_GPU_LEASES','[]'))
 require(os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1' and len(leases)==8,'real inherited leases required')
 require({r['uuid']for r in leases}=={r['uuid']for r in records} and len({r['fd']for r in leases})==8,'lease UUID/fd mismatch')
 for item in leases:
  fd=int(item['fd']);p=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(item['uuid']+'.lock');a=os.fstat(fd);b=p.lstat()
  require(stat.S_ISREG(a.st_mode) and stat.S_ISREG(b.st_mode) and a.st_uid==os.getuid() and (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'lease inode/uid differs')
  probe=os.open(p,os.O_RDWR)
  try:
   try:fcntl.flock(probe,fcntl.LOCK_SH|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise RuntimeError('lease is not held exclusively')
   fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  finally:os.close(probe)
  os.set_inheritable(fd,True)
def processes(run):
 p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
 result=[]
 for line in p.stdout.splitlines():
  if not line.strip():continue
  uuid,pid,memory=[v.strip() for v in line.split(',')];pid=int(pid);own=False;birth=None
  try:
   before=epoch(pid);proc=Path('/proc')/str(pid);env=(proc/'environ').read_bytes().split(b'\0')
   own=(('BAPO_RUN_ID='+str(run)).encode() in env and ('SLURM_JOB_ID='+os.environ['SLURM_JOB_ID']).encode() in env and proc.stat().st_uid==os.getuid() and epoch(pid)==before)
   if own:birth=before[0]
  except (OSError,ValueError,RuntimeError):pass # Unreadable foreign is never classified as own or signalled.
  result.append(dict(uuid=uuid,pid=pid,memory_mib=memory,own=own,start_ticks=birth))
 return result
def own_processes(run):
 excluded={os.getpid()};pid=os.getpid()
 for _ in range(256):
  try:pid=epoch(pid)[1]
  except (OSError,RuntimeError):break
  if pid<=1 or pid in excluded:break
  excluded.add(pid)
 found={}
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name) in excluded:continue
  pid=int(proc.name)
  try:
   before=epoch(pid);env=(proc/'environ').read_bytes().split(b'\0')
   if (('BAPO_RUN_ID='+str(run)).encode() in env and ('SLURM_JOB_ID='+os.environ['SLURM_JOB_ID']).encode() in env and proc.stat().st_uid==os.getuid() and epoch(pid)==before):found[pid]=before[0]
  except (OSError,ValueError,RuntimeError):continue
 return found
def cleanup(run):
 initial=own_processes(run);held={};errors=[]
 for pid,birth in initial.items():
  try:
   fd=os.pidfd_open(pid)
   if own_processes(run).get(pid)!=birth:os.close(fd);continue
   held[pid]=fd;signal.pidfd_send_signal(fd,signal.SIGTERM)
  except ProcessLookupError:pass
  except OSError as exc:errors.append(repr(exc))
 deadline=time.monotonic()+3
 while held and time.monotonic()<deadline:
  if not own_processes(run):break
  time.sleep(.1)
 for pid,fd in held.items():
  try:signal.pidfd_send_signal(fd,signal.SIGKILL)
  except ProcessLookupError:pass
  except OSError as exc:errors.append(repr(exc))
  finally:os.close(fd)
 deadline=time.monotonic()+2
 while own_processes(run) and time.monotonic()<deadline:time.sleep(.1)
 remaining=own_processes(run);report=dict(own_marker=str(run),job=os.environ['SLURM_JOB_ID'],births=initial,stopped_pids=sorted(held),remaining=[dict(pid=p,start_ticks=b)for p,b in remaining.items()],errors=errors)
 path=run/'own-process-cleanup.json'
 if path.exists():
  require(path.is_file() and not path.is_symlink() and path.resolve()==path,'canonical prior cleanup report required')
  prior=json.loads(path.read_bytes());require(prior.get('own_marker')==str(run) and prior.get('job')==os.environ['SLURM_JOB_ID'] and not prior.get('remaining') and not remaining and not errors,'prior cleanup cannot be reused')
 else:
  with path.open('x') as f:json.dump(report,f);f.flush();os.fsync(f.fileno())
 require(not remaining and not errors,'own process cleanup incomplete')
 return report
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reserve-free-mib',type=float,default=1024);p.add_argument('--interval',type=float,default=2);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();command=a.command[1:]if a.command[:1]==['--']else a.command
 identity(a.run);require(command and math.isfinite(a.interval) and 1<=a.interval<=10,'bounded interval and command required');admission([],a.reserve_free_mib)
 require(Path(__file__).resolve()==ROOT/'tools/resource_watch.py' and sys.argv[0]==str(ROOT/'tools/resource_watch.py'),'exact source080 guard path required')
 selected=[x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')if x];records=snapshot(selected);validate_leases(records)
 os.environ['COHOST_WATCHER_ADMITTED_RUN']=str(a.run)
 child=None
 def interrupted(signum,frame):raise SystemExit(128+signum)
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
 with (a.run/'resource-watch.jsonl').open('x',buffering=1) as log:
  while True:
   if child is not None and child.poll() is not None:return child.returncode if child.returncode>=0 else 128-child.returncode
   try:
    records=snapshot(selected);validate_leases(records);ok=admission(records,a.reserve_free_mib);procs=processes(a.run)
    event=dict(time=time.time(),run=str(a.run),slurm_job_id=os.environ['SLURM_JOB_ID'],phase_pid=child.pid if child else None,policy='COHOST_FREE_RESERVE_NO_FOREIGN_SM_OR_MEMORY_PERCENT_GATE',reserve_free_mib=a.reserve_free_mib,gpus=records,processes=procs,pause=not ok,reason='within_free_reserve'if ok else'free_reserve_exhausted')
   except Exception as exc:
    event=dict(time=time.time(),run=str(a.run),pause=True,reason='unknown_memory_or_lease_fail_closed',error=repr(exc));ok=False
   log.write(json.dumps(event,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
   if not ok:return 42 # Outer trap performs only own run/job pidfd cleanup; never signal foreign here.
   if child is None:
    os.environ['SEARCH_R1_GPU_RESOURCE_GUARD']=json.dumps(dict(pid=os.getpid(),start_ticks=epoch(os.getpid())[0],run=str(a.run)))
    child=subprocess.Popen(command,close_fds=False)
   time.sleep(a.interval)
if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--cleanup-run':
  run=Path(sys.argv[2]);identity(run);print(json.dumps(cleanup(run)));sys.exit(0)
 run=None
 try:
  if '--run' in sys.argv:run=Path(sys.argv[sys.argv.index('--run')+1])
  code=main()
 finally:
  if run is not None and os.environ.get('COHOST_WATCHER_ADMITTED_RUN')==str(run):
   identity(run);cleanup(run)
 sys.exit(code)

