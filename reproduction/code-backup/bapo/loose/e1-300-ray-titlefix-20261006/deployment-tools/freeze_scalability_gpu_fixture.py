"""Forced OWN freeze fixture: real NCCL8 +301CPU, never an unsafe GPU sample."""
import argparse,datetime,hashlib,importlib.util,json,math,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006')
WATCH_SHA='53ed622dcce4aa5475d80f6489683860a3f43d7d1b6e4c6aa5657649d2a75a80'
SCOPE='FORCED_OWN_FREEZE_FIXTURE_NOT_PRODUCTION_UNSAFE_OBSERVATION'

def require(condition,message):
 if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with tmp.open('w')as f:json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def stat(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,ppid=int(s[1]),state=s[0],start_ticks=int(s[19]),uid=p.stat().st_uid)
def load_watch():
 path=ROOT/'tools/resource_watch.py';require(path.resolve()==path and not path.is_symlink()and sha(path)==WATCH_SHA,'Exact approved watcher module required')
 spec=importlib.util.spec_from_file_location('fixture_bound_watch_'+WATCH_SHA[:12],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);require(sha(path)==WATCH_SHA,'Watcher changed on import');return module

def validate_fixture_identity(run,source,mode):
 job=os.environ.get('SLURM_JOB_ID','');require(job.isdigit()and int(job)>0,'Actual Slurm job required');run=Path(run)
 require(run==ROOT/'runs'/f'bapo-instruct300-hardware-{job}'and run.is_dir()and run.resolve()==run,'Actual canonical hardware RUN required')
 marker=str(run)+'-fixture-foreign'if mode=='sentinel'else str(run)
 require(os.environ.get('BAPO_RUN_ID')==marker,'Exact actual run environment required')
 require((run/'source-commit.txt').read_text().strip()==source==(ROOT/'source-expected.txt').read_text().strip(),'Frozen actual source markers required')
 require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==source,'Actual source HEAD changed')
 require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True),'Actual source dirty')
 self_path=Path(__file__).resolve();require(self_path.parent==ROOT/'tools'and not Path(__file__).is_symlink(),'Actual newROOT tool required')
 tools=read(ROOT/'logs/tools-manifest.json');require(tools['source_commit']==source and tools['files'][self_path.name]==sha(self_path)and tools['files']['resource_watch.py']==WATCH_SHA,'Independent tools binding required')
 return int(job),run

def validate_rank_record(row,rank,source,job,run,expected_epoch):
 require(type(row.get('rank'))is int and row['rank']==rank and row.get('world_size')==8 and row.get('backend')=='nccl','Actual rank/NCCL identity mismatch')
 require(row.get('source_commit')==source and row.get('job_id')==job and row.get('run_root')==str(run)and row.get('watch_sha256')==WATCH_SHA and row.get('fixture_sha256')==sha(__file__),'Rank source/tool/allocation changed')
 require(type(row.get('cuda_device'))is int and row['cuda_device']==rank and row.get('nccl_socket_ifname')=='lo'and row.get('nccl_ib_disable')=='1','Actual local CUDA/NCCL transport mismatch')
 require(row.get('pid')==expected_epoch['pid']and row.get('start_ticks')==expected_epoch['start_ticks'],'Rank process epoch changed')
 require(type(row.get('completed_collectives'))is int and row['completed_collectives']>=1 and type(row.get('allreduce_value'))in (float,int)and math.isfinite(row['allreduce_value'])and row['allreduce_value']==36.,'Real allreduce heartbeat nonfinite/incorrect')
 return row['completed_collectives']

def validate_sentinel_exit(returncode):
 require(type(returncode)is int and returncode==0,'Foreign sentinel did not exit naturally')

def validate_paused(first,second,states,created):
 require(set(first)==set(range(8))and first==second,'Stopped rank counters advanced/missing')
 require(len(states)>=310 and set(states)==set(created)and all(v['state']in ('T','t')and all(v[k]==created[pid][k]for k in ('pid','ppid','start_ticks','uid'))for pid,v in states.items()),'All original own CPU/GPU epochs must be stopped')

def checked_inventory(w,run,job,uuids,phase_pid,created):
 rows=w.freeze_inventory(run,job,uuids,phase_pid)
 # Refuse unrelated same-run processes. Never filter them out or STOP them.
 require(set(rows)<=set(created),'Unrelated same-run PID prevents forced fixture freeze')
 for pid,row in rows.items():require(row['start_ticks']==created[pid]['start_ticks']and row['uid']==created[pid]['uid'],'Fixture PID epoch replaced')
 return rows

def fresh_outer_slot(row,job,run,now):
 timestamp=row.get('time')
 return type(timestamp)in (int,float)and math.isfinite(timestamp)and 0<=now-timestamp<=1.5 and row.get('run')==str(run)and row.get('slurm_job_id')==str(job)and row.get('pause')is False and row.get('reason')=='within_threshold'and 'event'not in row

def wait_outer_slot(run,job,phase_child,directory):
 until=time.monotonic()+25
 while time.monotonic()<until:
  require(phase_child.poll()is None,'Fixture phase exited waiting for real outer sample')
  path=run/'resource-watch.jsonl'
  if path.is_file():
   data=path.read_bytes();lines=data.splitlines()
   if data.endswith(b'\n')and lines:
    row=json.loads(lines[-1])
    if fresh_outer_slot(row,job,run,time.time()):
     atomic(directory/'outer-sample-selection.json',dict(scope='SCHEDULING_ONLY_NOT_RESOURCE_PERMISSION',raw_row=row,raw_line_sha256=hashlib.sha256(lines[-1]).hexdigest(),selection_time=time.time()));return
  time.sleep(.02)
 raise ValueError('No fresh true outer within-threshold sample for forced own test')

def bounded_created_cleanup(created,run,job,log):
 """CPU coordinator cleanup only for original test-created /proc epochs."""
 for pid,old in created.items():
  fd=None
  try:
   now=stat(pid);require(now['start_ticks']==old['start_ticks']and now['uid']==os.getuid(),'Cleanup epoch/UID changed')
   raw=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
   require(raw.count(('BAPO_RUN_ID='+str(run)).encode())==1 and raw.count(('SLURM_JOB_ID='+str(job)).encode())==1,'Cleanup foreign marker refuses signal')
   command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0');require(len(command)>1 and command[1]==str(Path(__file__).resolve()).encode(),'Cleanup command no longer fixture-owned')
   fd=os.pidfd_open(pid,0);last=stat(pid);require(last['start_ticks']==old['start_ticks']and last['uid']==os.getuid(),'Cleanup PID reused after pidfd')
   reread=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
   require(reread.count(('BAPO_RUN_ID='+str(run)).encode())==1 and reread.count(('SLURM_JOB_ID='+str(job)).encode())==1 and (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')==command,'Cleanup identity changed after pidfd')
   signal.pidfd_send_signal(fd,signal.SIGKILL,None,0);log(dict(event='fixture_created_cleanup',pid=pid,start_ticks=old['start_ticks']))
  except FileNotFoundError:pass
  except ProcessLookupError:pass
  except Exception as exc:log(dict(event='fixture_cleanup_refused',pid=pid,error=str(exc)))
  finally:
   if fd is not None:os.close(fd)


def worker(args,job,run):
 import torch,torch.distributed as dist
 rank=args.rank;directory=Path(args.output);torch.cuda.set_device(rank)
 visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',');require(len(visible)==8,'Eight actual visible CUDA devices required')
 uuid=subprocess.check_output(['nvidia-smi','--id='+visible[rank],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 w=load_watch();require(uuid in w.lease_uuids(),'Actual CUDA UUID outside own allocation')
 dist.init_process_group('nccl',init_method='tcp://127.0.0.1:'+str(args.port),rank=rank,world_size=8,timeout=datetime.timedelta(seconds=30))
 identity=stat(os.getpid());count=0;start=time.monotonic()
 with (directory/f'rank-{rank}-raw.jsonl').open('x',buffering=1)as raw:
  try:
   while time.monotonic()-start<120:
    stop=torch.tensor([int((directory/'stop').exists())],device='cuda',dtype=torch.int32);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
    if stop.item():break
    value=torch.tensor([rank+1.],device='cuda',dtype=torch.float32);dist.all_reduce(value);torch.cuda.synchronize()
    require(torch.isfinite(value).all().item()and value.item()==36.,'NCCL allreduce finite oracle failed');count+=1
    record=dict(scope=SCOPE,rank=rank,world_size=8,backend=dist.get_backend(),pid=os.getpid(),start_ticks=identity['start_ticks'],uid=os.getuid(),source_commit=args.source,job_id=job,run_root=str(run),fixture_sha256=sha(__file__),watch_sha256=WATCH_SHA,cuda_device=rank,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_uuid=uuid,nccl_socket_ifname=os.environ['NCCL_SOCKET_IFNAME'],nccl_ib_disable=os.environ['NCCL_IB_DISABLE'],completed_collectives=count,allreduce_value=value.item(),time_unix=time.time())
    raw.write(json.dumps(record,allow_nan=False)+'\n');atomic(directory/f'heartbeat-{rank}.json',record);time.sleep(.025)
   require(count>0,'No real GPU collective');atomic(directory/f'rank-{rank}-finished.json',dict(status='ACTUAL_NCCL_RANK_FINISHED',**record))
  finally:dist.destroy_process_group()

def phase(args,job,run):
 directory=Path(args.output);created={os.getpid():stat(os.getpid())};children=[];ranks=[];streams=[]
 created_raw=(directory/'created-so-far.jsonl').open('x',buffering=1)
 def created_record(pid):
  created[pid]=stat(pid);created_raw.write(json.dumps(dict(source_commit=args.source,job_id=job,run_root=str(run),creator_pid=os.getpid(),process=created[pid]))+'\n')
 created_record(os.getpid())
 for _ in range(301):
  pid=os.fork()
  if pid==0:
   deadline=time.monotonic()+150
   while not (directory/'cpu-stop').exists()and time.monotonic()<deadline:time.sleep(.02)
   os._exit(0)
  children.append(pid);created_record(pid)
 env=dict(os.environ,NCCL_SOCKET_IFNAME='lo',NCCL_IB_DISABLE='1')
 for rank in range(8):
  stream=(directory/f'rank-{rank}.log').open('x');streams.append(stream)
  command=[sys.executable,str(Path(__file__).resolve()),'--mode','rank','--run',str(run),'--source',args.source,'--output',str(directory),'--port',str(args.port),'--rank',str(rank)]
  child=subprocess.Popen(command,env=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='8'),stdout=stream,stderr=subprocess.STDOUT);ranks.append(child);created_record(child.pid)
 atomic(directory/'topology.json',dict(phase_pid=os.getpid(),cpu_pids=children,rank_pids=[x.pid for x in ranks],created={str(p):r for p,r in created.items()}))
 try:
  deadline=time.monotonic()+120
  while not (directory/'stop').exists()and time.monotonic()<deadline:
   require(all(x.poll()is None for x in ranks),'CUDA rank exited before fixture control finished');time.sleep(.02)
 finally:
  (directory/'stop').touch();(directory/'cpu-stop').touch()
  for child in ranks:child.wait(timeout=35)
  for pid in children:os.waitpid(pid,0)
  for stream in streams:stream.close()
  created_raw.close()
 require(all(x.returncode==0 for x in ranks),'Real NCCL worker failed')

def controller(args,job,run):
 w=load_watch();require(hasattr(os,'pidfd_open')and hasattr(signal,'pidfd_send_signal'),'Actual Linux pidfd required')
 directory=run/'freeze-scalability-gpu';directory.mkdir();require(args.output is None,'Controller owns fresh output directory')
 logpath=directory/'forced-own-freeze-journal.jsonl';start=time.monotonic();created={};f=None;phase_child=None;foreign=None;recovered=False
 with logpath.open('x',buffering=1)as raw:
  def log(event):event.update(scope=SCOPE,run_root=str(run),job_id=job,source_commit=args.source,fixture_sha256=sha(__file__),watch_sha256=WATCH_SHA,time_unix=time.time());raw.write(json.dumps(event,allow_nan=False)+'\n')
  import socket
  sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
  sentinel_cmd=[sys.executable,str(Path(__file__).resolve()),'--mode','sentinel','--run',str(run),'--source',args.source,'--output',str(directory)]
  foreign=subprocess.Popen(sentinel_cmd,env=dict(os.environ,BAPO_RUN_ID=str(run)+'-fixture-foreign',CUDA_VISIBLE_DEVICES=''))
  stream=(directory/'phase.log').open('x');phase_cmd=[sys.executable,str(Path(__file__).resolve()),'--mode','phase','--run',str(run),'--source',args.source,'--output',str(directory),'--port',str(port)]
  phase_child=subprocess.Popen(phase_cmd,stdout=stream,stderr=subprocess.STDOUT);created[phase_child.pid]=stat(phase_child.pid)
  try:
   deadline=time.monotonic()+90
   while not (directory/'topology.json').exists():
    require(phase_child.poll()is None and time.monotonic()<deadline,'Actual fixture process creation failed/timed out');time.sleep(.02)
   topology=read(directory/'topology.json');created={int(p):v for p,v in topology['created'].items()};require(len(topology['cpu_pids'])>=301 and len(topology['rank_pids'])==8 and len(created)>=310,'Real CPU/GPU process tree incomplete')
   epochs={i:created[p]for i,p in enumerate(topology['rank_pids'])}
   def heartbeats():
    records={i:read(directory/f'heartbeat-{i}.json')for i in range(8)}
    require({v.get('cuda_uuid')for v in records.values()}==w.lease_uuids()and all(v.get('cuda_visible_devices')==os.environ['CUDA_VISIBLE_DEVICES']for v in records.values()),'Eight actual CUDA UUIDs/visibility do not match leased allocation')
    return {i:validate_rank_record(records[i],i,args.source,job,run,epochs[i])for i in range(8)}
   def wait_ready(minimum=None):
    until=time.monotonic()+60
    while time.monotonic()<until:
     require(phase_child.poll()is None,'Actual NCCL phase exited')
     try:
      values=heartbeats()
      if all(values[i]>= (3 if minimum is None else minimum[i]+3)for i in range(8)):return values
     except FileNotFoundError:pass
     time.sleep(.02)
    raise ValueError('Eight real NCCL heartbeat progress deadline')
   before=wait_ready();require(foreign.poll()is None,'Foreign sentinel exited unexpectedly')
   wait_outer_slot(run,job,phase_child,directory)
   inventory=lambda r,j,u,p:checked_inventory(w,r,j,u,p,created)
   f=w.OwnedFreezer(str(run),phase_child.pid,log,inventory_fn=inventory);freeze_start=time.monotonic();f.freeze();freeze_seconds=time.monotonic()-freeze_start
   require(set(f.held)==set(created),'Actual full fixture tree not frozen');first=heartbeats();time.sleep(.3);second=heartbeats();states={pid:stat(pid)for pid in created};validate_paused(first,second,states,created)
   require(stat(foreign.pid)['state']not in ('T','t')and foreign.pid not in f.held,'Foreign sentinel was stopped')
   atomic(directory/'paused-proof.json',dict(before=before,paused_first=first,paused_second=second,states={str(p):s for p,s in states.items()},held_certificates={str(p):v['proof']for p,v in f.held.items()},foreign_pid=foreign.pid,foreign_state=stat(foreign.pid),freeze_seconds=freeze_seconds))
   f.thaw();after=wait_ready(first);recovered=True;atomic(directory/'resumed-proof.json',dict(before=first,after=after,finite_allreduce=36.,all_eight_progress=True))
   (directory/'stop').touch();phase_child.wait(timeout=40);require(phase_child.returncode==0,'Real CUDA/CPU phase exit failed')
  finally:
   if f and f.held:f.thaw(abort=True)
   (directory/'stop').touch();(directory/'cpu-stop').touch();(directory/'sentinel-stop').touch()
   partial=directory/'created-so-far.jsonl'
   if partial.is_file():
    for line in partial.read_bytes().splitlines():
     try:
      row=json.loads(line)
      require(row['creator_pid']==phase_child.pid and row['source_commit']==args.source and row['job_id']==job and row['run_root']==str(run),'Foreign creation record rejected')
      old=row['process'];require(type(old['pid'])is int and type(old['start_ticks'])is int and old['start_ticks']>0 and old['uid']==os.getuid(),'Malformed created epoch rejected');created[old['pid']]=old
     except json.JSONDecodeError:continue
   if phase_child.poll()is None:
    try:phase_child.wait(timeout=3)
    except subprocess.TimeoutExpired:bounded_created_cleanup(created,str(run),job,log);phase_child.wait(timeout=5)
   foreign.wait(timeout=5);validate_sentinel_exit(foreign.returncode);stream.close()
  require(recovered,'Forced own freeze did not prove actual recovery')
  signal_events=[json.loads(line)for line in logpath.read_text().splitlines()if line]
  require(all(x.get('pid')in created for x in signal_events if x.get('event')in ('owned_freeze_signal','owned_resume_signal','owned_abort_signal','fixture_created_cleanup'))and all(x.get('pid')!=foreign.pid for x in signal_events),'Signal journal contains nonfixture/foreign PID')
  elapsed=time.monotonic()-start;files={p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))for p in directory.iterdir()if p.is_file()and p!=logpath}
  log(dict(event='fixture_finished',forced_own_freeze=True,actual_unsafe_gpu_observation=False,wall_seconds=elapsed,reserved_gpu_hours=elapsed*8/3600))
 atomic(directory/'result.json',dict(status='PASS_ACTUAL_NCCL8_301CPU_FORCED_OWN_FREEZE_ONLY',scope=SCOPE,source_commit=args.source,job_id=job,run_root=str(run),fixture_sha256=sha(__file__),watch_sha256=WATCH_SHA,allocated_uuids=sorted(w.lease_uuids()),cpu_processes=len(topology['cpu_pids']),actual_ranks=8,all_own_stopped=True,all_rank_counters_stable=True,all_rank_counters_resumed=True,foreign_pid=foreign.pid,foreign_signals=0,actual_unsafe_gpu_observation=False,freeze_seconds=freeze_seconds,wall_seconds=elapsed,reserved_gpu_hours=elapsed*8/3600,raw_files=files,journal=dict(path=str(logpath),bytes=logpath.stat().st_size,sha256=sha(logpath))))
 print('ACTUAL_FORCED_OWN_NCCL8_CPU301_FREEZE_JSON='+json.dumps(read(directory/'result.json')),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('controller','phase','rank','sentinel'),default='controller');p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--output');p.add_argument('--rank',type=int);p.add_argument('--port',type=int);a=p.parse_args();job,run=validate_fixture_identity(a.run,a.source,a.mode)
 if a.mode=='controller':return controller(a,job,run)
 require(Path(a.output)==run/'freeze-scalability-gpu'and Path(a.output).resolve()==Path(a.output),'Owned fixture output required')
 if a.mode=='sentinel':
  deadline=time.monotonic()+150
  while not (Path(a.output)/'sentinel-stop').exists()and time.monotonic()<deadline:time.sleep(.02)
  return
 require(type(a.port)is int and 0<a.port<65536,'Local NCCL port required')
 if a.mode=='rank':require(type(a.rank)is int and 0<=a.rank<8,'Actual local rank required');return worker(a,job,run)
 return phase(a,job,run)
if __name__=='__main__':main()
