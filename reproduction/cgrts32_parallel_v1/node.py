"""Four allocated node orchestrators; cleanup targets only their own children."""
import json,os,signal,socket,subprocess,sys,threading,time
from pathlib import Path
from .contract import read_plan,sha,node_coordinates,validate_gate
from .runtime import save,memory

def wait_for(path,seconds,root):
 end=time.monotonic()+seconds
 while not Path(path).exists():
  assert time.monotonic()<end,'Bounded startup wait expired: '+str(path)
  assert not list(root.glob('node-*/fatal.json')),'Another allocated node failed; inspect without replay'
  time.sleep(2)
 return json.loads(Path(path).read_text())
def stop_owned(proc):
 if proc is None or proc.poll() is not None:return
 os.killpg(proc.pid,signal.SIGTERM)
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
def launch(pp):
 p=read_plan(pp);root=Path(p['root']);n,master=node_coordinates(os.environ);d=root/f'node-{n}';d.mkdir()
 ip=socket.gethostbyname(socket.gethostname());assert not ip.startswith('127.')
 save(d/'entry-once.json',dict(at=time.time(),node=n,hostname=socket.gethostname(),ip=ip,master=master,master_port=int(os.environ['MASTER_PORT']),plan_sha256=sha(pp)))
 gate=wait_for(root/'scheduler-release-once.json',600,root);sub=json.loads((root/'submission.json').read_text());validate_gate(p,sub,gate)
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20)
 devices=[r.split(',') for r in rows.splitlines() if r.strip()];assert len(devices)==8 and len({r[0].strip() for r in devices})==8
 assert all('A100' in r[1] and int(r[2])>=79000 for r in devices)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
 save(d/'hardware.json',dict(devices=devices,no_prior_gpu_processes=True,scheduler_uid=sub['uid']))
 os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in range(8))
 cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=4','--nproc-per-node=8','--node-rank='+str(n),'--master-addr='+master,'--master-port='+os.environ['MASTER_PORT'],'--max-restarts=0','-m','reproduction.cgrts32_parallel_v1.warmup',str(pp)]
 proc=None
 try:
  with (d/'warmup.log').open('xb') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   rc=proc.wait(timeout=600)
  assert rc==0,'32-rank in-allocation infrastructure warmup failed'
  wait_for(root/'warmup/all32-identities.json',60,root)
  cmd=[sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,'-m','reproduction.cgrts32_parallel_v1.node','serve',str(pp)]
  os.execvp(cmd[0],cmd)
 finally:stop_owned(proc)
def serve(pp):
 p=read_plan(pp);root=Path(p['root']);n,_=node_coordinates(os.environ);d=root/f'node-{n}';rayproc=trainproc=None;finished=threading.Event()
 def memory_watch():
  while not finished.is_set():
   value=dict(at=time.time(),node=n,memory=memory());tmp=d/'memory.pending.json';tmp.write_text(json.dumps(value));os.replace(tmp,d/'memory.latest.json')
   if value['memory']['limit'] and value['memory']['current']>.92*value['memory']['limit']:
    save(d/'fatal.json',dict(reason='own allocated cgroup memory >92%',**value));return
   finished.wait(3)
 threading.Thread(target=memory_watch,daemon=True).start()
 try:
  entries=[wait_for(root/f'node-{i}/hardware.json',180,root) for i in range(4)]
  head=json.loads((root/'node-0/entry-once.json').read_text());address=head['ip']+':'+str(p['ray_port'])
  cmd=[sys.executable,'-m','release.recursive_bran_a.h100_migration.ray_cli_compat','start','--node-ip-address='+json.loads((d/'entry-once.json').read_text())['ip'],'--num-gpus=8','--num-cpus='+str(64 if n==0 else 48),'--object-store-memory='+str(8*1024**3),'--disable-usage-stats','--block']
  if n==0:cmd+=['--head','--port='+str(p['ray_port']),'--include-dashboard=false','--temp-dir=/tmp/'+p['job']]
  else:
   wait_for(root/'ray-head-ready.json',180,root);cmd+=['--address='+address]
  with (d/'ray.log').open('xb') as log:rayproc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  if n==0:
   end=time.monotonic()+180
   while True:
    assert rayproc.poll() is None,'Owned Ray head exited'
    try:
     with socket.create_connection((head['ip'],p['ray_port']),timeout=2):break
    except OSError:assert time.monotonic()<end;time.sleep(2)
   save(root/'ray-head-ready.json',dict(at=time.time(),address=address))
   fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
   with (root/'training.log').open('xb') as log:trainproc=subprocess.Popen([sys.executable,'-m','reproduction.cgrts32_parallel_v1.train',str(pp)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=fds)
  end=time.monotonic()+p['run_timeout_seconds']
  while True:
   assert time.monotonic()<end,'New training bounded wall-time exceeded; preserve native state'
   assert rayproc.poll() is None,'Owned Ray node exited'
   assert not list(root.glob('node-*/fatal.json')),'Allocated node fatal; no automatic retry'
   if n==0 and trainproc.poll() is not None:
    rc=trainproc.returncode;save(root/'training-parent-exit.json',dict(at=time.time(),rc=rc));break
   if n!=0 and (root/'training-parent-exit.json').exists():rc=json.loads((root/'training-parent-exit.json').read_text())['rc'];break
   time.sleep(3)
  assert rc==0,'New32 training returned nonzero; preserve phase'
 finally:
  finished.set();stop_owned(trainproc);stop_owned(rayproc)
if __name__=='__main__':
 mode,pp=sys.argv[1:];p=read_plan(pp)
 try:(launch if mode=='launch' else serve)(pp)
 except BaseException as e:
  n=int(os.environ.get('NODE_RANK',-1));dest=Path(p['root'])/f'node-{n}/fatal.json'
  if dest.parent.exists() and not dest.exists():save(dest,dict(at=time.time(),reason=repr(e),no_auto_retry=True))
  raise
