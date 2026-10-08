import pathlib,json,hashlib,base64,time,datetime,subprocess
N=pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008');B=N/'group-C1'
def fp(p):
 p=pathlib.Path(p)
 try:a=p.stat();raw=p.read_bytes();b=p.stat()
 except FileNotFoundError:return dict(path=str(p),exists=False)
 fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns');return dict(path=str(p),exists=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),metadata={k:getattr(b,k) for k in fields},stable_during_read=all(getattr(a,k)==getattr(b,k) for k in fields),raw_base64=base64.b64encode(raw).decode())
def obj(p):
 x=fp(p);return json.loads(base64.b64decode(x['raw_base64'])) if x['exists'] else None,x

def proc(pid):
 d=pathlib.Path('/proc')/str(pid)
 try:
  a=(d/'stat').read_text();f=a[a.rfind(')')+2:].split();uid=[int(x) for x in next(x for x in (d/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1:]];argv=[x.decode() for x in (d/'cmdline').read_bytes().split(b'\0') if x];env=dict(x.split(b'=',1) for x in (d/'environ').read_bytes().split(b'\0') if b'=' in x);b=(d/'stat').read_text();ff=b[b.rfind(')')+2:].split();assert f[19]==ff[19]
  return dict(pid=pid,uid=uid[0],euid=uid[1],starttick=int(f[19]),state=ff[0],ppid=int(ff[1]),pgid=int(ff[2]),argv=argv,run_id=env.get(b'E1_RUN_ID',b'').decode(),cuda_visible_devices=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode())
 except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,state='ABSENT',actual_returncode_not_inferred=True)
 except Exception as e:return dict(pid=pid,state='UNKNOWN',error_type=type(e).__name__,reason=str(e))
s,st=obj(B/'runs/status.json');i=s['outer_index'];O=B/'runs/outers'/('outer-%05d'%i);files=[st,fp(B/'receipts/memory-binding.json'),fp(B/'receipts/memory-live.json'),fp(B/'receipts/authorized-launch-20261009/config.json'),fp(O/'merged-model.json'),fp(O/'outer-manifest.json'),fp(O/'sampling-complete.json'),fp(O/'training/live-update.json'),fp(O/'training/outer-complete.json')]
if i>0:files.append(fp(B/'runs/outers'/('outer-%05d'%(i-1))/'training/outer-complete.json'))
workers=[]
for ident in s['known_children']:
 args=ident['argv']
 if any(a.endswith('/sampling.py') for a in args) and '--output' in args:
  p=pathlib.Path(args[args.index('--output')+1])
  if O not in p.parents:continue
  a=p.stat() if p.exists() else None;n=0;tail=b''
  if a:
   with p.open('rb') as h:
    while True:
     chunk=h.read(65536)
     if not chunk:break
     n+=chunk.count(b'\n');tail=chunk[-1:]
  b=p.stat() if a else None
  complete,cp=obj(str(p)+'.worker-complete.json');files.append(cp)
  workers.append(dict(rank=int(args[args.index('--rank')+1]),recorded_identity=ident,actual_process=proc(ident['pid']),rows=n,partial_tail=bool(tail and tail!=b'\n'),bytes=b.st_size if b else 0,stable_during_read=bool(a and all(getattr(a,k)==getattr(b,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))),native_completion={k:complete.get(k) for k in ['accepted','rows','pid','uid','rank','config_sha256','run_id']} if complete else None))
binding,_=obj(B/'receipts/memory-binding.json');observer=proc(binding['observer_identity']['pid']) if binding else None
queries=[];apps=[]
for args in [['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,memory.total','--format=csv,noheader,nounits'],['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']]:
 t=time.time();r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE);queries.append(dict(argv=args,started_unix=t,ended_unix=time.time(),actual_returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr).hexdigest()))
 if 'compute-apps' in args[1] and r.returncode==0:
  for line in r.stdout.decode().splitlines():
   p=[x.strip() for x in line.split(',')]
   if len(p)==3 and p[1].isdigit():apps.append(proc(int(p[1])))
print(json.dumps(dict(schema='opsd-c1-current-outer-small-readonly-native-snapshot-v1',observed_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),observed_unix=time.time(),files=files,driver=proc(1894801),controller=proc(s['owner']['pid']),observer=observer,phase=s['phase'],outer_index=i,status_fields={k:s.get(k) for k in ['optimizer_updates','actual_optimizer_updates','optimizer_updates_before','latest_sampling_model','latest_checkpoint']},current_sampling_workers=workers,direct_memory_queries=queries,actual_compute_processes=apps,no_node_writes=True,no_gpu_test=True,no_numeric_reader=True,no_resource_gate_scan=True),ensure_ascii=False))
