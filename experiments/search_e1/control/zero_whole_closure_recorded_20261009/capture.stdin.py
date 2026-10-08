import pathlib,json,hashlib,time,datetime,base64
N=pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008');S=N/'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs';Z=N/'eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/zero/runs'
def fp(path):
 path=pathlib.Path(path)
 try:
  a=path.stat();raw=path.read_bytes();b=path.stat()
 except FileNotFoundError:return dict(path=str(path),exists=False)
 keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns');assert all(getattr(a,k)==getattr(b,k) for k in keys),'Mutable capture during read'
 return dict(path=str(path),exists=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),metadata={k:getattr(b,k) for k in keys},raw_base64=base64.b64encode(raw).decode())
def proc(pid):
 d=pathlib.Path('/proc')/str(pid)
 try:
  a=(d/'stat').read_text();f=a[a.rfind(')')+2:].split();uid=[int(x) for x in next(x for x in (d/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1:]];argv=[x.decode() for x in (d/'cmdline').read_bytes().split(b'\0') if x];e=dict(x.split(b'=',1) for x in (d/'environ').read_bytes().split(b'\0') if b'=' in x);b=(d/'stat').read_text();ff=b[b.rfind(')')+2:].split();assert f[19]==ff[19],'PID changed'
  return dict(pid=pid,uid=uid[0],euid=uid[1],starttick=int(f[19]),state=ff[0],ppid=int(ff[1]),pgid=int(ff[2]),argv=argv,run_id=e.get(b'E1_RUN_ID',b'').decode(),cuda_visible_devices=e.get(b'CUDA_VISIBLE_DEVICES',b'').decode())
 except FileNotFoundError:return dict(pid=pid,state='ABSENT',actual_returncode_not_inferred=True)
files=[Z/'summary.json',Z/'controller-complete.json',Z/'fp16_zero/phase-complete.json',Z/'fp16_zero/actual-terminal.json',Z/'bf16_zero/phase-complete.json',S/'common-zero/evaluation/actual-child-exit.json',S/'common-zero/evaluation/actual-child-release.json',S/'common-zero/evaluation/result-accepted.json',S/'01-C1/training/child-start.json',S/'01-C1/training/actual-child-exit.json',S/'01-C1/training/group-terminal.json',N/'group-C1/runs/status.json']
result=dict(schema='opsd-common-zero-whole-and-c1-original-file-capture-v1',observed_unix=time.time(),observed_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),files=[fp(p) for p in files],driver=proc(1894801),zero_controller=proc(2009320),no_node_writes=True,no_gpu_or_model_test=True,no_score_recomputation=True)
for item in result['files']:
 if item['exists'] and item['path']==str(S/'01-C1/training/child-start.json'):
  start=json.loads(base64.b64decode(item['raw_base64']));ident=start.get('identity');result['actual_c1_child']=proc(ident['pid']) if ident else None
print(json.dumps(result,ensure_ascii=False))
