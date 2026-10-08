import pathlib,sys,json,base64,hashlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
SRC=r'''
import pathlib,json,base64,hashlib,time,subprocess
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');ev=R/'logs/r1-standard32b-grpo100-common-v3-r2/evidence';out={'epoch':time.time(),'job':3218,'files':[],'errors':[],'old_cgroup_members':[]}
paths=[ev/n for n in ('train-result.json','owned-cleanup.json','effective-runtime-config.json','trainer-first-exception.json','trainer.log')]+[R/'logs/r1-32b-grpo-3218.log']
for p in paths+list(ev.glob('window-*.jsonl')):
 if not p.exists():continue
 a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  offset=max(0,a.st_size-1024*1024) if p.suffix=='.jsonl' else 0;f.seek(offset);b=f.read()
 z=p.stat();assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
 out['files'].append({'path':str(p),'original_sha256':h.hexdigest(),'original_size':a.st_size,'offset':offset,'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:cg=(p/'cgroup').read_text()
 except FileNotFoundError:continue
 if '/job_3218/' in cg or cg.rstrip().endswith('/job_3218'):out['old_cgroup_members'].append({'pid':int(p.name),'cgroup':cg})
r=subprocess.run(['scontrol','show','job','-dd','3218'],text=True,capture_output=True,timeout=5);out['controller']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
for query,name in (('index,uuid,memory.used,memory.total,utilization.gpu','gpu'),('gpu_uuid,pid,used_memory','compute-apps')):
 r=subprocess.run(['nvidia-smi','--query-'+name+'='+query,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);out[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
ck=ev.parent/'checkpoints/actor/global_step_5';out['step5']={'path':str(ck),'exists':ck.exists(),'files':[]}
if ck.exists():
 for q in sorted(ck.rglob('*')):
  if not q.is_file():continue
  a=q.stat();row={'path':str(q),'size':a.st_size,'mtime_epoch':a.st_mtime}
  if q.name.endswith('.index.json') or q.name=='config.json':
   b=q.read_bytes();row.update(sha256=hashlib.sha256(b).hexdigest(),data=json.loads(b))
  if q.suffix=='.safetensors':
   with q.open('rb') as h:
    n=int.from_bytes(h.read(8),'little');b=h.read(n)
   z=json.loads(b);row['header']={'size':n,'sha256':hashlib.sha256(b).hexdigest(),'tensor_count':len([k for k in z if k!='__metadata__'])}
  out['step5']['files'].append(row)
# Actual common manifests are recorded from this run configuration, not guessed.
c=json.loads((ev/'effective-runtime-config.json').read_text());out['configuration_common_refs']={k:v for k,v in c.items() if 'manifest' in k or 'support' in k}
print(json.dumps(out))
'''
r=send(SRC,attempts=3,retry_safe=True,timeout=90);r.check_returncode();d=json.loads(r.stdout);folder=pathlib.Path('outputs/remote-evidence/direct3218-small-'+str(int(d['epoch']*1e6)));folder.mkdir(parents=True,exist_ok=False)
for f in d['files']:
 b=base64.b64decode(f.pop('base64'));assert hashlib.sha256(b).hexdigest()==f['sha256'];name=pathlib.Path(f['path']).name+('.tail1m.raw' if f['offset'] else '');p=folder/name;p.write_bytes(b);f['local_path']=str(p.resolve())
(folder/'original-refs-and-physical.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'controller':d['controller'],'cg':d['old_cgroup_members'],'files':d['files'],'gpu':d['gpu'],'compute':d['compute-apps'],'commonrefs':d['configuration_common_refs'],'step5':d['step5'],'directory':str(folder)}))
