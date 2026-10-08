"""Once-only immutable ALL + continuous training range; prepare/review before use."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,json,math,os,re,stat,sys,types
SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
RUN=ROOT/'runs/formal-3304';JOB='3304';LIMIT=512*1024*1024
FORMAL_SHA='ffb3bdc64f91c4a60905cd15994132c63c6494952c4ca0818c0cf7eee119d017'
RECEIPT_SHA='7042f5a93158541e9ff0044bf4c21849d8de16eed159c9787fdd7ae8406c564e'
GATE_SHA='09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'
ENV_SHA='b8f6264c59c88cd0efd89eb830da0ccc72d2115541c46d52a1a0890809ed16b2'
IDLE_SHA='cf42d42a54ba9940113a053db263639ab1683a910a5c182d10ac469f056f245b'
EXIT_SHA='6561cf2efc968809cb5df3a911ec07a7cb465120a6dfdec86593c4fa38abca39'
BOUNDARY_NS=1791287254032340562
HISTORICAL_BYTES=100995983
HISTORICAL_SHA='c0a6dccaec7502045c468278e05c6f0de1a69038188d35e5b7b986a75b86f0fb'
HISTORICAL_RESOURCE_DEVICE=64768
HISTORICAL_RESOURCE_INODE=138683997
ACTUAL_MANIFEST_SHA='30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577'
ACTUAL_NONCE='20bf4e0dddc9ac6eed8b0d1a9da656ad60d99e8d8bbf07f93ef8356abdfd5764'
ACTUAL_DRIVER=2548319
ACTUAL_DRIVER_START='1834784786'

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def unique(raw):
 def pairs(items):
  out={}
  for k,v in items:need(k not in out,'Duplicate JSON key');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def stat_record(st):return dict(device=st.st_dev,inode=st.st_ino,uid=st.st_uid,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
def read_physical(path,maximum,expected=None,prefix=False):
 p=Path(path);need(p.is_absolute() and p.resolve(strict=True)==p,'Physical input required')
 before=p.lstat();need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid(),'Owned regular input required')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd);need((opened.st_dev,opened.st_ino,opened.st_uid)==(before.st_dev,before.st_ino,before.st_uid),'Input replaced before open')
  need(0<opened.st_size<=maximum,'Input size outside fixed bound')
  raw=b'';chunks=[];remaining=opened.st_size
  while remaining:
   chunk=os.read(fd,min(1048576,remaining));need(bool(chunk),'Short original read');chunks.append(chunk);remaining-=len(chunk)
  raw=b''.join(chunks)
  if prefix:
   end=raw.rfind(b'\n')+1;need(end>0,'No complete NDJSON line');tail=raw[end:];raw=raw[:end]
   os.lseek(fd,0,os.SEEK_SET);digest=hashlib.sha256();remaining=len(raw)
   while remaining:
    chunk=os.read(fd,min(1048576,remaining));need(bool(chunk),'Short repeated prefix read');digest.update(chunk);remaining-=len(chunk)
   need(digest.hexdigest()==sha(raw),'Complete prefix changed during collection')
  else:tail=b''
  after=os.fstat(fd);path_after=p.lstat()
  if prefix:need((opened.st_dev,opened.st_ino,opened.st_uid)==(after.st_dev,after.st_ino,after.st_uid)==(path_after.st_dev,path_after.st_ino,path_after.st_uid) and min(after.st_size,path_after.st_size)>=opened.st_size,'Growing sampler identity changed')
  else:need(stat_record(opened)==stat_record(after)==stat_record(path_after),'Immutable input changed while read')
  if expected is not None:need(sha(raw)==expected,'Exact input SHA differs')
  return raw,dict(path=str(p),before=stat_record(before),opened=stat_record(opened),after=stat_record(after),path_after=stat_record(path_after),bytes=len(raw),sha256=sha(raw),unclosed_tail_bytes=len(tail),unclosed_tail_sha256=sha(tail))
 finally:os.close(fd)
def load_pinned(path,pin,name):
 raw,proof=read_physical(path,4*1024*1024,pin);module=types.ModuleType(name);module.__file__=str(path)
 exec(compile(raw,str(path),'exec',dont_inherit=True),module.__dict__);return module,proof

def row_time_ns(value):
 need(type(value) is str,'Resource UTC time string required');t=datetime.datetime.fromisoformat(value)
 need(t.tzinfo is not None and t.utcoffset()==datetime.timedelta(0),'Resource UTC required')
 delta=t-datetime.datetime(1970,1,1,tzinfo=datetime.timezone.utc)
 return (delta.days*86400+delta.seconds)*1000000000+delta.microseconds*1000

def finite_number(value):return type(value) in (int,float) and math.isfinite(value)

def phase_plan(raw,boundary_ns,historical_bytes,historical_sha,job=JOB):
 need(type(boundary_ns) is int and boundary_ns>0,'Typed actual boundary mtime required')
 need(type(historical_bytes) is int and 0<historical_bytes<=len(raw),'Historical complete prefix missing')
 need(raw.endswith(b'\n') and raw[:historical_bytes].endswith(b'\n') and sha(raw[:historical_bytes])==historical_sha,'Historical prefix drift')
 offset=0;first=None;previous=None;rows=0;problems=[];times=[];uuid_inventory=None
 for line in raw.splitlines(keepends=True):
  row=unique(line);need(row['job_id']==job,'Resource job differs');time=row_time_ns(row['time'])
  need(previous is None or time>previous,'Resource duplicate/backward time');previous=time
  if first is None and time>=boundary_ns:first=(offset,rows,row['time'])
  bad=[{k:a.get(k) for k in ('pid','process_start_ticks','uid','uuid','in_slurm_job','unavailable')} for a in row.get('gpu_processes',[]) if a.get('unavailable') or a.get('in_slurm_job') is not True or str(a.get('uid'))!=str(os.getuid())]
  inventory_errors=[];gpus=row.get('gpus',[]);current={g.get('uuid') for g in gpus}
  if len(gpus)!=8 or len(current)!=8 or None in current:inventory_errors.append('Not eight distinct GPU UUIDs')
  if uuid_inventory is not None and current!=uuid_inventory:inventory_errors.append('GPU UUID inventory changed')
  if len(gpus)==8 and len(current)==8 and None not in current:uuid_inventory=current
  if any(not finite_number(g.get('util')) or not 0<=g['util']<=100 or not finite_number(g.get('used_mib')) or g['used_mib']<0 for g in gpus):inventory_errors.append('Invalid GPU resource scalar')
  if bad or inventory_errors or 'query_error' in row:problems.append(dict(line_number=rows+1,offset=offset,time=row['time'],query_error=row.get('query_error'),inventory_errors=inventory_errors,applications=bad))
  times.append(time);offset+=len(line);rows+=1
 need(first is not None and first[1]<rows-1,'At least two complete training rows required')
 start,index,firsttime=first;slice_raw=raw[start:]
 training_problems=[p for p in problems if p['offset']>=start]
 need(not training_problems,'Training resource range contains unresolved/query-error rows')
 return slice_raw,dict(all_bytes=len(raw),all_sha256=sha(raw),all_lines=rows,all_problem_rows=problems,boundary_ns=boundary_ns,slice_start_offset=start,slice_end_offset=len(raw),slice_bytes=len(slice_raw),slice_sha256=sha(slice_raw),slice_first_line=index+1,slice_last_line=rows,slice_lines=rows-index,slice_first_time=firsttime,slice_last_time=unique(raw.splitlines()[-1])['time'],maximum_all_gap_seconds=max((b-a)/1e9 for a,b in zip(times,times[1:])),maximum_training_gap_seconds=max((b-a)/1e9 for a,b in zip(times[index:],times[index+1:])),historical_prefix=dict(bytes=historical_bytes,sha256=historical_sha),scope='Exact continuous original byte range from first sample time>=actual pre-Popen marker physical mtime; no filtering or normalization; whole-preflight raw failures retained')

def boundaries(env,envproof,idle,idleproof,exitrow,exitproof,source=SOURCE,run=RUN):
 for proof,inode in ((envproof,138684044),(idleproof,138684043),(exitproof,138684042)):
  r=proof['opened'];need((r['device'],r['inode'],r['uid'])==(64768,inode,0),'Original pretraining boundary file identity changed')
 need(envproof['opened']['mtime_ns']==BOUNDARY_NS and envproof['sha256']==ENV_SHA,'Actual startup marker identity differs')
 need(idleproof['sha256']==IDLE_SHA and exitproof['sha256']==EXIT_SHA and exitproof['opened']['mtime_ns']<=idleproof['opened']['mtime_ns']<=envproof['opened']['mtime_ns'],'Pretraining order differs')
 need(type(exitrow['returncode']) is int and exitrow['returncode']==0 and type(exitrow['rl_outer_updates']) is int and exitrow['rl_outer_updates']==0,'Actual capacity exit was not0/zeroRL')
 values=env['values'];need(values['SEARCH_R1_REPO']==str(ROOT/('repo-'+source[:12])) and values['SEARCH_R1_BASELINE_RUN_ROOT']==str(run) and values['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7','Actual pre-Popen environment differs')
 need(type(idle) is list and len(idle)==2,'Two original post-capacity idle samples required')
 uuids=None;owner=None
 for sample in idle:
  inventory=sample['inventory'];apps=sample['applications'];current={r['uuid'] for r in inventory}
  need(len(inventory)==8 and len(current)==8 and all(all(finite_number(r[k]) for k in ('used','total','util')) and 0<=r['used']<=.15*r['total'] and r['total']>0 and 0<=r['util']<=15 for r in inventory),'Actual doubleidle15% boundary did not pass')
  need(len(apps)==8 and {r['uuid'] for r in apps}==current and len({r['pid'] for r in apps})==1,'Post-capacity probe PID did not disappear')
  if uuids is not None:need(current==uuids,'Boundary UUID map differs')
  uuids=current
  for app in apps:
   r=app['owner'];need(type(r['pid']) is int and type(app['pid']) is int and app['pid']==r['pid'] and type(r['uid']) is int and r['uid']==os.getuid() and r['state'] in ('R','S','D','I') and type(r['start']) is str and r['start'].isdigit() and re.search(r'(?:^|/)job_'+JOB+r'(?:/|$)',r['cgroup'],re.M),'Boundary service ownership differs')
   if owner is not None:need((r['pid'],r['start'],r['uid'],r['cgroup'])==(owner['pid'],owner['start'],owner['uid'],owner['cgroup']),'Boundary service identity differs')
   owner=r
 return owner,sorted(uuids)

def context(args,reader,gate):
 raw,mproof=read_physical(args.manifest,4*1024*1024,args.manifest_sha256);manifest=unique(raw);reader.identity(manifest,RUN)
 need(manifest['job_id']==JOB and manifest['source_commit']==SOURCE and manifest['pause_nonce']==args.pause_nonce and type(manifest['driver_pid']) is int and manifest['driver_pid']==ACTUAL_DRIVER,'First5 actual context differs')
 readyraw,readyproof=read_physical(RUN/'approval-ready.json',4*1024*1024);ready=unique(readyraw)
 need(ready['source_commit']==SOURCE and ready['job_id']==JOB and ready['run_root']==str(RUN),'Controller context differs')
 driver=reader.actual_live_process(manifest['driver_pid'],manifest);controller=reader.actual_live_process(ready['pid'],manifest)
 need(driver['start']==args.driver_start and controller['start']==ready['process_start_ticks'],'Driver/controller PID reused')
 need(not (RUN/'acceptance-5-receipt.json').exists(),'First5 release already exists')
 reports=[]
 for rank in range(8):
  rec=manifest['files']['actor/training_state/rank_'+str(rank)+'.json'];b,p=read_physical(rec['path'],4*1024*1024,rec['sha256']);r=unique(b)
  need(type(r['rank']) is int and r['rank']==rank and type(r['step']) is int and r['step']==5 and type(r['world_size']) is int and r['world_size']==8 and r['status']=='EXACT_FULL_STATE_RELOAD_OK' and type(r['pid']) is int and r['pid']>0,'Actual rank5 metadata differs');reports.append(r)
 need(len({r['pid'] for r in reports})==8,'Eight distinct saved worker PIDs required')
 workers=[reader.actual_live_process(r['pid'],manifest) for r in reports]
 log,logproof=read_physical(RUN/'train.log',LIMIT,prefix=True)
 workerpids=tuple(r['pid'] for r in reports)
 paused=[];released=[];outers=[]
 for line in log.decode().splitlines(keepends=True):
  for marker,target in [('OPSD_E1_STEP5_PAUSED_JSON=',paused),('OPSD_E1_STEP5_RELEASED_JSON=',released),('OPSD_E1_OUTER_JSON=',outers)]:
   if marker in line:target.append(gate.parse_log_marker(line,marker,driver_pid=manifest['driver_pid'],job_id=JOB,worker_pids=workerpids))
 need(len(paused)==1 and not released and len(outers)==5 and [r['actual_outer_updates'] for r in outers]==[1,2,3,4,5],'Actual process is not uniquely paused after5')
 p=paused[0];need(p['manifest_path']==str(args.manifest) and p['manifest_sha256']==args.manifest_sha256 and p['pause_nonce']==args.pause_nonce and p['driver_pid']==manifest['driver_pid'] and p['job_id']==JOB,'PAUSED marker identity differs')
 return manifest,reports,dict(manifest=mproof,ready=readyproof,driver=driver,controller=controller,workers=workers,paused=p,log_prefix=logproof)

def exclusive_file(path,raw):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.chmod(path,0o444);need(path.read_bytes()==raw,'Published original bytes differ')

def sync_directory(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)

def reserve_once_directory(path):
 path.mkdir(mode=0o700)
 # Persist the once-only reservation before any member write. An unknown
 # failure may not erase the marker and permit a misleading second snapshot.
 sync_directory(path.parent);sync_directory(path)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--pause-nonce',required=True);parser.add_argument('--driver-start',required=True);args=parser.parse_args()
 need(os.getuid()==0 and args.manifest==RUN/'full_state/global_step_5/manifest.json' and re.fullmatch('[0-9a-f]{64}',args.manifest_sha256) and re.fullmatch('[0-9a-f]{64}',args.pause_nonce) and re.fullmatch('[0-9]+',args.driver_start),'Fixed actual3304 context arguments required')
 need(args.manifest_sha256==ACTUAL_MANIFEST_SHA and args.pause_nonce==ACTUAL_NONCE and args.driver_start==ACTUAL_DRIVER_START,'Only independently observed actual first5 context allowed')
 out=RUN/'first5-resource-training-phase-v1';need(not out.exists() and not out.is_symlink(),'Once output already exists; never overwrite or retry unknown outcome')
 reader,readerproof=load_pinned(ROOT/'tools/first5-1b16c2227caf/receipt_payload.py',RECEIPT_SHA,'frozen_phase_receipt');gate,gateproof=load_pinned(ROOT/'repo-fcecbea5d939/search_r1/baselines/checkpoint_gate.py',GATE_SHA,'frozen_phase_gate')
 formalraw,formalproof=read_physical(ROOT/'tools/formal-71d2501d4512/formal_supervisor.py',4*1024*1024,FORMAL_SHA)
 need(b"emit(run/'evidence/actual-training-wrapper-environment.json',environment())" in formalraw,'Actual frozen startup ordering marker absent')
 manifest,reports,before=context(args,reader,gate)
 envraw,envproof=read_physical(RUN/'evidence/actual-training-wrapper-environment.json',4*1024*1024,ENV_SHA);idleraw,idleproof=read_physical(RUN/'evidence/pre-training-double15-after-capacity.json',4*1024*1024,IDLE_SHA);exitraw,exitproof=read_physical(RUN/'evidence/capacity-micro4/exit.json',4*1024*1024,EXIT_SHA)
 service,uuids=boundaries(unique(envraw),envproof,unique(idleraw),idleproof,unique(exitraw),exitproof)
 allraw,resourceproof=read_physical(RUN/'resources.ndjson',LIMIT,prefix=True)
 need((resourceproof['opened']['device'],resourceproof['opened']['inode'],resourceproof['opened']['uid'])==(HISTORICAL_RESOURCE_DEVICE,HISTORICAL_RESOURCE_INODE,0),'Actual historical resource sampler inode changed')
 slice_raw,plan=phase_plan(allraw,envproof['opened']['mtime_ns'],HISTORICAL_BYTES,HISTORICAL_SHA)
 # Existing unmodified strict ownership/join runs on the complete training
 # range; it still rejects any unresolved,foreign,query-error training sample.
 reserve_once_directory(out)
 try:
  exclusive_file(out/'all-resources.ndjson',allraw);exclusive_file(out/'all-resources.ndjson.gz',gzip.compress(allraw,mtime=0));exclusive_file(out/'training-resources.ndjson',slice_raw);exclusive_file(out/'training-resources.ndjson.gz',gzip.compress(slice_raw,mtime=0))
  join=reader.resource_join(out/'training-resources.ndjson',manifest,reports,service)
  need(join['uuid_inventory']==uuids,'Training/boundary UUIDs differ')
  service_live=reader.actual_live_process(service['pid'],manifest)
  need(service_live['start']==service['start'],'Own retriever PID recycled')
  for worker in before['workers']:
   for sample in join['samples']:need(sample['identity'][worker['pid']]['start']==worker['start'],'Sampled/native live worker start differs')
  _,_,after=context(args,reader,gate)
  for key in ('manifest','ready'):
   need(before[key]['sha256']==after[key]['sha256'] and before[key]['opened']==after[key]['opened'],'Context input replaced during snapshot')
  for key in ('driver','controller'):need(before[key]['pid']==after[key]['pid'] and before[key]['start']==after[key]['start'],'Context process changed')
  need(before['workers']==after['workers'],'Live eight-worker identity changed')
  need(reader.actual_live_process(service['pid'],manifest)==service_live,'Live own service identity changed')
  for original in (readerproof,gateproof,formalproof,envproof,idleproof,exitproof):
   _,checked=read_physical(original['path'],4*1024*1024,original['sha256']);need(checked['opened']==original['opened'],'Immutable dependency replaced while preparing')
  runtime,runtime_proof=read_physical(Path(__file__).absolute(),4*1024*1024)
  report=dict(version=1,kind='IMMUTABLE_FIRST5_RESOURCE_TRAINING_CONTIGUOUS_RANGE_WITH_UNFILTERED_ALL',source_commit=SOURCE,job_id=JOB,driver_pid=manifest['driver_pid'],manifest_sha256=args.manifest_sha256,pause_nonce=args.pause_nonce,plan=plan,original_sampler=resourceproof,actual_boundary=dict(environment=envproof,post_capacity_idle=idleproof,capacity_exit=exitproof),before_context=before,after_context=after,peer_frozen_resource_join=join,dependencies=dict(receipt=readerproof,gate=gateproof,formal=formalproof),caller_provenance=dict(runtime=runtime_proof,argv=sys.argv,execution_pid=os.getpid(),execution_uid=os.getuid(),actual_environment_job_id=os.environ.get('SLURM_JOB_ID'),actual_environment_step_id=os.environ.get('SLURM_STEP_ID'),scope='Actual current caller observation; external approved exact-byte bootstrap and raw stdout/stderr/real exit must additionally be retained'),files={name:dict(path=str(out/name),sha256=sha((out/name).read_bytes()),bytes=(out/name).stat().st_size) for name in ['all-resources.ndjson','all-resources.ndjson.gz','training-resources.ndjson','training-resources.ndjson.gz']},scope='Training-range first5 resource join only; unfiltered preflight/all byte originals retained; frozen whole-run formal audit may still fail on historical pretraining cgroupFalse; not final terminal/release/effect PASS')
  exclusive_file(out/'manifest.json',(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());sync_directory(out);os.chmod(out,0o555);sync_directory(out);sync_directory(out.parent)
 except Exception:raise  # Retain partial outputs and actual failure; never cleanup/retry.
 print(json.dumps(dict(directory=str(out),manifest_sha256=sha((out/'manifest.json').read_bytes()),scope=report['scope'])))
if __name__=='__main__':main()
