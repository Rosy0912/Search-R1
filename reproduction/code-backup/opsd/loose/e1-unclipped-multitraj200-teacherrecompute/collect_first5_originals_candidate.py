"""Read-only actual3304 PAUSED5 capsule; no CPU launch, signature, release or GPU."""
import argparse,base64,datetime,gzip,hashlib,io,json,os,re,shlex,stat,subprocess,sys,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006'
RUN=ROOT+'/runs/formal-3304';JOB='3304';REPO=ROOT+'/repo-'+SOURCE[:12]
BINDING_NAME='runtime-tools-bound-fcecbea5d939-v5.json'
BINDING_SHA='8cec4d887f656cecf7d533b91ac020da169ded4142a14890351501c9c76b2228'
CONTROLLER_SHA='8807429a3538165d38bc8ec8f5e3d09c5fa8cd91cdf8e7638631f7b2aa412944'
CONTROLLER_SCOPE='ACTUAL3304_CPU_FIRST5_ONCE_NOT_SIGNATURE_RELEASE_GPU_OR_EFFECT_ACCEPTANCE'
CONTROLLER_CONTRACT={'global_mini_batch':256,'optimizer_steps_per_rank_per_outer':10,'saved_adam_steps':50,'horizon':200,'warmup_steps':57,'base_learning_rate':1e-6}
NATIVE_WARNING_NAME='native-source/fully_sharded_data_parallel.py'
NATIVE_WARNING_SHA='ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
PROBE_COLLECTOR_SHA='ea9a1a77dee8bad4f2f657061eb883dc03e6233edf245276cb85860abf66fc62'
SCOPE='ACTUAL3304_PAUSED5_WHOLE_ARCHIVE_ORIGINAL_BYTES_NOT_ACCEPTANCE_RELEASE_OR_TERMINAL'
RAW_LIMIT=3*1024**3;GZIP_LIMIT=192*1024**2;WIRE_LIMIT=256*1024**2;SMALL_LIMIT=20*1024**2
CPU_PREFIX='evidence/first5-cpu-inspection-once-v1/'
CPU_NAMES=('launch-original.json','root-start-context.original.json','durable-launch.original.json',
 'supervisor-stdout.original','supervisor-stderr.original','supervisor-start.original.json',
 'controller-stdout.original','controller-stderr.original','controller-start.original.json',
 'controller-child-start.original.json','controller-exit.original.json',
 'srun-child-start.original.json','srun-start.original.json','srun-stdout.original','srun-stderr.original',
 'srun-exit.original.json','cpu-step-before-exec.original.json','frozen-inspector-command.original.json',
 'cpu-step-original.json','native-command.json','native-stdout-original.txt','native-stderr-original.txt',
 'native-exit-original.json','native.json','hf-command.json','hf-stdout-original.txt','hf-stderr-original.txt',
 'hf-exit-original.json','hf.json','inspection-completed-review-required.json')
ANCHOR_NAMES=('approval-ready.json','bindings/manifest.json','bindings/public.json')+tuple('bindings/'+r+'.json' for r in ('allocation','data_manifest','environment','import_closure','model_manifest','source','tool_manifest'))
REQUIRED=ANCHOR_NAMES+('full_state/global_step_5/manifest.json','train.log','evidence/service.json','evidence/http-top3-24.json')
OPTIONAL=('formal-wrapper.log','retriever.log')+tuple(CPU_PREFIX+n for n in CPU_NAMES)
TRAJECTORY_NAMES=('questions.json','student.jsonl','student-manifest.json','paired-manifest.json')+tuple('teacher-rank-'+str(r)+ext for r in range(8) for ext in ('.jsonl','.manifest.json'))
CHECKPOINT_MAP={
 'algorithm':'checkpoints/algorithm/global_step_5/state.json',
 'resolved_config':'full_state/global_step_5/resolved-config.json',
 'driver/driver.json':'full_state/global_step_5/driver.json','driver/driver.pt':'full_state/global_step_5/driver.pt',
 'launch_binding':'bindings/manifest.json','review_public_key':'bindings/public.json'}
CHECKPOINT_MAP.update({r:'bindings/'+r+'.json' for r in ('allocation','data_manifest','environment','import_closure','model_manifest','source','tool_manifest')})
CHECKPOINT_MAP.update({'actor/'+n:'checkpoints/actor/global_step_5/'+n for n in ('added_tokens.json','config.json','generation_config.json','merges.txt','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json')+tuple('model-0000'+str(r)+'-of-00003.safetensors' for r in range(1,4))})
CHECKPOINT_MAP.update({'actor/training_state/rank_'+str(r)+e:'checkpoints/actor/global_step_5/training_state/rank_'+str(r)+e for r in range(8) for e in ('.pt','.json')})
CHECKPOINT_MAP.update({'trajectory_outer_'+str(s).zfill(6)+'/'+n:'trajectories/outer_'+str(s).zfill(6)+'/'+n for s in range(1,6) for n in TRAJECTORY_NAMES})
TENSOR_NAMES=frozenset(n for n in CHECKPOINT_MAP.values() if n.endswith(('.pt','.safetensors')))
TORCH_PATH='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch'

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def unique(raw):
 def pairs(items):
  out={}
  for k,v in items:need(k not in out,'Duplicate JSON key');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def pin(v):need(type(v)is str and re.fullmatch('[0-9a-f]{64}',v) is not None,'Exact observed SHA required');return v

def cpu_status(bodies,manifest,request):
 """Status of saved original CPU outputs, never independent acceptance."""
 def get(n):return unique(bodies[CPU_PREFIX+n])
 exits=('native-exit-original.json','hf-exit-original.json','srun-exit.original.json','controller-exit.original.json')
 for n in exits:
  if CPU_PREFIX+n in bodies:
   e=get(n);need(type(e.get('returncode'))is int,'CPU exit must be actual integer')
   if e['returncode']!=0:return {'status':'CPU_PIPELINE_RECORDED_NONZERO_ORIGINALS','exit_member':n,'returncode':e['returncode']}
 complete=exits+('native.json','hf.json','inspection-completed-review-required.json','cpu-step-original.json','cpu-step-before-exec.original.json','launch-original.json','controller-child-start.original.json','controller-start.original.json','srun-child-start.original.json')
 missing=[n for n in complete if CPU_PREFIX+n not in bodies]
 if missing:return {'status':'CPU_PIPELINE_PENDING_ORIGINALS_NOT_COMPLETION','missing_completion_members':missing}
 launch=get('launch-original.json');need(launch['controller_sha256']==CONTROLLER_SHA and launch['scope']==CONTROLLER_SCOPE and json.dumps(launch['contract'],sort_keys=True,separators=(',',':'))==json.dumps(CONTROLLER_CONTRACT,sort_keys=True,separators=(',',':')) and launch['output']==RUN+'/'+CPU_PREFIX.rstrip('/'),'CPU current controller version/contract/path differs');need(launch['job_id']==JOB and launch['source_commit']==SOURCE and launch['checkpoint_manifest_sha256']==request['checkpoint_manifest_sha256'] and launch['pause_nonce']==manifest['pause_nonce'] and launch['driver_pid']==manifest['driver_pid'] and launch['driver_start_ticks']==request['driver_start_ticks'],'CPU launch actual pause identity differs')
 for n in ('srun-exit.original.json','controller-exit.original.json'):
  e=get(n);need(e.get('waited')is True and type(e.get('waited_by_pid'))is int and e['waited_by_pid']>0 and type(e.get('child_pid'))is int and e['child_pid']>0,'CPU waited exit identity missing')
 for n,s in (('native.json','NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE'),('hf.json','ACTUAL_FULL_CPU_HF5_RELOAD_NO_CUDA')):
  e=get(n);need(e['scope']==s and e['source_commit']==SOURCE and e['job_id']==JOB and e['run_root']==RUN and e['checkpoint_manifest_sha256']==request['checkpoint_manifest_sha256'] and e.get('cuda_is_initialized')is False and e.get('passed')is True,'CPU report source/checkpoint/scope differs')
 done=get('inspection-completed-review-required.json');need(done['scope']=='CPU_STORED_STATE_AND_HF_INSPECTION_ONLY_NOT_SIGNATURE_OR_RELEASE' and done['source_commit']==SOURCE and done['job_id']==JOB and done['manifest_sha256']==request['checkpoint_manifest_sha256'] and done['cuda_is_initialized']is False,'CPU completion marker identity differs')
 need(type(done['files'])is dict and {'native.json','hf.json','native-exit-original.json','hf-exit-original.json','cpu-step-original.json'}<=set(done['files']),'CPU completed marker must pin actual reports and exit originals')
 for n,p in done['files'].items():
  need(n in CPU_NAMES and CPU_PREFIX+n in bodies and sha(bodies[CPU_PREFIX+n])==pin(p),'CPU completed marker member SHA differs')
 step=get('cpu-step-original.json');before=get('cpu-step-before-exec.original.json');need(step['job_id']==JOB and type(step['uid'])is int and step['uid']==0 and step['hostname'].split('.')[0]=='h100-10' and step['cuda_visible_devices']=='' and step['cuda_is_initialized']is False and step['manifest_sha256']==request['checkpoint_manifest_sha256'],'CPU saved execution identity differs')
 proc=before['process'];need(before['job_id']==JOB and before['cuda_visible_devices']=='' and before['omp_num_threads']=='1' and type(step['pid'])is int and step['pid']>0 and proc['pid']==step['pid'] and type(proc['uid'])is int and proc['uid']==0 and proc['cgroup']==step['cgroup'] and re.fullmatch('[0-9]+',before['step_id'])is not None and re.search(r'(?:^|/)job_3304(?:/|$)',step['cgroup'],re.M) and re.search(r'(?:^|/)step_'+re.escape(before['step_id'])+r'(?:\.scope)?(?:/|$)',step['cgroup'],re.M),'CPU exec/srun cgroup identity differs')
 controller=get('controller-start.original.json')['process'];parent=get('controller-child-start.original.json');se=get('srun-exit.original.json');ce=get('controller-exit.original.json');sp=get('srun-child-start.original.json')
 need(parent['child_pid']==controller['pid']==ce['child_pid']==se['waited_by_pid'] and parent['child']['pid']==controller['pid'] and parent['child']['start_ticks']==controller['start_ticks'] and sp['child_pid']==se['child_pid'] and sp['parent']['pid']==controller['pid'],'CPU parent/child waited identity differs')
 return {'status':'CPU_PIPELINE_WAITED_COMPLETED_ORIGINALS_REVIEW_REQUIRED','scope':'Preserved native/HF/true-wait originals; not first5 acceptance, signature or release'}

REMOTE_LIBRARY=r'''
from pathlib import Path
import os,stat,json,hashlib,gzip,base64,datetime,socket,io,re,types,subprocess

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def unique(raw):
 def pairs(items):
  d={}
  for k,v in items:need(k not in d,'Duplicate JSON key');d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def sig(s):return [s.st_dev,s.st_ino,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def budget(raw_total,gzip_total):
 need(type(raw_total)is int and type(gzip_total)is int and 0<=raw_total<=RAW_LIMIT and 0<=gzip_total<=GZIP_LIMIT,'Whole capsule engineering bound exceeded; no sampling')
def physical(p,uid=0,source_hardlink_pin=None):
 need(p.is_absolute() and p.resolve(strict=True)==p,'Nonphysical original')
 a=p.lstat();links=1
 if source_hardlink_pin is not None:
  need(source_hardlink_pin==NATIVE_WARNING_SHA and str(p)==TORCH_PATH+'/distributed/fsdp/fully_sharded_data_parallel.py' and a.st_size==99345,'Only exact observed installed third-party source may have two links');links=2
 need(stat.S_ISREG(a.st_mode) and a.st_uid==uid and a.st_nlink==links,'Owned original must have exact permitted link count');return a

def member(p,prefix=False,expected=None,metadata=False,max_size=SMALL_LIMIT,uid=0,source_hardlink_pin=None):
 if source_hardlink_pin is not None:need(not prefix and not metadata and expected==NATIVE_WARNING_SHA==source_hardlink_pin,'Third-party linked source must be immutable and full-SHA verified')
 a=physical(p,uid,source_hardlink_pin);need(metadata or a.st_size<=max_size,'Original exceeds member bound; no sampling')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  z=os.fstat(fd);need(sig(a)==sig(z) and a.st_nlink==z.st_nlink,'Original replaced before open')
  if metadata:
   f=os.fstat(fd);b=p.lstat();need(sig(z)==sig(f)==sig(b) and z.st_nlink==f.st_nlink==b.st_nlink,'Tensor metadata changed')
   return dict(physical=str(p),bytes=z.st_size,uid=z.st_uid,device=z.st_dev,inode=z.st_ino,stat_before=sig(z),stat_after=sig(b),reading_scope='METADATA_ONLY_DECLARED_CHECKPOINT_SHA_NOT_FRESH_FULL_SHA',declared_sha256=expected,downloaded=False,link_count=z.st_nlink,link_count_after=b.st_nlink)
  h=hashlib.sha256();n=0;lines=0;last=b'';buf=io.BytesIO()
  with gzip.GzipFile(fileobj=buf,mode='wb',compresslevel=1,mtime=0) as gz:
   left=z.st_size
   while left:
    raw=os.read(fd,min(1024**2,left));need(bool(raw),'Original shortened');left-=len(raw);n+=len(raw);h.update(raw);gz.write(raw);lines+=raw.count(b'\n');last=raw[-1:];need(buf.tell()<=GZIP_LIMIT,'Compressed original exceeds capsule bound')
  packed=buf.getvalue();need(len(packed)<=GZIP_LIMIT,'Compressed original exceeds capsule bound')
  after=os.fstat(fd);b=p.lstat();need(z.st_nlink==after.st_nlink==b.st_nlink,'Original link count changed');need(z.st_dev==after.st_dev==b.st_dev and z.st_ino==after.st_ino==b.st_ino and z.st_uid==after.st_uid==b.st_uid==uid,'Original physical identity changed')
  if prefix:need(after.st_size>=z.st_size and b.st_size>=z.st_size and after.st_size<=max_size and b.st_size<=max_size,'Live prefix truncated/exceeded bound')
  else:need(sig(z)==sig(after)==sig(b),'Immutable original changed')
  os.lseek(fd,0,os.SEEK_SET);again=hashlib.sha256();left=z.st_size
  while left:
   raw=os.read(fd,min(1024**2,left));need(bool(raw),'Original shortened on second read');left-=len(raw);again.update(raw)
  c=os.fstat(fd);d=p.lstat();need(z.st_nlink==c.st_nlink==d.st_nlink,'Original link count changed on re-read');need(c.st_dev==d.st_dev==z.st_dev and c.st_ino==d.st_ino==z.st_ino and c.st_uid==d.st_uid==uid and again.hexdigest()==h.hexdigest(),'Same-FD initial bytes overwritten/replaced')
  if prefix:need(c.st_size>=z.st_size and d.st_size>=z.st_size and c.st_size<=max_size and d.st_size<=max_size,'Live prefix changed length/bound')
  else:need(sig(z)==sig(c)==sig(d),'Immutable original changed after re-read')
  if expected is not None:need(h.hexdigest()==expected,'Declared raw original SHA differs')
  return dict(physical=str(p),bytes=n,sha256=h.hexdigest(),gzip_bytes=len(packed),gzip_sha256=sha(packed),gzip_base64=base64.b64encode(packed).decode(),uid=z.st_uid,device=z.st_dev,inode=z.st_ino,stat_before=sig(z),stat_after=sig(d),reading_scope=('VERIFIED_INITIAL_LENGTH_PREFIX_POSSIBLY_LIVE_NOT_CLOSED' if prefix else 'STABLE_FULL_BYTES_AT_OBSERVATION_NOT_ACCEPTANCE'),line_count=lines,ends_lf=last==b'\n',link_count=z.st_nlink,link_count_after=d.st_nlink,third_party_hardlink_source_pin=source_hardlink_pin)
 finally:os.close(fd)

def raw_of(row):
 packed=base64.b64decode(row['gzip_base64'],validate=True);need(len(packed)==row['gzip_bytes'] and sha(packed)==row['gzip_sha256'],'Internal compressed SHA changed')
 raw=gzip.decompress(packed);need(len(raw)==row['bytes'] and sha(raw)==row['sha256'],'Internal raw SHA changed');return raw

def process(pid,start):
 p=Path('/proc')/str(pid);a=(p/'stat').read_text().rsplit(')',1)[1].split()[19];st=(p/'status').read_text();cg=(p/'cgroup').read_text();b=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
 need(a==b==start and set(map(int,re.search(r'^Uid:\s+(.*)',st,re.M).group(1).split()))=={0} and re.search(r'^State:\s+(\S+)',st,re.M).group(1) in ('R','S','D','I') and re.search(r'(?:^|/)job_3304(?:/|$)',cg,re.M),'Actual paused driver no longer original live owned process')
 out=p/'fd/1';target=out.resolve(strict=True);physical(target)
 return dict(pid=pid,start_ticks=start,uid=0,cgroup=cg,status_original=st,stdout_physical=str(target))

def check_manifest(m,request):
 need(all(type(m[k])is int for k in ('version','actual_outer_updates','global_step','next_global_step','horizon','warmup_steps','world_size','driver_pid','uid')),'Checkpoint counter types differ')
 need((m['version'],m['actual_outer_updates'],m['global_step'],m['next_global_step'],m['horizon'],m['warmup_steps'],m['world_size'],m['uid'])==(1,5,5,6,200,57,8,0) and m['driver_pid']==request['driver_pid'] and m['job_id']=='3304' and m['source_commit']==SOURCE and m['run_root']==RUN and m['method']=='pure_opsd' and m['pause_nonce']==request['pause_nonce'],'Wrong actual PAUSED5 checkpoint')
 need(set(m['files'])==set(CHECKPOINT_MAP),'Checkpoint member schema differs')
 for role,name in CHECKPOINT_MAP.items():
  r=m['files'][role];need(set(r)=={'path','sha256'} and r['path']==RUN+'/'+name and type(r['sha256'])is str and re.fullmatch('[0-9a-f]{64}',r['sha256']),'Checkpoint role path/SHA differs')

def check_logs(raw,g,manifest,request,workers,fsdp_source=None):
 pauses=[];outers=[]
 for line in raw.decode().splitlines(keepends=True):
  if not line.endswith('\n'):continue
  for marker in ('OPSD_E1_OUTER_JSON=','OPSD_E1_STEP5_PAUSED_JSON=','OPSD_E1_STEP5_RELEASED_JSON='):
   if marker in line:
    value=g.parse_log_marker(line.rstrip('\n'),marker,driver_pid=manifest['driver_pid'],job_id='3304',worker_pids=workers,checkpoint_warning_path=Path(TORCH_PATH)/'utils/checkpoint.py',fsdp_warning_source_bytes=fsdp_source)
    if marker=='OPSD_E1_STEP5_RELEASED_JSON=':raise ValueError('Training already released; PAUSED5 capsule scope invalid')
    if marker=='OPSD_E1_STEP5_PAUSED_JSON=':pauses.append(value)
    else:
     need(value['driver_pid']==manifest['driver_pid'] and value['job_id']==JOB and value['source_commit']==SOURCE and value['config_sha256']==manifest['config_sha256'] and type(value['actual_outer_updates'])is int,'Outer actual identity differs');g.validate_outer(value['metrics'],world=8,profile='multitraj200');outers.append(value)
 need(len(pauses)==1 and [x['actual_outer_updates'] for x in outers]==list(range(1,6)),'Actual complete first5 outer/pause prefix required')
 p=pauses[0];need(p['manifest_path']==RUN+'/full_state/global_step_5/manifest.json' and p['manifest_sha256']==request['checkpoint_manifest_sha256'] and p['pause_nonce']==manifest['pause_nonce'],'Actual PAUSED marker checkpoint differs')
 return dict(outer_updates=[x['actual_outer_updates'] for x in outers],paused=p)

def collect(request):
 need(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','Own UID0 native node reader required')
 root=Path(RUN);need(root.resolve(strict=True)==root and root.stat().st_uid==0,'Physical own run required')
 need(not os.path.lexists(root/'acceptance-5-receipt.json'),'Already released/signed receipt present')
 rows={};metadata={};used_raw=0;used_gzip=0
 def add(name,p=None,expected=None,prefix=False):
  nonlocal used_raw,used_gzip
  if name in rows:
   need(rows[name]['physical']==str(p or root/name) and (expected is None or rows[name]['sha256']==expected),'Aliased repeated member differs');return
  p=p or root/name;linked_pin=NATIVE_WARNING_SHA if name==NATIVE_WARNING_NAME else None;a=physical(p,source_hardlink_pin=linked_pin);budget(used_raw+a.st_size,used_gzip)
  r=member(p,prefix=prefix,expected=expected,max_size=(RAW_LIMIT if p.suffix=='.jsonl' else SMALL_LIMIT),source_hardlink_pin=linked_pin);used_raw+=r['bytes'];used_gzip+=r['gzip_bytes'];budget(used_raw,used_gzip);rows[name]=r
 for name in REQUIRED:add(name,expected=(request['checkpoint_manifest_sha256'] if name=='full_state/global_step_5/manifest.json' else None),prefix=name=='train.log')
 m=unique(raw_of(rows['full_state/global_step_5/manifest.json']));check_manifest(m,request)
 ready=unique(raw_of(rows['approval-ready.json']));bound=unique(raw_of(rows['bindings/manifest.json']));src=unique(raw_of(rows['bindings/source.json']))
 need(ready['source_commit']==SOURCE and ready['job_id']==JOB and ready['run_root']==RUN and ready['binding_sha256']==rows['bindings/manifest.json']['sha256'] and ready['public_key_sha256']==rows['bindings/public.json']['sha256'],'Actual ready/run binding differs')
 need(bound['source_commit']==SOURCE and bound['run_root']==RUN and set(bound['files'])=={'allocation','data_manifest','environment','import_closure','model_manifest','source','tool_manifest'},'Actual run binding roles differ')
 for role,r in bound['files'].items():need(r['path']==RUN+'/bindings/'+role+'.json' and r['sha256']==rows['bindings/'+role+'.json']['sha256'],'Actual run binding member SHA differs')
 need(src['clean']is True and src['source_commit']==SOURCE and src['repo']==REPO and src['source_files']==request['source_files'] and src['formal_manifest_sha256']==request['formal_sha'],'Actual original source binding differs')
 source_pins={}
 for name,p in request['source_files'].items():
  add('source/'+name,Path(REPO)/name,expected=p);source_pins[name]=p
 add(NATIVE_WARNING_NAME,Path(TORCH_PATH)/'distributed/fsdp/fully_sharded_data_parallel.py',expected=NATIVE_WARNING_SHA);fsdp_source=raw_of(rows[NATIVE_WARNING_NAME])
 state=['git','-C',REPO,'status','--porcelain'];head=['git','-C',REPO,'rev-parse','HEAD']
 need(subprocess.check_output(head,text=True).strip()==SOURCE and not subprocess.check_output(state,text=True).strip(),'Actual source repo changed')
 gate_raw=raw_of(rows['source/search_r1/baselines/checkpoint_gate.py']);g=types.ModuleType('frozen_gate');g.__file__=REPO+'/search_r1/baselines/checkpoint_gate.py';exec(compile(gate_raw,g.__file__,'exec',dont_inherit=True),g.__dict__)
 for role,name in CHECKPOINT_MAP.items():
  rec=m['files'][role]
  if name in TENSOR_NAMES:metadata[name]=member(root/name,expected=rec['sha256'],metadata=True)
  else:add(name,expected=rec['sha256'])
 workers=[unique(raw_of(rows['checkpoints/actor/global_step_5/training_state/rank_'+str(r)+'.json']))['pid'] for r in range(8)]
 need(all(type(p)is int and p>0 for p in workers) and len(set(workers))==8,'Actual distinct eight worker report PIDs required')
 before=process(m['driver_pid'],request['driver_start_ticks']);add('driver-current-prefix.out',Path(before['stdout_physical']),prefix=True)
 train_join=check_logs(raw_of(rows['train.log']),g,m,request,workers,fsdp_source);driver_join=check_logs(raw_of(rows['driver-current-prefix.out']),g,m,request,workers,fsdp_source)
 missing=[]
 for name in OPTIONAL:
  p=root/name
  if os.path.lexists(p):add(name,prefix=(p.suffix=='.log' or 'stdout' in p.name or 'stderr' in p.name))
  else:missing.append(name)
 for step in range(1,6):
  base='trajectories/outer_'+str(step).zfill(6)+'/'
  q=unique(raw_of(rows[base+'questions.json']));paired=unique(raw_of(rows[base+'paired-manifest.json']));student=unique(raw_of(rows[base+'student-manifest.json']))
  need(type(q['questions'])is list and len(q['questions'])==512 and paired['student_count']==2560 and paired['teacher_count']==512 and student['student_count']==2560 and student['recorded_before_balancing']is True,'Archive declared complete counts differ')
  for n in TRAJECTORY_NAMES:
   if n.endswith('.jsonl'):
    expected=2560 if n=='student.jsonl' else 64;need(rows[base+n]['line_count']==expected and rows[base+n]['ends_lf']is True,'Full archive line cardinality differs; no sampling')
 after=process(m['driver_pid'],request['driver_start_ticks']);need(all(before[k]==after[k] for k in ('pid','start_ticks','uid','cgroup','stdout_physical')),'Original driver identity/stdout changed')
 need(not os.path.lexists(root/'acceptance-5-receipt.json'),'Release receipt appeared during collection')
 # Re-read actual initial prefixes and source bytes; never mutate or normalize them.
 for name in ('train.log','driver-current-prefix.out'):
  fresh=member(Path(rows[name]['physical']),prefix=True,max_size=SMALL_LIMIT);check_logs(raw_of(fresh),g,m,request,workers,fsdp_source)
 need(subprocess.check_output(head,text=True).strip()==SOURCE and not subprocess.check_output(state,text=True).strip(),'Actual source drifted during collection')
 for name,p in request['source_files'].items():need(member(Path(REPO)/name,expected=p)['sha256']==p,'Final source bytes changed')
 need(member(root/'full_state/global_step_5/manifest.json',expected=request['checkpoint_manifest_sha256'])['sha256']==request['checkpoint_manifest_sha256'],'Checkpoint manifest changed')
 return dict(version=1,kind=SCOPE,scope=SCOPE,job_id=JOB,source_commit=SOURCE,run_root=RUN,uid=0,hostname=socket.gethostname(),observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,tensor_metadata=metadata,missing_optional=missing,driver_before=before,driver_after=after,train_prefix_join=train_join,driver_prefix_join=driver_join,full_archive_members=100,raw_bytes=used_raw,gzip_bytes=used_gzip,resources_scope='EXCLUDED_SEPARATE_RESOURCE_REVIEW',tensor_scope='METADATA_ONLY_NO_FRESH_TENSOR_SHA_OR_LOAD_CLAIM')
'''

def local_owned_raw(p):
 need(p.is_absolute() and p.resolve(strict=True)==p,'Local pinned path must be physical');a=p.lstat();need(stat.S_ISREG(a.st_mode) and a.st_uid==os.getuid() and a.st_nlink==1 and a.st_size<=SMALL_LIMIT,'Local pin ownership/type/bound differs');fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  z=os.fstat(fd);raw=b''
  while True:
   b=os.read(fd,1024**2)
   if not b:break
   raw+=b
  f=lambda q:(q.st_dev,q.st_ino,q.st_uid,q.st_size,q.st_mtime_ns,q.st_ctime_ns)
  need(f(a)==f(z)==f(os.fstat(fd))==f(p.lstat()) and len(raw)==z.st_size,'Local pinned bytes changed');return raw
 finally:os.close(fd)

def local_request(a):
 # Reuse the unchanged reviewed V5 actual positive-submission join, from same opened bytes.
 p=HERE/'collect_probe_originals.py';raw=local_owned_raw(p);need(sha(raw)==PROBE_COLLECTOR_SHA,'Pinned local submission helper changed')
 ns={'__file__':str(p),'__name__':'first5_originals_submission_join'};exec(compile(raw,str(p),'exec',dont_inherit=True),ns)
 request=ns['local_request'](a.binding);need(request['job_id']==JOB,'Only actual submitted job3304 allowed')
 need(type(a.driver_pid)is int and a.driver_pid>0 and type(a.driver_start_ticks)is str and re.fullmatch('[1-9][0-9]*',a.driver_start_ticks),'Actual original driver identity required')
 request.update(checkpoint_manifest_sha256=pin(a.checkpoint_manifest_sha256),pause_nonce=pin(a.pause_nonce),driver_pid=a.driver_pid,driver_start_ticks=a.driver_start_ticks)
 return request

def reader_constants():
 names=('SOURCE','ROOT','RUN','JOB','REPO','SCOPE','RAW_LIMIT','GZIP_LIMIT','WIRE_LIMIT','SMALL_LIMIT','CPU_PREFIX','CPU_NAMES','REQUIRED','OPTIONAL','TRAJECTORY_NAMES','CHECKPOINT_MAP','TENSOR_NAMES','TORCH_PATH','NATIVE_WARNING_NAME','NATIVE_WARNING_SHA')
 return '\n'.join(k+'='+repr(globals()[k]) for k in names)+'\n'

def remote_program(request):
 return reader_constants()+REMOTE_LIBRARY+'\npacket=json.dumps(collect('+repr(request)+'),sort_keys=True).encode();need(len(packet)<=WIRE_LIMIT,\"Whole wire exceeds256MiB; no sampling\");print(packet.decode())\n'

def decompressed(row):
 need(type(row['bytes'])is int and 0<=row['bytes']<=RAW_LIMIT and type(row['gzip_bytes'])is int and 0<=row['gzip_bytes']<=GZIP_LIMIT,'Typed bounded raw/compressed lengths required')
 packed=base64.b64decode(row['gzip_base64'],validate=True);need(len(packed)==row['gzip_bytes'] and sha(packed)==pin(row['gzip_sha256']),'Compressed SHA/length differs')
 d=zlib.decompressobj(16+zlib.MAX_WBITS);raw=d.decompress(packed,row['bytes']+1)
 need(len(raw)==row['bytes'] and d.eof and not d.unused_data and not d.unconsumed_tail and sha(raw)==pin(row['sha256']),'Bounded complete gzip/raw SHA differs')
 return raw

def save_streamed_member(row,target):
 """Bounded single-file gzip decode directly to O_EXCL raw output, preserving failures."""
 need(type(row['bytes'])is int and 0<=row['bytes']<=RAW_LIMIT and type(row['gzip_bytes'])is int and 0<=row['gzip_bytes']<=GZIP_LIMIT,'Typed bounded raw/compressed lengths required')
 packed=base64.b64decode(row['gzip_base64'],validate=True);need(len(packed)==row['gzip_bytes'] and sha(packed)==pin(row['gzip_sha256']),'Compressed SHA/length differs')
 d=zlib.decompressobj(16+zlib.MAX_WBITS);h=hashlib.sha256();n=0;lines=0;last=b''
 with target.open('xb')as f:
  for offset in range(0,len(packed),65536):
   pending=packed[offset:offset+65536]
   while pending:
    raw=d.decompress(pending,min(1024**2,row['bytes']-n+1));pending=d.unconsumed_tail
    need(not d.unused_data and n+len(raw)<=row['bytes'],'Gzip extra stream/raw bound differs')
    if raw:f.write(raw);h.update(raw);n+=len(raw);lines+=raw.count(b'\n');last=raw[-1:]
  need(d.eof and not d.unused_data and not d.unconsumed_tail and n==row['bytes'] and h.hexdigest()==pin(row['sha256']),'Complete streamed gzip/raw SHA differs')
  f.flush();os.fsync(f.fileno())
 fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return {'bytes':n,'sha256':h.hexdigest(),'line_count':lines,'ends_lf':last==b'\n'}

def validate_row(name,row,request,metadata=False):
 need(type(row['bytes'])is int and row['bytes']>=0,'Typed original byte length required')
 links=2 if name==NATIVE_WARNING_NAME else 1
 need(type(row.get('link_count'))is int and type(row.get('link_count_after'))is int and row['link_count']==row['link_count_after']==links,'Exact original link count differs')
 if name==NATIVE_WARNING_NAME:need(row.get('third_party_hardlink_source_pin')==NATIVE_WARNING_SHA and row.get('sha256')==NATIVE_WARNING_SHA and row['bytes']==99345,'Exact linked installed-source SHA/size differs')
 else:need(row.get('third_party_hardlink_source_pin')is None,'Unexpected linked-source exception')
 expected=(request['repo']+'/'+name[7:] if name.startswith('source/') else request['run_root']+'/'+name)
 if name=='driver-current-prefix.out':expected=request['driver_stdout_physical']
 if name==NATIVE_WARNING_NAME:expected=TORCH_PATH+'/distributed/fsdp/fully_sharded_data_parallel.py'
 need(row['physical']==expected and type(row['uid'])is int and row['uid']==0 and type(row['device'])is int and row['device']>=0 and type(row['inode'])is int and row['inode']>0,'Original physical path/UID identity differs')
 a,b=row['stat_before'],row['stat_after'];need(type(a)is list and type(b)is list and len(a)==len(b)==6 and all(type(v)is int for v in a+b) and a[:3]==b[:3]==[row['device'],row['inode'],0] and a[3]==row['bytes'],'Original physical stat types/length differ')
 if metadata:
  need(name in TENSOR_NAMES and a==b and row['downloaded']is False and row['reading_scope']=='METADATA_ONLY_DECLARED_CHECKPOINT_SHA_NOT_FRESH_FULL_SHA' and 'gzip_base64' not in row,'Tensor must stay metadata-only');pin(row['declared_sha256']);return
 prefix=name=='driver-current-prefix.out' or name=='train.log' or name.endswith('.log') or 'stdout' in Path(name).name or 'stderr' in Path(name).name
 need(row['reading_scope']==('VERIFIED_INITIAL_LENGTH_PREFIX_POSSIBLY_LIVE_NOT_CLOSED' if prefix else 'STABLE_FULL_BYTES_AT_OBSERVATION_NOT_ACCEPTANCE') and (a[3]<=b[3]<=SMALL_LIMIT if prefix else a==b),'Original stability/scope differs')


def write_once(p,raw):
 with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def save_snapshot(wire,request,out):
 need(type(wire)is bytes and len(wire)<=WIRE_LIMIT,'Wire bound exceeded');r=unique(wire);need(r['kind']==r['scope']==SCOPE and type(r['version'])is int and r['version']==1 and r['job_id']==JOB and r['source_commit']==SOURCE and r['run_root']==RUN and type(r['uid'])is int and r['uid']==0 and r['hostname'].split('.')[0]=='h100-10','Capsule identity/scope differs')
 rows=r['files'];metas=r['tensor_metadata'];source_names={'source/'+n for n in request['source_files']};required=set(REQUIRED)|(set(CHECKPOINT_MAP.values())-TENSOR_NAMES)|source_names|{'driver-current-prefix.out',NATIVE_WARNING_NAME}
 need(type(rows)is dict and required<=set(rows)<=required|set(OPTIONAL) and type(metas)is dict and set(metas)==TENSOR_NAMES,'Capsule missing/unknown raw/tensor roles')
 need(type(r['raw_bytes'])is int and 0<=r['raw_bytes']<=RAW_LIMIT and r['raw_bytes']==sum(x['bytes'] for x in rows.values()) and type(r['gzip_bytes'])is int and 0<=r['gzip_bytes']<=GZIP_LIMIT and r['gzip_bytes']==sum(x['gzip_bytes'] for x in rows.values()),'Capsule whole bounds differ')
 need(all(r['driver_before'][k]==r['driver_after'][k] for k in ('pid','start_ticks','uid','cgroup','stdout_physical')) and r['driver_before']['pid']==request['driver_pid'] and r['driver_before']['start_ticks']==request['driver_start_ticks'] and r['driver_before']['uid']==0,'Actual original driver changed')
 request=dict(request,driver_stdout_physical=r['driver_before']['stdout_physical']);bodies={};inventory={}
 for name,row in rows.items():
  need(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe original output name');validate_row(name,row,request)
  target=out/'raw'/name;target.parent.mkdir(parents=True,exist_ok=True);decoded=save_streamed_member(row,target);inventory[name]={k:v for k,v in row.items() if k!='gzip_base64'}
  if name.endswith('.jsonl'):
   need(decoded['line_count']==(2560 if Path(name).name=='student.jsonl' else 64) and decoded['ends_lf']is True and row['line_count']==decoded['line_count'] and row['ends_lf']is True,'Whole raw trajectory line cardinality differs')
  else:
   need(row['bytes']<=SMALL_LIMIT,'Non-trajectory original exceeds small-file bound');bodies[name]=target.read_bytes()
 for name,row in metas.items():validate_row(name,row,request,metadata=True)
 m=unique(bodies['full_state/global_step_5/manifest.json']);ns={};exec(compile(reader_constants()+REMOTE_LIBRARY,'local_reader_validation','exec'),ns);ns['check_manifest'](m,request)
 for role,name in CHECKPOINT_MAP.items():
  if name in TENSOR_NAMES:need(m['files'][role]['sha256']==metas[name]['declared_sha256'],'Declared tensor SHA differs')
  else:need(m['files'][role]['sha256']==inventory[name]['sha256'],'Checkpoint raw role SHA differs')
 for name,p in request['source_files'].items():need(inventory['source/'+name]['sha256']==p,'Source raw bytes pin differs')
 need(inventory['full_state/global_step_5/resolved-config.json']['sha256']==m['config_sha256'],'Actual resolved config digest differs')
 ready=unique(bodies['approval-ready.json']);bound=unique(bodies['bindings/manifest.json']);src=unique(bodies['bindings/source.json'])
 need(ready['source_commit']==SOURCE and ready['job_id']==JOB and ready['run_root']==RUN and ready['binding_sha256']==inventory['bindings/manifest.json']['sha256'] and ready['public_key_sha256']==inventory['bindings/public.json']['sha256'],'Ready actual saved binding differs')
 need(bound['source_commit']==SOURCE and bound['run_root']==RUN and set(bound['files'])=={'allocation','data_manifest','environment','import_closure','model_manifest','source','tool_manifest'},'Saved run binding roles differ')
 for role,rec in bound['files'].items():need(rec['path']==RUN+'/bindings/'+role+'.json' and rec['sha256']==inventory['bindings/'+role+'.json']['sha256'],'Saved run binding member differs')
 need(src['clean']is True and src['source_commit']==SOURCE and src['repo']==REPO and src['source_files']==request['source_files'] and src['formal_manifest_sha256']==request['formal_sha'],'Saved source original binding differs')
 gate=type(sys)('first5_local_original_gate');gate.__file__=REPO+'/search_r1/baselines/checkpoint_gate.py';exec(compile(bodies['source/search_r1/baselines/checkpoint_gate.py'],gate.__file__,'exec',dont_inherit=True),gate.__dict__)
 workers=[unique(bodies['checkpoints/actor/global_step_5/training_state/rank_'+str(rank)+'.json'])['pid'] for rank in range(8)]
 need(all(type(pid)is int and pid>0 for pid in workers) and len(set(workers))==8,'Saved rank process identities differ')
 need(inventory[NATIVE_WARNING_NAME]['sha256']==NATIVE_WARNING_SHA,'Captured installed Torch warning source differs')
 need(ns['check_logs'](bodies['train.log'],gate,m,request,workers,bodies[NATIVE_WARNING_NAME])==r['train_prefix_join'] and ns['check_logs'](bodies['driver-current-prefix.out'],gate,m,request,workers,bodies[NATIVE_WARNING_NAME])==r['driver_prefix_join'],'Preserved complete first5 log replay differs')
 need(type(r['full_archive_members'])is int and r['full_archive_members']==100,'Whole five-outer archive member count differs')
 for step in range(1,6):
  base='trajectories/outer_'+str(step).zfill(6)+'/'
  q=unique(bodies[base+'questions.json']);paired=unique(bodies[base+'paired-manifest.json']);student=unique(bodies[base+'student-manifest.json'])
  need(type(q['questions'])is list and len(q['questions'])==512 and type(paired['student_count'])is int and paired['student_count']==2560 and type(paired['teacher_count'])is int and paired['teacher_count']==512 and type(student['student_count'])is int and student['student_count']==2560 and student['recorded_before_balancing']is True,'Saved archive complete declared count differs')
 missing=r['missing_optional'];need(type(missing)is list and len(set(missing))==len(missing) and set(missing)==set(OPTIONAL)-set(rows),'Optional pending list differs')
 r['files']=inventory;r['local_request']=request;r['cpu_pipeline_originals']=cpu_status(bodies,m,request);r['collector_source_sha256']=sha(Path(__file__).read_bytes());r['full_archive_byte_preservation']='ALL_100_MEMBERS_SHA_VERIFIED_NOT_SEMANTIC_ACCEPTANCE';write_once(out/'inventory.json',(json.dumps(r,indent=2,allow_nan=False)+'\n').encode());return r

def main():
 from transport import remote
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-manifest-sha256',required=True);p.add_argument('--pause-nonce',required=True);p.add_argument('--driver-pid',type=int,required=True);p.add_argument('--driver-start-ticks',required=True);a=p.parse_args();request=local_request(a);out=a.output
 need(out.is_absolute() and out.parent.resolve(strict=True)==out.parent and out.parent.stat().st_uid==os.getuid() and not os.path.lexists(out),'Fresh physical local output required');out.mkdir(mode=0o700)
 program=remote_program(request);compile(program,'first5_originals_read_only_remote','exec');command='python3 -c '+shlex.quote(program);write_once(out/'reader-source.py',program.encode());write_once(out/'command.original.json',(json.dumps({'command':command,'request':request,'reader_source_sha256':sha(program.encode()),'scope':SCOPE},indent=2)+'\n').encode())
 try:r=remote(command,timeout=900)
 except (subprocess.CalledProcessError,subprocess.TimeoutExpired)as e:
  write_once(out/'transport.stdout.original',e.stdout or b'');write_once(out/'transport.stderr.original',e.stderr or b'');write_once(out/'transport-exit.original.json',(json.dumps({'returncode':getattr(e,'returncode',None),'timed_out':isinstance(e,subprocess.TimeoutExpired),'error_type':type(e).__name__,'scope':SCOPE,'retry':'NO_AUTOMATIC_RETRY_OR_OVERWRITE'},indent=2)+'\n').encode());raise
 write_once(out/'transport.stdout.original',r.stdout);write_once(out/'transport.stderr.original',r.stderr);write_once(out/'transport-exit.original.json',(json.dumps({'returncode':r.returncode,'scope':SCOPE,'stdout_sha256':sha(r.stdout),'stderr_sha256':sha(r.stderr)},indent=2)+'\n').encode());need(type(r.returncode)is int and r.returncode==0,'Actual transport nonzero');save_snapshot(r.stdout,request,out)
if __name__=='__main__':main()
