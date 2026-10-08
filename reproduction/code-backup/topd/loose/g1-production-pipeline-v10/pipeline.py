"""Actual synchronous original refit routing, metadata publication and owned barriers.
No Torch import in this supervisor. Original dump/train/refit source stays sealed.
"""
import hashlib,importlib.util,json,os,pathlib,socket,struct,sys,time,types
from checkpoint_inventory import inventory,tokenizer_compatible,sha
from ipc import identity,read,send

def atomic(path,value):
 p=pathlib.Path(path);temp=p.with_name(p.name+'.tmp')
 fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
 d=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(d)
 finally:os.close(d)

def verify_files(root,manifest):
 root=pathlib.Path(root).resolve();actual=set()
 for directory,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   if pathlib.Path(directory,name).is_symlink():raise ValueError('source alias rejected')
  for name in files:actual.add(str(pathlib.Path(directory,name).relative_to(root)))
 expected=set(manifest)
 # Source manifest itself is an explicitly bound outside field, not circular.
 if actual!=expected:raise ValueError('exact source file set drift')
 for name,record in manifest.items():
  p=root/name
  if p.stat().st_size!=record['size'] or sha(p)!=record['sha256']:raise ValueError('source bytes drift: '+name)

def verify_overlay(root,receipt_path,expected):
 if sha(receipt_path)!=expected:raise ValueError('native overlay receipt changed')
 receipt=json.loads(pathlib.Path(receipt_path).read_text());files=receipt['overlay_files']
 # Native installer records may use SHA strings or size/SHA dictionaries.
 actual=set()
 for directory,dirs,names in os.walk(root,followlinks=False):
  for name in dirs+names:
   if pathlib.Path(directory,name).is_symlink():raise ValueError('overlay symlink rejected')
  actual.update(str(pathlib.Path(directory,name).relative_to(root)) for name in names)
 if actual!=set(files):raise ValueError('native overlay exact file set changed')
 for name,record in files.items():
  expected_sha=record if isinstance(record,str) else record['sha256']
  if sha(pathlib.Path(root)/name)!=expected_sha:raise ValueError('overlay bytes changed')
 return receipt

def validate_peer(pid,uid,trainer_pid,job_cgroup):
 if uid!=os.getuid():raise ValueError('foreign IPC uid')
 actual=identity(pid)
 if actual['cgroup']!=job_cgroup:raise ValueError('foreign job peer')
 current=pid;seen=set()
 while current not in seen and current>1:
  if current==trainer_pid:return actual
  seen.add(current);current=int(pathlib.Path('/proc',str(current),'stat').read_text().rsplit(')',1)[1].split()[1])
 raise ValueError('peer not owned trainer descendant')

def verify_worker_receipts(groups,uuids,job,cgroup):
 if set(groups)!= {'actor_rollout','ref'}:raise ValueError('production requires actor and reference groups')
 for role,records in groups.items():
  if len(records)!=8 or {r['rank'] for r in records}!=set(range(8)) or {r['gpu_uuid'] for r in records}!=set(uuids):raise ValueError('actual worker rank/UUID coverage')
  for r in records:
   if r['status']!='RELEASED_ZERO' or r['allocated']!=0 or r['reserved']!=0 or r['process_gpu_uuids']!=[r['gpu_uuid']]:raise ValueError('actual worker release not zero')
   observed=identity(r['pid'])
   if observed['starttime_ticks']!=r['starttime_ticks'] or observed['cgroup']!=r['cgroup'] or str(r['job'])!=str(job) or r['host']!='h100-11':raise ValueError('worker actual identity drift')
   if cgroup!=observed['cgroup']:raise ValueError('foreign worker cgroup')
   for key in ('parameter_devices','buffer_devices'):
    if any(any(not x.startswith('cpu') for x in ds) for ds in r[key].values()):raise ValueError('worker tensor still CUDA')
 return groups

def load_refit(producer):
 sys.path.insert(0,str(producer))
 import refit_with_owner
 return refit_with_owner

def original_round(config,checkpoint,step,environment,round_root,refit=None):
 """Runs ORIGINAL coevolve.refit; intercepts only final replace to freeze evidence first."""
 producer=pathlib.Path(config['producer']);refit=refit or load_refit(producer)
 model_inventory=inventory(checkpoint);tokenizer=tokenizer_compatible(checkpoint,config['base_model'])
 args=types.SimpleNamespace(**config['refit_args']);ctx=dict(config['owner_context']);ctx['config']=dict(ctx['config']);ctx['evidence_root']=str(round_root)
 ctx['config']['dump_args']=dict(ctx['config'].get('dump_args',{}),model=str(checkpoint))
 manifest=pathlib.Path(args.manifest_root)/('live_'+config['experiment'])/'manifest.json';manifest.parent.mkdir(parents=True,exist_ok=True)
 old=manifest.read_bytes() if manifest.exists() else None
 real_load=refit.load_original_coevolve
 def guarded_original():
  original=real_load();original_os=original.os
  def replace(src,dst):
   if pathlib.Path(dst).resolve()!=manifest.resolve():return os.replace(src,dst)
   payload=json.loads(pathlib.Path(src).read_text())
   if payload['round']!=step or pathlib.Path(payload['encoder']).resolve()!=pathlib.Path(checkpoint).resolve():raise ValueError('original manifest model identity changed')
   # Original headfit must have actually exited and canonical dump barriers passed.
   from publication import complete_dump
   directory=pathlib.Path(round_root)/f'outer-{step}';_,result,npz=complete_dump(directory)
   exits=[]
   for p in pathlib.Path(round_root).glob('heads-*.json'):
    r=json.loads(p.read_text());exits.append({'path':str(p),'sha256':sha(p),'receipt':r})
   if len(exits)!=1 or exits[0]['receipt']['returncode']!=0 or exits[0]['receipt']['own_processes_remaining']!=[] or exits[0]['receipt']['own_gpu_records_remaining']!=[] or not exits[0]['receipt']['baseline_preserved']:raise ValueError('missing actual CPU head exit')
   heads={k:{'path':payload[k],'sha256':sha(payload[k])} for k in ('commit','boundary')}
   frozen={'scope':'ORIGINAL_G1_CALIBRATION_SOURCE_NOT_GPU_ACCEPTANCE','outer_step':step,'checkpoint':model_inventory,'tokenizer':tokenizer,'heads':heads,'dump_result_sha256':sha(directory/'result.json'),'head_exits':exits,'manifest_file_sha256':sha(src)}
   atomic(pathlib.Path(round_root)/'frozen-calibration.json',frozen)
   os.chmod(src,0o600)
   with open(src,'rb') as f:os.fsync(f.fileno())
   os.replace(src,dst);fd=os.open(pathlib.Path(dst).parent,os.O_RDONLY)
   try:os.fsync(fd)
   finally:os.close(fd)
  original.os=types.SimpleNamespace(**{k:getattr(original_os,k) for k in dir(original_os) if k!='replace'},replace=replace)
  return original
 refit.load_original_coevolve=guarded_original
 try:
  complete,calls=refit.refit_with_owner(config['experiment'],step,checkpoint,args,environment,ctx)
  if not complete:raise RuntimeError('original refit did not complete')
  frozen=json.loads((pathlib.Path(round_root)/'frozen-calibration.json').read_text())
  if sha(manifest)!=frozen['manifest_file_sha256']:raise ValueError('published manifest bytes differ')
  return frozen
 except BaseException:
  if old is None:
   if manifest.exists():manifest.unlink()
  else:
   temp=manifest.with_name(manifest.name+'.pipeline-rollback')
   with open(temp,'xb') as f:f.write(old);f.flush();os.fsync(f.fileno())
   os.replace(temp,manifest)
  raise
 finally:refit.load_original_coevolve=real_load

class Supervisor:
 def __init__(self,config,environment,http,round_call=original_round):
  self.config=config;self.environment=environment;self.http=http;self.round_call=round_call;self.completed=[]
 def refresh(self,request):
  step=request['outer_step']
  if step!=([25,50,75][len(self.completed)] if len(self.completed)<3 else None):raise ValueError('refresh out of original order')
  verify_worker_receipts(request['worker_releases'],self.config['uuids'],self.config['job'],self.config['cgroup'])
  paused=self.http('barrier_pause')
  if paused['status']!='PAUSED_ENCODER_RELEASED' or len(paused['release'])!=8 or any(r['allocated'] or r['reserved'] for r in paused['release'].values()):raise ValueError('actual probe encoder release failed')
  checkpoint=pathlib.Path(request['checkpoint']).resolve()
  if not checkpoint.is_relative_to(pathlib.Path(self.config['checkpoint_root']).resolve()):raise ValueError('foreign checkpoint')
  root=pathlib.Path(self.config['evidence_root'])/f'refresh-{step}';root.mkdir(exist_ok=False)
  frozen=self.round_call(self.config,checkpoint,step,self.environment,root)
  resumed=self.http('barrier_resume')
  if resumed['status']!='READY' or resumed['round']!=step or pathlib.Path(resumed['encoder']).resolve()!=checkpoint or resumed['manifest_sha256']!=frozen['manifest_file_sha256']:raise ValueError('owned probe reloaded wrong evaluator')
  self.completed.append(step)
  return {'status':'COMPLETE','outer_step':step,'frozen_receipt':str(root/'frozen-calibration.json'),'frozen_receipt_sha256':sha(root/'frozen-calibration.json'),'manifest_sha256':frozen['manifest_file_sha256']}

def reuse_initial(config,reference,output):
 """Re-use completed SAME_BASE round0 readouts; original server supports manifests."""
 frozenpath=pathlib.Path(reference['path']);reviewpath=pathlib.Path(reference['independent_review']['path'])
 if sha(frozenpath)!=reference['sha256'] or sha(reviewpath)!=reference['independent_review']['sha256']:raise ValueError('Initial reuse source/review drift')
 review=json.loads(reviewpath.read_text())
 if review.get('status')!='PASS' or review.get('reviewed_frozen_calibration_sha256')!=reference['sha256']:raise ValueError('Need actual round0 independent review')
 frozen=json.loads(frozenpath.read_text())
 from publication import complete_dump
 source_config,source_result,source_npz=complete_dump(frozenpath.parent/'outer-0')
 if sha(frozenpath.parent/'outer-0/result.json')!=frozen['dump_result_sha256']:raise ValueError('actual initial dump result drift')
 data=pathlib.Path(config['refit_args']['data'])
 if source_config['data_sha256']!=sha(data) or pathlib.Path(source_config['dump_args']['data']).resolve()!=data.resolve():raise ValueError('initial calibration data differs')
 expected={'model':config['base_model'],'tag':config['refit_args']['tag'],'cap':300,'offset':0,'G':5,'layers':'21,24,27','gpu_mem':.55,'max_turns':4}
 if any(source_config['dump_args'].get(k)!=v for k,v in expected.items()):raise ValueError('initial original calibration recipe differs')
 current=inventory(config['base_model'])
 if frozen['outer_step']!=0 or frozen['checkpoint']['sha256']!=current['sha256']:raise ValueError('Only exact same official base round0 reuse')
 for h in frozen['heads'].values():
  if sha(h['path'])!=h['sha256']:raise ValueError('Original fitted readout changed')
 for x in frozen['head_exits']:
  if sha(x['path'])!=x['sha256'] or x['receipt']['returncode']!=0 or x['receipt']['own_processes_remaining'] or x['receipt']['own_gpu_records_remaining'] or not x['receipt']['baseline_preserved']:raise ValueError('Initial actual headfit exit drift')
 manifest=pathlib.Path(config['refit_args']['manifest_root'])/('live_'+config['experiment'])/'manifest.json';manifest.parent.mkdir(parents=True,exist_ok=True)
 if manifest.exists():raise ValueError('Refuse replacing preexisting initial manifest')
 payload={'round':0,'encoder':config['base_model'],'commit':frozen['heads']['commit']['path'],'boundary':frozen['heads']['boundary']['path']}
 atomic(manifest,payload)
 frozen=dict(frozen,reused_from=reference,manifest_file_sha256=sha(manifest))
 atomic(pathlib.Path(output)/'frozen-calibration.json',frozen)
 return frozen
