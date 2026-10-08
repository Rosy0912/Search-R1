
import base64,hashlib,json,os,re,socket,subprocess,sys,types
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN=ROOT/'runs/formal-3261';REPO=ROOT/'repo-188f4e5e7b63'
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
RUNTIME='d83ef9dcbc950978a96b280789574917b0c8a81148c52d73ea1109a22f828406'
BASE='90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'
READY='24d876c05496678648fa1d19228fe87c19685bd35903c2f0d1f641ef73af4892'
def need(v,m):
 if not v:raise ValueError(m)
def unique(raw):
 def pairs(items):
  out={}
  for k,v in items:need(k not in out,'Duplicate JSON key');out[k]=v
  return out
 v=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:need(False,'Nonfinite JSON'))
 json.dumps(v,allow_nan=False);return v
def validate_packet(packet):
 need(set(packet)=={'manifest_sha256','manifest_b64','members'},'Packet schema differs')
 pin=packet['manifest_sha256'];need(type(pin)is str and re.fullmatch('[0-9a-f]{64}',pin),'Invalid manifest pin')
 raw=base64.b64decode(packet['manifest_b64'],validate=True)
 need(hashlib.sha256(raw).hexdigest()==pin,'Manifest bytes changed')
 m=unique(raw);need(type(m['version'])is int and m['version']==1,'Typed manifest version required')
 need(m['kind']=='AUTHOR200_FINAL_ORIGINAL_ARCHIVE_PREPARATION_TOOLS' and m['source_commit']==SOURCE,'Wrong archive/source kind')
 need(m['tools']=={'collect_actual3261_final_originals.py':RUNTIME},'Only exact d83 is runtime')
 need(not(set(m['tools'])&set(m['review_records'])) and len(m['review_records'])==21,'Review/member closure differs')
 contract=m['contract'];need(contract['training_job_id']=='3261' and contract['run_root']==str(RUN) and contract['source_root']==str(REPO),'Foreign job/run/source')
 need(type(contract['original_driver_pid'])is int and contract['original_driver_pid']==1533312 and contract['original_driver_start_ticks']=='1829085316','Original driver identity differs')
 need(contract['deployment_requires_original_driver_alive']is True and contract['deployment_requires_typed_running_allocation']is True and contract['requires_live_training_driver']is False,'Deployment/collector life scopes differ')
 need(contract['runtime_execution_authorized_by_freeze']is False,'Bundle cannot execute collector')
 need(contract['config_sha256']=='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60' and type(contract['horizon'])is int and contract['horizon']==200 and type(contract['world_size'])is int and contract['world_size']==8,'Typed original config/budget differs')
 need(type(contract['fresh_existing_dependency_count'])is int and contract['fresh_existing_dependency_count']==len(m['dependencies'])==31,'Exact31 collector dependencies required')
 pins={**m['tools'],**m['review_records']};need(set(packet['members'])==set(pins),'Member set differs')
 members={n:base64.b64decode(b,validate=True) for n,b in packet['members'].items()}
 for n,data in members.items():need(Path(n).name==n and n not in ('manifest.json','.','..') and hashlib.sha256(data).hexdigest()==pins[n],'Member name/SHA differs')
 return m,raw,members
def load_io(raw):
 need(hashlib.sha256(raw).hexdigest()==RUNTIME,'Collector byte pin differs')
 c=types.ModuleType('verified_packet_archive_collector_io');c.__file__='<exact-reviewed-d83-packet>'
 exec(compile(raw,c.__file__,'exec'),c.__dict__);return c
def read(c,path,pin):
 fd=c.open_physical(path,uid=0)
 try:
  before=c.metadata(os.fstat(fd))
  with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read()
  need(c.metadata(os.fstat(fd))==before==c.physical_metadata(path,0) and len(raw)==before['bytes'] and hashlib.sha256(raw).hexdigest()==pin,'Dependency changed: '+str(path))
  return raw
 finally:os.close(fd)
def expected_dependencies(c):
 basepath=ROOT/'tools/final-acceptance-90c2695a7f46/manifest.json'
 base=unique(read(c,basepath,BASE));need(base['source_commit']==SOURCE and len(base['dependencies'])==278,'Existing90 reference differs')
 sourcepath=RUN/'bindings/source.json'
 source=unique(read(c,sourcepath,base['dependencies'][str(sourcepath)]))
 need(source['source_commit']==SOURCE and source['repo']==str(REPO) and len(source['source_files'])==16,'Existing actual source16 identity differs')
 wanted={str(REPO/n):h for n,h in source['source_files'].items()}
 selected=[RUN/n for n in ('bindings/source.json','bindings/allocation.json','bindings/manifest.json','bindings/import_closure.json','bindings/public.json','full_state/global_step_5/resolved-config.json','full_state/global_step_5/manifest.json','acceptance-5-receipt.json')]
 formal=ROOT/'tools/formal-bb5f4872142f'
 selected += [formal/n for n in ('manifest.json','formal_supervisor.py','formal_job.sh','owned_runtime.py','resource_sample.py')]
 for path in selected:wanted[str(path)]=base['dependencies'][str(path)]
 wanted[str(basepath)]=BASE;wanted[str(RUN/'approval-ready.json')]=READY
 need(len(wanted)==31,'Exact31 existing collector paths differ')
 return wanted
def source_and_allocation(c,verified):
 need(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','Wrong actual UID/node')
 head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD']).decode().strip()
 clean=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'])
 need(head==SOURCE and not clean,'Source188 changed or dirty')
 raw=subprocess.check_output(['scontrol','show','job','3261','-o'])
 fields=c.scheduler_fields(raw)['fields']
 original=unique(verified[str(RUN/'bindings/allocation.json')])
 old=c.scheduler_fields(original['scontrol'].encode())['fields']
 need(fields.get('JobId')=='3261' and fields.get('JobState')=='RUNNING' and fields.get('NodeList')=='h100-10' and fields.get('UserId')=='root(0)','Original typed RUNNING allocation missing')
 need(fields.get('NumNodes')=='1' and fields.get('TresPerNode')=='gres:gpu:h100:8' and fields.get('OverSubscribe')=='NO' and fields.get('Restarts')=='0','Original8GPU/exclusive/noRestart allocation differs')
 need(fields.get('NumCPUs')=='192' and fields.get('MinMemoryNode')=='1T' and fields.get('StartTime')==old['StartTime'] and fields.get('SubmitTime')==old['SubmitTime'],'Original allocation start/resources differ')
 identity=c.process_observation(1533312,'1829085316')
 need(identity['status']=='ORIGINAL_OR_CURRENT_PROCESS_ALIVE' and identity['uids']==[0,0,0,0] and all(type(v)is int for v in identity['uids']) and identity['cgroup']==c.CGROUP,'Original driver must actually remain alive at deployment')
 return {'scontrol_original':raw.decode(),'source_commit':head,'source_clean':True,'original_driver_current_identity':identity,'scope':'Fresh deployment identity only; no collector or terminal acceptance'}
def publish(packet):
 need(sys.flags.optimize==0,'Optimized Python forbidden')
 m,mraw,members=validate_packet(packet);c=load_io(members['collect_actual3261_final_originals.py'])
 need(m['dependencies']==expected_dependencies(c),'Publication cannot read unapproved dependency paths or pins')
 verified={p:read(c,p,h) for p,h in m['dependencies'].items()}
 need(len(verified)==31,'Dependency count differs')
 accepted=unique(verified[str(RUN/'full_state/global_step_5/manifest.json')])
 need(accepted['source_commit']==SOURCE and accepted['job_id']=='3261' and accepted['run_root']==str(RUN) and type(accepted['driver_pid'])is int and accepted['driver_pid']==1533312 and accepted['config_sha256']==m['contract']['config_sha256'],'Existing accepted5 identity differs')
 source=unique(verified[str(RUN/'bindings/source.json')])
 need(source['source_commit']==SOURCE and source['repo']==str(REPO) and len(source['source_files'])==16,'Existing source inventory differs')
 for n,h in source['source_files'].items():need(m['dependencies'].get(str(REPO/n))==h,'Exact source16 closure differs')
 before=source_and_allocation(c,verified)
 d=ROOT/'tools'/('final-archive-'+packet['manifest_sha256'][:12]);parent=d.parent
 need(parent.resolve(strict=True)==parent and parent.stat().st_uid==0,'Physical own tools parent required')
 os.mkdir(d,0o700);published=[]
 for name,data in list(members.items())+[('manifest.json',mraw)]:
  c.write_exclusive(d/name,data);(d/name).chmod(0o444)
  fd=c.open_physical(d/name,uid=0)
  try:os.fsync(fd)
  finally:os.close(fd)
  pin=hashlib.sha256(data).hexdigest();need(read(c,d/name,pin)==data,'Published member readback differs')
  published.append({'name':name,'bytes':len(data),'sha256':pin,'mode':oct((d/name).stat().st_mode&0o777)})
 need(published[-1]['name']=='manifest.json' and set(p.name for p in d.iterdir())==set(members)|{'manifest.json'},'Manifest ordering/member set differs')
 d.chmod(0o555)
 for path in (d,parent):
  fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:os.fsync(fd)
  finally:os.close(fd)
 verified_after={p:read(c,p,h) for p,h in m['dependencies'].items()}
 need(verified_after==verified,'Existing dependencies changed during publication')
 after=source_and_allocation(c,verified_after)
 return {'scope':'ACTUAL_NEW_ARCHIVE_SOFTWARE_PUBLICATION_NOT_COLLECTOR_OR_FINAL_ACCEPTANCE','directory':str(d),'manifest_sha256':packet['manifest_sha256'],'manifest_last':True,'published':published,'runtime_count':1,'existing_dependency_count':31,'source_commit':SOURCE,'allocation_before':before,'allocation_after':after,'verified_dependencies':{p:{'sha256':hashlib.sha256(r).hexdigest(),'bytes':len(r)} for p,r in verified.items()},'directory_mode':oct(d.stat().st_mode&0o777),'collector_executed':False}
if __name__=='__main__':print(json.dumps(publish(unique(sys.stdin.buffer.read())),sort_keys=True))
