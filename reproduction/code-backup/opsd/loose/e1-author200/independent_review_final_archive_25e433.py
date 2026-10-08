"""Read-only independent immutable bundle/original byte joins; no remote execution."""
from pathlib import Path
import datetime,hashlib,json,os,re,stat,subprocess,types
W=Path(__file__).resolve().parent
D=W/'final-archive-25e433c0a1d3'
PIN='25e433c0a1d32e9a13f228537e13bda337089186a7a4be0349923580a97a9354'
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
RUNTIME='d83ef9dcbc950978a96b280789574917b0c8a81148c52d73ea1109a22f828406'
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN=ROOT/'runs/formal-3261';REPO=ROOT/'repo-188f4e5e7b63'
SNAP=W/'actual-snapshots/3261-20261005T232501896059Z/original-files'
records={}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def meta(s):return {k:getattr(s,k)for k in ('st_dev','st_ino','st_uid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
def raw(p,pin=None):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts,'Physical absolute path required')
 d=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for name in p.parts[1:-1]:
   nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=d);os.close(d);d=nxt
  need(os.fstat(d).st_uid==os.getuid(),'Local original parent owner differs')
  before=os.stat(p.name,dir_fd=d,follow_symlinks=False)
  fd=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=d)
  try:
   opened=os.fstat(fd);need(stat.S_ISREG(opened.st_mode)and opened.st_uid==os.getuid()and opened.st_nlink==1,'Owned nonalias regular original required')
   need(opened.st_size<=20*1024*1024,'Small-original bound exceeded')
   chunks=[]
   while True:
    block=os.read(fd,1048576)
    if not block:break
    chunks.append(block)
   b=b''.join(chunks)
   need(meta(before)==meta(opened)==meta(os.fstat(fd))==meta(os.stat(p.name,dir_fd=d,follow_symlinks=False)),'Original replaced or changed')
   need(len(b)==opened.st_size and(pin is None or sha(b)==pin),'Whole SHA/size differs: '+str(p))
   records[str(p)]={'path':str(p),'bytes':len(b),'sha256':sha(b),'mode':oct(stat.S_IMODE(opened.st_mode)),'metadata':meta(opened)}
   return b
  finally:os.close(fd)
 finally:os.close(d)
def unique(b):
 def pairs(xs):
  v={}
  for k,x in xs:need(k not in v,'Duplicate JSON key');v[k]=x
  return v
 def bad(x):raise ValueError('Nonfinite JSON')
 v=json.loads(b,object_pairs_hook=pairs,parse_constant=bad);json.dumps(v,allow_nan=False);return v
def load(p,pin,name):
 b=raw(p,pin);mod=types.ModuleType(name);mod.__file__=str(p);exec(compile(b,str(p),'exec'),mod.__dict__);return mod
m=unique(raw(D/'manifest.json',PIN))
need(D.name=='final-archive-'+PIN[:12]and D.resolve(strict=True)==D and D.stat().st_uid==os.getuid()and stat.S_IMODE(D.stat().st_mode)==0o555,'Directory identity/name/mode differs')
need(m['version']==1 and type(m['version'])is int and m['kind']=='AUTHOR200_FINAL_ORIGINAL_ARCHIVE_PREPARATION_TOOLS' and m['scope']=='IMMUTABLE_SOFTWARE_AND_EXISTING_IDENTITY_NOT_COLLECTOR_EXECUTION_OR_FINAL_ACCEPTANCE' and m['source_commit']==SOURCE,'Manifest source/scope differs')
need(m['tools']=={'collect_actual3261_final_originals.py':RUNTIME} and len(m['review_records'])==21 and not(set(m['tools'])&set(m['review_records'])),'Runtime/review separation differs')
union={**m['tools'],**m['review_records']};need(len(union)==22 and {p.name for p in D.iterdir()}==set(union)|{'manifest.json'},'Exact23 immutable members required')
for name,pin in union.items():
 need(type(name)is str and Path(name).name==name and name not in ('.','..','manifest.json') and re.fullmatch('[0-9a-f]{64}',pin),'Unsafe name/pin')
 raw(D/name,pin);need(records[str(D/name)]['mode']=='0o444','Member immutable mode differs')
need(records[str(D/'manifest.json')]['mode']=='0o444','Manifest mode differs')
receipt=unique(raw(W/'final-archive-freeze-25e433c0a1d3.json'))
need(receipt['directory']==str(D)and receipt['remote_directory']==str(ROOT/'tools'/D.name)and receipt['manifest_sha256']==PIN and receipt['manifest_last']is True and receipt['runtime_count']==1 and receipt['review_member_count']==21 and receipt['existing_dependency_count']==31,'Actual local freeze receipt differs')
need(type(receipt['published_members'])is list and len(receipt['published_members'])==23 and set(receipt['published_members'])==set(union)|{'manifest.json'} and receipt['published_members'][-1]=='manifest.json' and receipt['published_members'][0]=='collect_actual3261_final_originals.py'and receipt['directory_mode']=='0o555'and receipt['scope']=='LOCAL_IMMUTABLE_PREPARATION_NOT_DEPLOYMENT_EXECUTION_OR_FINAL_PASS','Receipt order/scope differs')
mtimes={p.name:p.stat().st_mtime_ns for p in D.iterdir()};need(all(mtimes['manifest.json']>=t for t in mtimes.values()),'Manifest write timestamp not last')
peer=unique(raw(W/'FINAL_ARCHIVE_BUNDLE_V4_INDEPENDENT_SOFTWARE_REVIEW.json','b8812d097221a0539ba5b712e9945623a4026a2128f110a5119f8f4c0060caec'))
need(peer['status']=='PASS_PREPARED_FINAL_ARCHIVE_BUNDLE_FREEZER_AND_PUBLISHER_SOFTWARE_NOT_FROZEN_DEPLOYED_OR_COLLECTED'and peer['runtime_collector_sha256']==RUNTIME and peer['independent_CPU_replay']['tests']==22 and peer['independent_CPU_replay']['suite_result']=='OK','Prepared software peer differs')
for name,record in peer['files'].items():
 member='PREPARATION_ONLY_'+name if name.endswith('.py')else name
 need(m['review_records'][member]==record['sha256']and records[str(D/member)]['bytes']==record['bytes'],'Preparation exact reviewed bytes differ')
for stream in ('stdout','stderr'):
 r=peer['independent_CPU_replay'][stream];raw(Path(r['path']),r['sha256'])
io=load(D/'collect_actual3261_final_originals.py',RUNTIME,'independent_exact_d83')
binding=unique(raw(D/'ACTUAL3261_INFLIGHT_STDERR_SLURM_PREFIX_INDEPENDENT_REVIEW.json',io.BINDINGS_SHA))
expected=io.validate_bindings(binding)
base=unique(raw(W/'final-acceptance-90c2695a7f46/manifest.json','90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'))
need(base['source_commit']==SOURCE and len(base['dependencies'])==278,'Old90 reference differs')
source=unique(raw(SNAP/'bindings/source.json',base['dependencies'][str(RUN/'bindings/source.json')]))
need(source['source_commit']==SOURCE and source['repo']==str(REPO)and source['clean']is True and len(source['source_files'])==16,'Actual source16 identity differs')
need(subprocess.check_output(['git','-C',str(W.parent/'opsd-e1-300'),'rev-parse','HEAD']).decode().strip()==SOURCE and not subprocess.check_output(['git','-C',str(W.parent/'opsd-e1-300'),'status','--porcelain']),'Local source188 is not clean')
expected_deps={str(REPO/name):pin for name,pin in source['source_files'].items()}
for name,pin in source['source_files'].items():raw(W.parent/'opsd-e1-300'/name,pin)
selected=[RUN/name for name in('bindings/source.json','bindings/allocation.json','bindings/manifest.json','bindings/import_closure.json','bindings/public.json','full_state/global_step_5/resolved-config.json','full_state/global_step_5/manifest.json','acceptance-5-receipt.json')]+[ROOT/'tools/formal-bb5f4872142f'/name for name in('manifest.json','formal_supervisor.py','formal_job.sh','owned_runtime.py','resource_sample.py')]
for p in selected:
 expected_deps[str(p)]=base['dependencies'][str(p)]
 local=SNAP/p.relative_to(RUN)if RUN in p.parents else W/'formal-bb5f4872142f'/p.name
 raw(local,expected_deps[str(p)])
expected_deps[str(ROOT/'tools/final-acceptance-90c2695a7f46/manifest.json')]=sha(raw(W/'final-acceptance-90c2695a7f46/manifest.json'))
expected_deps[str(RUN/'approval-ready.json')]='24d876c05496678648fa1d19228fe87c19685bd35903c2f0d1f641ef73af4892'
ready=unique(raw(SNAP/'approval-ready.json',expected_deps[str(RUN/'approval-ready.json')]))
need(len(expected_deps)==31 and m['dependencies']==expected_deps,'Exact31 dependency paths/pins differ')
manifest=unique(raw(SNAP/'full_state/global_step_5/manifest.json',expected_deps[str(RUN/'full_state/global_step_5/manifest.json')]))
signed=unique(raw(SNAP/'acceptance-5-receipt.json',expected_deps[str(RUN/'acceptance-5-receipt.json')]))
public=unique(raw(SNAP/'bindings/public.json',expected_deps[str(RUN/'bindings/public.json')]))
gate=load(W.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py',source['source_files']['search_r1/baselines/checkpoint_gate.py'],'independent_exact_c738')
gate.verify_rsa_signature(signed['payload'],signed['signature_hex'],public)
need(manifest['source_commit']==SOURCE and manifest['job_id']=='3261'and manifest['driver_pid']==1533312 and manifest['horizon']==200 and manifest['global_step']==5 and manifest['method']=='pure_opsd'and manifest['config_sha256']==m['contract']['config_sha256'],'First5 manifest source/identity differs')
need(signed['payload']['checkpoint_manifest_sha256']==expected_deps[str(RUN/'full_state/global_step_5/manifest.json')]and signed['payload']['driver_pid']==1533312 and signed['payload']['source_commit']==SOURCE and signed['payload']['horizon']==200,'RSA envelope join differs')
need(ready['source_commit']==SOURCE and ready['job_id']=='3261'and ready['run_root']==str(RUN)and ready['binding_sha256']==expected_deps[str(RUN/'bindings/manifest.json')]and ready['public_key_sha256']==expected_deps[str(RUN/'bindings/public.json')],'Actual preGPU ready context differs')
contract={'training_job_id':'3261','run_root':str(RUN),'source_root':str(REPO),'config_sha256':'73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60','original_driver_pid':1533312,'original_driver_start_ticks':'1829085316','world_size':8,'horizon':200,'collector_runtime_count':1,'runtime_execution_authorized_by_freeze':False,'requires_live_training_driver':False,'deployment_requires_original_driver_alive':True,'deployment_requires_typed_running_allocation':True,'future_final_review_requires_historical_inode_compare':True,'historical_binding_summary_sha256':'a855ab5b63e607218d40db8de87647438c424868e83af654a231936fbbad2349','historical_inode_members':20,'historical_prefix_members':11,'fresh_existing_dependency_count':31}
need(m['contract']==contract and all(type(m['contract'][k])is type(v)for k,v in contract.items()),'Typed exact contract differs')
history=unique(raw(D/'ACTUAL_FD_BINDINGS_SUMMARY_ORIGINAL.json',contract['historical_binding_summary_sha256']))
need(history['errors']==[]and history['scope']=='READ_ONLY_INFLIGHT_RAY_STDERR_SLURM_BINDING_NOT_CLOSED_LIFETIME_OR_RELEASE','Historical scope/errors differ')
projection={};prefix_records={}
packet=W/'ray-stderr-slurm-bindings-inflight-20261005T233707660781Z'
for b in expected:
 p=history['processes'][str(b['pid'])]
 need(all(type(p[k])is type(b[k])and p[k]==b[k]for k in b)and p['uids']==[0]*4,'Historical process/start/fd differs')
 for stream,record in(('stdout',p['stdout']),('stderr',history['files'][p['stderr_key']])):
  need(record['physical_path']==b[stream+'_fd_target']and record['before']==record['after'],'Historical record path/stat differs')
  v=record['before'];role=f"ray-pid-{b['pid']}-{stream}"
  projection[role]={'physical_path':record['physical_path'],'device':v['device'],'inode':v['inode'],'uid':v['uid'],'permission_mode':v['mode'],'minimum_final_bytes':v['bytes'],'historical_prefix_sha256':record.get('sha256'),'prefix_scope':'ACTUAL_RECORDED_BYTES'if'sha256'in record else'NO_PREFIX_HASH_IN_EXISTING_READER_NOT_GUESSED'}
  if stream=='stderr':
   local=packet/p['stderr_key'];bb=raw(local,record['sha256']);need(len(bb)==record['bytes']==v['bytes']and record['source_offset']==0 and record['closed_lifetime']is False,'Actual stderr original byte/scope differs');prefix_records[role]=records[str(local)]
for role in('slurm-formal-3261.out','slurm-formal-3261.err'):
 record=history['files'][role];v=record['before'];need(record['before']==record['after']and record['source_offset']==0 and record['closed_lifetime']is False,'Historical Slurm scope/offset differs')
 projection[role]={'physical_path':record['physical_path'],'device':v['device'],'inode':v['inode'],'uid':v['uid'],'permission_mode':v['mode'],'minimum_final_bytes':v['bytes'],'historical_prefix_sha256':record['sha256'],'prefix_scope':'ACTUAL_RECORDED_BYTES'}
 bb=raw(packet/role,record['sha256']);need(len(bb)==record['bytes']==v['bytes'],'Actual Slurm prefix size differs');prefix_records[role]=records[str(packet/role)]
need(m['historical_file_bindings']==projection and len(projection)==20 and len(prefix_records)==11 and len({(v['device'],v['inode'])for v in projection.values()})==20,'Historical20 inode/11 original prefix closure differs')
need(sum(v['historical_prefix_sha256']is None for v in projection.values())==9,'Absent nine stdout prefix SHA must remain absent')
report={'version':1,'status':'PASS_INDEPENDENT_FINAL_ARCHIVE_IMMUTABLE_BUNDLE_REVIEW','reviewer_role':'INDEPENDENT_NONAUTHOR_OF_NEW_FINAL_ARCHIVE_BUNDLE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':PIN,'collector_runtime_sha256':RUNTIME,'source_commit':SOURCE,'directory':str(D),'freeze_receipt':records[str(W/'final-archive-freeze-25e433c0a1d3.json')],'prepared_software_review':records[str(W/'FINAL_ARCHIVE_BUNDLE_V4_INDEPENDENT_SOFTWARE_REVIEW.json')],'scope':'LOCAL_IMMUTABLE_BUNDLE_BYTES_MODES_EXISTING_ORIGINAL_DEPENDENCY_HISTORY_JOINS_ONLY_NOT_REMOTE_DEPLOYMENT_OR_COLLECTOR_FINAL_ACCEPTANCE','checks':{'all23_members_SHA_and0444_directory0555':True,'actual_receipt_order_manifest_last_and_timestamps':True,'one_reviewed_d83_runtime21_separate_review_records':True,'exact31_dependency_paths_from16_actualsource_and_old90_join':True,'local16_source_current188_clean_fullSHA':True,'accepted5_RSA_and_small_context_config_ready_formal5_wholeSHA':True,'historical20_inode_projection_and11_original_full_prefixSHA':True,'nine_missing_stdout_prefix_SHA_not_reconstructed':True,'same_opened_verified_bytes_parsed_or_compiled_no_pyc':True,'deployment_and_final_scope_distinguished':True},'member_records':{n:records[str(D/n)]for n in list(union)+['manifest.json']},'dependencies':expected_deps,'historical_file_bindings':projection,'historical_original_prefix_records':prefix_records,'all_read_original_records':records,'extra_CPU_suite_or_publisher_execution_performed':False,'remote_reads_or_deployment_performed':False,'collector_execution_or_final_acceptance_performed':False,'limitations':['These31 dependency byte joins use previously captured exact local originals and frozen local source; no fresh remote31 validation occurred.','Actual freeze receipt/source and member timestamps support manifest-last local publication; no remote publication occurred.','Signed accepted5 RSA and small identity records are reused only as anchors, not a repetition of full first5 tensor inspection.','Historical20 identities and11 raw prefixes remain in-flight original scope; final current bytes/closure still require actual collector output and independent comparison.','Known frozen supervisor FSDP rescan failure remains unchanged; no future child/wrapper/Slurm exit, resources, HF200, checkpoint200, final51713 or effect acceptance is asserted.']}
out=W/'FINAL_ARCHIVE_25E433_IMMUTABLE_BUNDLE_INDEPENDENT_REVIEW.json'
with out.open('x')as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'report':str(out),'sha256':sha(out.read_bytes()),'scope':report['scope'],'member_count':23,'existing_dependency_count':31,'historical_inode_count':20,'original_prefix_count':11}))
