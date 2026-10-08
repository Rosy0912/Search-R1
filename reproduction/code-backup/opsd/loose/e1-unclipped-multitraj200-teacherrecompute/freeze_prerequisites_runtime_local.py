"""Local immutable packaging only; does not invoke any verifier or job."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types

HERE=Path(__file__).resolve().parent
PRIOR=HERE/'final200-prerequisites-candidate'
CANDIDATE=HERE/'final200-prerequisites-v2-candidate'
SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
VERIFIER='7dff811b01b302f45c729ee12ae9ee22bb561b359f24fcd0d37352192db2d075'
COMMON='e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581'
PEER='2a0955191663a55b607ed9e774fe69b36727ea28d33bd39beb2f87dcf43361c6'
PUBLIC='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
GATE='09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'
RECEIPT5='d1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
FIELDS=('st_dev','st_ino','st_uid','st_nlink','st_mode','st_size','st_mtime_ns','st_ctime_ns')

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def identity(value):return tuple(getattr(value,key) for key in FIELDS)
def stable(path,expected=None):
 p=Path(path).absolute();require(p.resolve(strict=True)==p,'Physical owned input required')
 before=p.lstat();require(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_nlink==1 and before.st_size<=32*1024**2,'Owned bounded single-link regular input required')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  require(identity(os.fstat(fd))==identity(before),'Opened input identity differs')
  raw=b''
  while True:
   chunk=os.read(fd,1024**2)
   if not chunk:break
   raw+=chunk;require(len(raw)<=32*1024**2,'Input read bound exceeded')
  require(identity(os.fstat(fd))==identity(p.stat())==identity(before) and len(raw)==before.st_size,'Input changed during read')
 finally:os.close(fd)
 if expected is not None:require(sha(raw)==expected,'Input SHA differs: '+str(p))
 return raw,dict(physical_path=str(p),sha256=sha(raw),bytes=len(raw),device=before.st_dev,inode=before.st_ino,uid=before.st_uid,nlink=before.st_nlink,mode=stat.S_IMODE(before.st_mode),mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns)
def sync(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
def write_once(path,raw):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  with os.fdopen(fd,'wb',closefd=False) as f:f.write(raw);f.flush();os.fsync(fd)
  os.fchmod(fd,0o444);os.fsync(fd)
 finally:os.close(fd)
 sync(path.parent)

def prepare():
 require(sys.platform=='darwin','This packager is restricted to the current local host')
 inputs={};members={};roles={}
 def add(name,path,expected=None,role='NON_RUNTIME_REVIEW_OR_ORIGINAL'):
  require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',name) and name not in members,'Unsafe/duplicate bundle member')
  raw,record=stable(path,expected);inputs[str(Path(path).absolute())]=record;members[name]=raw;roles[name]=role;return raw
 add('verify_actual200_prerequisites.py',CANDIDATE/'verify_actual200_prerequisites.py',VERIFIER,'RUNTIME')
 common_raw=add('common.py',CANDIDATE/'common.py',COMMON,'RUNTIME')
 readiness_raw=add('ORIGINAL_V1_SOFTWARE_READINESS.json',PRIOR/'IMPLEMENTATION_SOFTWARE_READINESS.json','544a4eb65f7742cd2871042d98f78cc6a57d1ed5d066df33f8760a101ce2b2f9')
 readiness=json.loads(readiness_raw)
 for name in ['test_prerequisites.py','DESIGN.md','BLOCKED_LAUNCH.json']:
  add(name,PRIOR/name,readiness['members'][name]['sha256'])
 add('ORIGINAL_V1_REVIEWED_VERIFIER_SOURCE.original.txt',PRIOR/'verify_actual200_prerequisites.py','98d4f9d2bc5c14a3b515630978db3fe76c14124e3e8a5c4375f8c65b09d36bd3','ORIGINAL_V1_SOFTWARE_SOURCE_NOT_RUNTIME')
 peer_raw=add('INDEPENDENT_PREREQUISITES_SOFTWARE_REVIEW.json',HERE/'FINAL200_PREREQUISITES_INDEPENDENT_SOFTWARE_REVIEW.json',PEER)
 peer=json.loads(peer_raw)
 require(peer['status']=='PASS_INDEPENDENT_FINAL200_PREREQUISITES_SOFTWARE_ONLY' and peer['actual200'] is None and peer['production_main_executed'] is False,'Software peer scope differs')
 add('INDEPENDENT_PREREQUISITES_SOFTWARE_REVIEW.md',HERE/'FINAL200_PREREQUISITES_INDEPENDENT_SOFTWARE_REVIEW.md','9518e406d19bce0043907791f9468634b3b75f0c4f3ac057a1352619f3d82a9c')
 for stamp in ['20261006T152519953374Z','20261006T152734296439Z','20261006T153101219499Z']:
  status_raw=add('AUTHOR_CPU_'+stamp+'_STATUS.original.json',PRIOR/'cpu-originals'/(stamp+'-status.json'))
  status=json.loads(status_raw)
  add('AUTHOR_CPU_'+stamp+'_STDOUT.original',Path(status['stdout']))
  add('AUTHOR_CPU_'+stamp+'_STDERR.original',Path(status['stderr']))
  if stamp=='20261006T153101219499Z':
   require(type(status['returncode']) is int and status['returncode']==0 and status['before']==status['after'],'Actual author CPU witness differs')
   require(sha(status_raw)==readiness['cpu_original_status']['sha256'],'Actual86 original SHA differs')
   for path,expected in status['before'].items():
    raw,record=stable(path,expected);inputs[record['physical_path']]=record
 for name,record in peer['independent_tests'].items():
  if name in ('execution','stdout','stderr'):add('INDEPENDENT_CPU_'+name.upper()+'.original',record['path'],record['sha256'])
 peer_execution=json.loads(members['INDEPENDENT_CPU_EXECUTION.original'])
 require(peer_execution['returncode']==0 and type(peer_execution['returncode']) is int and peer_execution['waited'] is True and peer_execution['before']==peer_execution['after'],'Independent86 actual witness differs')
 composition=peer['separate_real_local_wait_composition']
 add('HISTORICAL_LOCAL_WAIT_COMPOSITION.original.json',composition['path'],composition['sha256'])
 delta_peer_raw=add('INDEPENDENT_CROSS_PACKAGE_COMMON_DELTA_REVIEW_V2.json',HERE/'FINAL200_PREREQUISITES_CROSS_PACKAGE_COMMON_INDEPENDENT_REVIEW_V2.json','d9f84822ad1856e0077498a8b442b75fede7574a4ca7d740e841d80aa7d96142')
 delta_peer=json.loads(delta_peer_raw)
 require(delta_peer['status']=='PASS_INDEPENDENT_LOCAL_CROSS_PACKAGE_SOFTWARE_DELTA_ONLY' and delta_peer['reviewed_runtime']['sha256']==VERIFIER and delta_peer['reviewed_common']['sha256']==COMMON and delta_peer['production_main_executed'] is False and delta_peer['actual200'] is None,'Delta peer source or scope differs')
 add('INDEPENDENT_CROSS_PACKAGE_COMMON_DELTA_REVIEW_V2.md',HERE/'FINAL200_PREREQUISITES_CROSS_PACKAGE_COMMON_INDEPENDENT_REVIEW_V2.md')
 for name,expected in [('CROSS_PACKAGE_COMMON_DESIGN_ADDENDUM.md',delta_peer['reviewed_addendum']['sha256']),('SOFTWARE_READINESS_V2.json','472695b89288f1d865a9b4aec14e22a842a30282964adeb002306cb08f662a81'),('test_cross_package_common.py','9a637a94106ce49fecbba5a1275c641e69d95ec36304a3ca4bcf8dae506afab7')]:add(name,CANDIDATE/name,expected)
 delta_execution=add('AUTHOR_CROSS_PACKAGE_DELTA_EXECUTION.original.json',CANDIDATE/'cpu-cross-package-v2-originals/execution.original.json','b69b5953d2951f74dafcff4add9d08a48661193df21e4487a365bd2570364292')
 delta_value=json.loads(delta_execution);require(delta_value['returncode']==0 and type(delta_value['returncode']) is int and delta_value['waited'] is True and delta_value['before']==delta_value['after'],'Actual6 CPU delta witness differs')
 for name in ['stdout.original','stderr.original']:add('AUTHOR_CROSS_PACKAGE_DELTA_'+name,CANDIDATE/'cpu-cross-package-v2-originals'/name)
 for key in ('execution_original','stdout_original','stderr_original'):
  rec=delta_peer['independent_delta_cpu'][key];add('INDEPENDENT_CROSS_PACKAGE_'+key+'.original',rec['path'],rec['sha256'])
 add('CROSS_PACKAGE_COMMON_COUNTEREXAMPLE.original.json',HERE/'prerequisites-cross-package-common-counterexample-originals-v1/counterexample.original.json','dec5a2f420fd21b10f50eebb9e1b651d73eddadab7c1b8b7083e7f4677a1aba9')
 add('EXISTING_ACCEPTED5_PUBLIC.original.json',HERE/'job3304-first5-complete-originals-v3/raw/bindings/public.json',PUBLIC,'HISTORICAL_ACCEPTED5_ANCHOR_NOT_ACTUAL200')
 add('EXISTING_ACCEPTED5_RECEIPT.original.json',HERE/'job3304-release-live-originals-v8/acceptance-5-receipt.json',RECEIPT5,'HISTORICAL_ACCEPTED5_ANCHOR_NOT_ACTUAL200')
 add('EXISTING_FCEC_RSA_VERIFIER_SOURCE.original.txt',HERE.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py',GATE,'FIXED_SOURCE_TEXT_ANCHOR_NOT_RUNTIME_MODULE')
 add('LOCAL_PACKAGER.original.txt',Path(__file__).resolve(),role='LOCAL_PACKAGING_SOURCE_NOT_RUNTIME')
 # Read exact common declarations without any production verifier/context call.
 common=types.ModuleType('packaging_exact_common_declarations');common.__file__=str(CANDIDATE/'common.py')
 exec(compile(common_raw,common.__file__,'exec',dont_inherit=True,optimize=0),common.__dict__)
 checkpoint=common.checkpoint_paths();archives=common.archive_paths()
 require(len(checkpoint)==31 and len(archives)==4000,'Exact common expected role sets differ')
 blocked_record=lambda origin=None:dict(status='BLOCKED_REQUIRE_FUTURE_ACTUAL_OWNED_ORIGINAL',expected_origin_path=str(origin) if origin is not None else None,physical_path=None,sha256=None,bytes=None,execution_identity=None)
 bindings=dict(version=1,kind='PREREQUISITES_DEPENDENCY_BINDINGS_NOT_ACTUAL200_OR_DEPLOYMENT',
  status='SOFTWARE_ONLY_ALL_FUTURE_QUALIFICATION_BLOCKED',training_source_commit=SOURCE,stored_training_job_id='3304',config_sha256=common.CONFIG,
  runtime_tools={'verify_actual200_prerequisites.py':VERIFIER,'common.py':COMMON},
  runtime_derivative=dict(prior_verifier_sha256='98d4f9d2bc5c14a3b515630978db3fe76c14124e3e8a5c4375f8c65b09d36bd3',original86_scope='ORIGINAL_V1_ONLY_NOT_V2_PRODUCTION_MAIN',delta_review_sha256='d9f84822ad1856e0077498a8b442b75fede7574a4ca7d740e841d80aa7d96142',scope='ORIGINAL_LOG_COMMON_PHYSICAL_IDENTITY_REPLAY_STRICT_COMPARISON_UNCHANGED'),
  existing_anchors=dict(public_key=dict(origin_path=str(common.RUN/'bindings/public.json'),sha256=PUBLIC,member='EXISTING_ACCEPTED5_PUBLIC.original.json',scope='EXISTING_PUBLIC_ANCHOR_ONLY'),
   fcec_rsa_verifier=dict(origin_path=str(common.REPO/'search_r1/baselines/checkpoint_gate.py'),sha256=GATE,member='EXISTING_FCEC_RSA_VERIFIER_SOURCE.original.txt',source_commit=SOURCE),
   accepted5_receipt=dict(sha256=RECEIPT5,member='EXISTING_ACCEPTED5_RECEIPT.original.json',scope='HISTORICAL5_CANNOT_QUALIFY200')),
  unchanged_final200_helpers=dict(bundle_manifest_sha256='f11580b6a7bc52c04f51e7e6332e58e01611dae0a4f3ddda0e916a75757787ce',tools=peer['reviewed_files'],actual_future_receipt_tool_records=None),
  signed_layered_actual200_receipt=blocked_record(),independent_actual200_qualification=blocked_record(),
  actual200_reports={name:blocked_record() for name in ('inventory','native','hf','archives','logs_terminal')},
  expected_checkpoint31={name:blocked_record(path) for name,path in sorted(checkpoint.items())},
  expected_archives4000={name:blocked_record(path) for name,path in sorted(archives.items())},
  raw_exit_layers={name:blocked_record() for name in ('training_child_exit','wrapper_exit','slurm_terminal_original')},
  writer_lifetime_raw={name:blocked_record() for name in ('writer_processes','writer_fd_closure','driver_complete','aggregate_complete','resources_complete')},
  postterminal_source_bindings=blocked_record(),resource_release_raw={name:blocked_record() for name in ('release_proc','release_nvml','release_ports','release_cgroups','release_leases_locks')},
  known_whole_resource_status='FAILED_WHOLE_RESOURCE_OWNERSHIP_NOT_UPGRADED_BY_TRAINING_PHASE',
  actual_future_evaluation_allocation=None,actual_future_cpu_allocations=None,actual_hf200=None,actual200=None,effect_completion=None,
  no_private_key_or_signing=True,postterminal_fallback_included=False,
  limitations=['Expected physical origin names are fixed contracts, not claims of existing200 files','Historical software86 and accepted5 originals are not actual200 evidence','Real literal child/wrapper/Slurm states remain separate; no overall completion or resource release inferred','Verifier loads only same-byte common sibling; existing final200 helpers remain receipt-bound separate originals'])
 # Correct closure is the verifier's exact six helper declarations, not peer fields.
 verifier=types.ModuleType('packaging_exact_verifier_declarations');verifier.__file__=str(CANDIDATE/'verify_actual200_prerequisites.py')
 exec(compile(members['verify_actual200_prerequisites.py'],verifier.__file__,'exec',dont_inherit=True,optimize=0),verifier.__dict__)
 bindings['unchanged_final200_helpers']['tools']=dict(verifier.PINS)
 members['DEPENDENCY_BINDINGS_AND_BLOCKED_SCOPE.json']=canonical(bindings);roles['DEPENDENCY_BINDINGS_AND_BLOCKED_SCOPE.json']='EXPLICIT_BLOCKED_FUTURE_DEPENDENCY_CONTRACT'
 members['LOCAL_INPUT_PROVENANCE.original.json']=canonical(dict(kind='LOCAL_IMMUTABLE_PACKAGE_INPUT_RECORDS_NOT_RUNTIME_OR_ACTUAL200',source_commit=SOURCE,inputs=inputs,roles=roles));roles['LOCAL_INPUT_PROVENANCE.original.json']='PACKAGING_ORIGINAL_PROVENANCE'
 manifest=dict(version=1,kind='STRICT_FINAL200_PREREQUISITES_READ_ONLY_RUNTIME_BUNDLE',source_commit=SOURCE,tools={name:sha(raw) for name,raw in sorted(members.items())})
 manifest_raw=canonical(manifest);manifest_sha=sha(manifest_raw)
 destination=HERE/('final200-prerequisites-runtime-bundle-'+manifest_sha[:12]);require(not os.path.lexists(destination),'Existing bundle must not be overwritten')
 destination.mkdir(mode=0o700);sync(HERE);sync(destination)
 operations=[]
 for name,raw in sorted(members.items()):
  write_once(destination/name,raw);operations.append(dict(sequence=len(operations)+1,action='MEMBER_WRITTEN_FSYNCED_READ_ONLY',name=name,sha256=sha(raw)))
 for path,record in inputs.items():
  _,current=stable(path,record['sha256']);require(current==record,'Input changed during packaging')
 write_once(destination/'manifest.json',manifest_raw);operations.append(dict(sequence=len(operations)+1,action='MANIFEST_WRITTEN_LAST_FSYNCED_READ_ONLY',name='manifest.json',sha256=manifest_sha))
 os.chmod(destination,0o555);sync(destination);sync(HERE)
 require(set(p.name for p in destination.iterdir())==set(members)|{'manifest.json'},'Final membership differs')
 inode_set=set();final_records={}
 for name,expected in manifest['tools'].items():
  _,record=stable(destination/name,expected);require(record['mode']==0o444,'Member mode differs');require((record['device'],record['inode']) not in inode_set,'Bundle members alias');inode_set.add((record['device'],record['inode']));final_records[name]=record
 _,mrecord=stable(destination/'manifest.json',manifest_sha);require(mrecord['mode']==0o444 and stat.S_IMODE(destination.stat().st_mode)==0o555,'Final manifest/directory mode differs')
 for path,record in inputs.items():
  _,current=stable(path,record['sha256']);require(current==record,'Input changed after final readback')
 receipt=dict(version=1,kind='LOCAL_IMMUTABLE_PREREQUISITES_PACKAGING_RECEIPT_NOT_DEPLOYMENT_OR_ACTUAL200',status='LOCAL_PACKAGING_BYTES_VERIFIED_PENDING_INDEPENDENT_BUNDLE_REVIEW',
  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE,stored_training_job_id='3304',bundle_directory=str(destination),manifest_sha256=manifest_sha,manifest_record=mrecord,
  runtime_members=['verify_actual200_prerequisites.py','common.py'],member_count=len(members),all_member_records=final_records,input_records_before=inputs,input_records_after=inputs,input_stability_verified=True,
  packaging_source_sha256=sha(Path(__file__).read_bytes()),packaging_argv=sys.argv,operations=operations,manifest_written_last=True,all_members_mode='0444',directory_mode='0555',
  actual_remote_deployment=None,actual200=None,actual_signed_layered_receipt=None,actual_evaluation_job_id=None,effect_completion=None,private_key_accessed=False,production_main_executed=False,tests86_repeated=False)
 receipt_path=HERE/('final200-prerequisites-freeze-'+manifest_sha[:12]+'.json');write_once(receipt_path,canonical(receipt))
 return dict(bundle_directory=str(destination),manifest_sha256=manifest_sha,member_count=len(members),receipt_path=str(receipt_path),receipt_sha256=sha(canonical(receipt)),actual200=None,actual_remote_deployment=None)

if __name__=='__main__':print(canonical(prepare()).decode())
