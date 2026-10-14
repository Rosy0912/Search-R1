
import sys
sys.dont_write_bytecode=True
import hashlib,json,os,socket,stat,subprocess,time
from pathlib import Path
payload=json.load(sys.stdin)
base=Path('/data/shaoshitong/Search-G1-a800-fresh200-27bab88929e048b4869f3889b99d8e18')
repo=base/'repo'
library=Path('/data/shaoshitong/Search-G1-a800-fresh200-6047f7f9674b49fdad718a256ab3db95/repo')
result=dict(schema_version=1,role='qprobe_bf4_failed_native_owned_cleanup_verification',
 base=str(base),run_id='0f49c2a664b448f59b5f21dd49f9b10b',execution_id='7d85bd0daa93462981aefc2a9d87fdb7',
 deployment_id='27bab88929e048b4869f3889b99d8e18',source_commit='bf4b88d04c166951691029b18d05d91ece7c7054',
 accepted=False,actual_cleanup_outcome='unknown',remote_queries_performed=1,nvml_queries_performed=0,
 no_stop_or_signal=True,no_models_tensors_tests_or_evaluations=True,no_remote_mutation=True,
 old_namespace_access_scope='Published library code only; no old run resources queried',
 recorded_pipeline_birth=dict(pid=2890600,start_ticks=604281842),
 recorded_trainer_birth=dict(pid=2902242,start_ticks=604328381,previous_exit_code=1))
phase='identity_source_preflight'
def sealed(path,limit):
 path=Path(path);assert path.resolve()==path
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd);assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_size<=limit
  raw=os.read(fd,before.st_size+1);after=os.fstat(fd)
  assert len(raw)==before.st_size and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  assert path.stat().st_ino==before.st_ino
 finally:os.close(fd)
 return raw,dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def document(path,limit=1024**2):
 raw,proof=sealed(path,limit);return json.loads(raw),proof
try:
 assert base.resolve()==base and base.stat().st_uid==os.getuid()
 intent=payload['intent'];assert intent['base']==str(base) and intent['deployment_id']==result['deployment_id']
 assert intent['source_commit']==result['source_commit']
 run,manifest_file=document(base/'logs/run-manifest.json')
 assert all(run[k]==result[k] for k in ('run_id','execution_id','source_commit'))
 assert run['source_sha256']==intent['source_sha256'] and len(run['source_sha256'])==469
 assert run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None and run['execution_host']==socket.gethostname()
 assert run['training_world_size']==4 and run['training_physical_indices']==[0,1,2,3] and run['excluded_physical_indices']==[4,7]
 assert run['pipeline_identity']['pid']==2890600 and run['pipeline_identity']['start_ticks']==604281842
 prepared,prepared_file=document(base/'preparation/deployment-intent.json')
 assert all(prepared[k]==intent[k] for k in ('base','deployment_id','source_commit','source_sha256'))
 qprobe,qprobe_file=document(base/'preparation/qprobe-intent.json');assert qprobe==intent
 ledger,ledger_file=document(base/'logs/process-ledger.json',256*1024**2)
 assert ledger['run_id']==run['run_id'] and ledger['jobs']
 assert ledger['jobs']['trainer']['leader']['pid']==2902242 and ledger['jobs']['trainer']['leader']['start_ticks']==604328381
 assert all(job['state']=='STOPPED' for job in ledger['jobs'].values())
 for source_repo,commit,mapping in [(repo,run['source_commit'],run['source_sha256']),(library,payload['library_source'],payload['library_source_sha256'])]:
  assert source_repo.resolve()==source_repo and source_repo.stat().st_uid==os.getuid()
  assert subprocess.check_output(['git','-C',str(source_repo),'rev-parse','HEAD'],text=True,timeout=10).strip()==commit
  assert not subprocess.check_output(['git','-C',str(source_repo),'status','--porcelain'],timeout=10).strip()
  for relative,expected in mapping.items():
   assert Path(relative).is_absolute() is False and '..' not in Path(relative).parts
   raw,_=sealed(source_repo/relative,16*1024**2);assert hashlib.sha256(raw).hexdigest()==expected
 phase='published_cpu_library_import'
 sys.path.insert(0,str(library))
 from reproduction.common8.actor_probe_owned_cleanup import verify_cleanup,birth_inventory
 assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
 inventory=birth_inventory(base,run)
 exact=set()
 for job in inventory['jobs'].values():
  for row in job['births']:exact.add(tuple(row[k] for k in ('pid','start_ticks','sid','uid')))
 exact.add(tuple(run['pipeline_identity'][k] for k in ('pid','start_ticks','sid','uid')))
 result['unique_recorded_births']=len(exact)
 result['owned_job_birth_counts']=[dict(name=name,recorded_births=len(job['births']),ledger_state=ledger['jobs'][name]['state'],
  leader={k:job['leader'][k] for k in ('pid','start_ticks','sid','uid')}) for name,job in sorted(inventory['jobs'].items())]
 phase='actual_native_owned_cleanup_verification'
 # The published routine invokes exactly one fresh bounded nvidia-smi compute-apps query.
 actual=verify_cleanup(base,run,preserved_births=inventory)
 result['nvml_queries_performed']=1
 assert actual['accepted'] is True and actual['owned_cuda_contexts_remaining']==0
 assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
 phase='imported_source_readback'
 imported={}
 for name,module in list(sys.modules.items()):
  path=getattr(module,'__file__',None)
  if not path:continue
  path=Path(path).resolve()
  if library not in path.parents:continue
  relative=str(path.relative_to(library));assert relative in payload['library_source_sha256']
  raw,_=sealed(path,16*1024**2);digest=hashlib.sha256(raw).hexdigest()
  assert digest==payload['library_source_sha256'][relative];imported[relative]=digest
 result.update(status='ACTUAL_NATIVE_OWNED_CLEANUP_PASS',accepted=True,actual_cleanup_outcome='pass',
  actual_native_verification_performed=True,verification={k:actual[k] for k in ('actual_pipeline_and_owned_sessions_exited',
  'owned_cuda_contexts_remaining','private_cache_removed','original_assets_present','reused_pids_belonging_to_other_births',
  'reused_session_ids','recorded_births')},files=actual['files']+[manifest_file,prepared_file,qprobe_file],
  auditor_library_source=dict(source_commit=payload['library_source'],repo=str(library),source_inventory_files=len(payload['library_source_sha256']),
  helper_sha256=payload['library_source_sha256']['reproduction/common8/actor_probe_owned_cleanup.py'],imported_module_sha256=imported),
  audited_source=dict(source_commit=run['source_commit'],repo=str(repo),source_inventory_files=len(run['source_sha256'])),
  checked_unix=actual['checked_unix'])
except Exception as failure:
 result.update(status='ACTUAL_NATIVE_OWNED_CLEANUP_FAILED',failure_type=type(failure).__name__,failure_phase=phase,
  actual_native_verification_performed=(phase in ('actual_native_owned_cleanup_verification','imported_source_readback')),checked_unix=time.time(),
  automatic_retry=False)
print('CLEANUP_NATIVE_JSON:'+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
