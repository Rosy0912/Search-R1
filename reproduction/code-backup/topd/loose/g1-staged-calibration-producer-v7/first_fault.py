"""Exclusive fsynced original failures before cleanup; never replace first fault."""
import hashlib,json,os,pathlib,tempfile,time,traceback

def exclusive(path,value):
 path=pathlib.Path(path)
 if path.is_symlink():raise ValueError('First-fault evidence alias')
 fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.private',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  try:os.link(temp,path)
  except FileExistsError:
   if not path.is_file() or path.is_symlink():raise ValueError('Existing first-fault path invalid')
  d=os.open(path.parent,os.O_RDONLY)
  try:os.fsync(d)
  finally:os.close(d)
 finally:
  if os.path.exists(temp):os.unlink(temp)
 return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def registration_raw(owned):
 tracker=getattr(owned,'stage_scope',None)
 if tracker is None:return {'scope':'STAGE_REGISTRATION_UNAVAILABLE','error':'owned tracker unavailable'}
 return {'scope':'ACTUAL_STAGE_REGISTRATION_RAW_NOT_ACCEPTANCE','roots':getattr(tracker,'roots',None),'ledger':getattr(tracker,'ledger',None),'events':getattr(tracker,'events',None),'sealed':getattr(tracker,'sealed',None),'open_pidfd_registered_pids':sorted(getattr(tracker,'fds',{})),'durable_events_path':str(getattr(getattr(tracker,'events',None),'path',None))}

def before_cleanup(directory,label,error,owned,config_sha256):
 directory=pathlib.Path(directory);result={'primary_error_ref':None,'registration_ref':None,'publication_errors':[]}
 if error is not None:
  value={'scope':'ORIGINAL_PRIMARY_FAILURE_BEFORE_OWNED_CLEANUP_NOT_CAUSAL_ATTRIBUTION','phase':label,'pid':os.getpid(),'ppid':os.getppid(),'job':os.environ.get('SLURM_JOB_ID'),'epoch':time.time(),'config_sha256':config_sha256,'error_type':type(error).__name__,'error':str(error),'traceback':''.join(traceback.format_exception(type(error),error,error.__traceback__))}
  try:result['primary_error_ref']=exclusive(directory/(label+'-primary-error.json'),value)
  except BaseException as exc:result['publication_errors'].append({'event':'primary-before-cleanup','error_type':type(exc).__name__,'error':str(exc)})
 try:result['registration_ref']=exclusive(directory/(label+'-registration-before-cleanup.json'),registration_raw(owned))
 except BaseException as exc:result['publication_errors'].append({'event':'registration-before-cleanup','error_type':type(exc).__name__,'error':str(exc)})
 return result

def after_cleanup(directory,label,owned):
 try:return {'registration_ref':exclusive(pathlib.Path(directory)/(label+'-registration-after-cleanup.json'),registration_raw(owned)),'publication_errors':[]}
 except BaseException as exc:return {'registration_ref':None,'publication_errors':[{'event':'registration-after-cleanup','error_type':type(exc).__name__,'error':str(exc)}]}
