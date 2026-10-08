import pathlib,shutil
old=pathlib.Path('work/search-r1-32b-grpo100-tools-v4');new=pathlib.Path('work/search-r1-32b-grpo100-tools-v5');assert not new.exists();shutil.copytree(old,new);(new/'files.json').unlink()
for name in ('prepare.py','README.md','native-cpu.sbatch','production.sbatch'):
 p=new/name;p.write_text(p.read_text().replace('search-r1-32b-grpo100-tools-v4','search-r1-32b-grpo100-tools-v5'))
p=new/'train_entry.py';s=p.read_text().replace('sys,time','sys,time,traceback,tempfile')
idx=s.index('\ndef main():')
addition='''
def persist_first_exception(config,error):
 root=pathlib.Path(config['evidence_root']);target=root/'trainer-first-exception.json'
 try:actual_identity=identity(os.getpid());identity_error=None
 except BaseException as diagnostic:actual_identity=None;identity_error=repr(diagnostic)
 record={'scope':'ACTUAL_TRAINER_FIRST_EXCEPTION_NOT_TRAINING_ACCEPTANCE','stage':'ORIGINAL_MAIN_TASK_BEFORE_RAY_SHUTDOWN','job':os.environ.get('SLURM_JOB_ID'),'pid':os.getpid(),'epoch':time.time(),'error_type':type(error).__name__,'message':str(error),'traceback':''.join(traceback.format_exception(type(error),error,error.__traceback__)),'identity':actual_identity,'identity_error':identity_error,'effective_config_sha256':sha(root/'effective-runtime-config.json')}
 fd,name=tempfile.mkstemp(prefix='.trainer-first-exception-',suffix='.private',dir=root)
 temp=pathlib.Path(name)
 try:
  os.fchmod(fd,0o600)
  with os.fdopen(fd,'w') as f:json.dump(record,f,sort_keys=True);f.flush();os.fsync(f.fileno())
  os.link(temp,target) # Exclusive first-error publication, never overwrite.
  directory=os.open(root,os.O_RDONLY)
  try:os.fsync(directory)
  finally:os.close(directory)
 finally:
  try:temp.unlink()
  except FileNotFoundError:pass
 return record

def run_original_main(main_task,resolved,shutdown,config):
 print('ORIGINAL_MAIN_TASK_BEGIN',flush=True)
 try:
  result=main_task(resolved)
  print('ORIGINAL_MAIN_TASK_RETURNED_AFTER_TERMINAL_AUTHORIZATION',flush=True)
  return result
 except BaseException as error:
  try:persist_first_exception(config,error)
  except BaseException as persistence:
   # Preserve the first original exception, report persistence failure before cleanup.
   error.first_exception_persistence_error=repr(persistence)
   print('TRAINER_FIRST_EXCEPTION_PERSISTENCE_FAILED '+repr(persistence),file=sys.stderr,flush=True)
  print('ORIGINAL_MAIN_TASK_EXCEPTION '+type(error).__name__+': '+str(error),file=sys.stderr,flush=True)
  raise
 finally:
  print('ORIGINAL_RAY_SHUTDOWN_BEGIN',flush=True)
  shutdown()
'''
s=s[:idx]+addition+s[idx:];s=s.replace(' try:entry.main_task._function(resolved)\n finally:ray.shutdown()',' run_original_main(entry.main_task._function,resolved,ray.shutdown,c)');p.write_text(s)
p=new/'native_entry.py';p.write_text(p.read_text().replace("str(HERE/'test_observer_factory.py')","str(HERE/'test_observer_factory.py'),str(HERE/'test_trainer_first_exception.py')").replace('expected=36','expected=40'))
p=new/'README.md';p.write_text(p.read_text()+'''\n## V5 first-fault observability only\n\n3163 failed after initial64 and actual16 READY. Actual Ray driver log reports intended ray.shutdown before guardian rejected exited workers; original first Python exception was not persisted. V5 calls the unchanged original main_task under exception/then-finally original ray.shutdown and durably publishes trainer-first-exception.json (exclusive0600 tempfile fsync→hardlink→directoryfsync) before shutdown. Record contains actual exception traceback/type/message/stage/job/PID/epoch/process identity/configSHA; no capability or fullconfig is logged. Publication failure is explicitly flushed to stderr and attached to the original exception; original failure remains fatal. Normal completed terminal-hold returns do not write a first-error receipt. Original shutdown remains called even on failure. This fixes observability only; it does not assert any training root cause or success. KVbudget0.4 and original source50fcaec unchanged. Native expected40; actual profile pending.\n''')
