"""Run the original standard main_task, retaining its fit frame until parent drain."""
import argparse,json,os,pathlib,socket,struct,sys,time,traceback,tempfile
sys.dont_write_bytecode=True
from allocator_policy import bootstrap,check_config,check_environment,worker_environment
if __name__=='__main__':bootstrap()
from pipeline import atomic
from ipc import identity,send,read
from checkpoint_inventory import sha
from source_binding import check as check_source_v4,check_device_correctness,check_entropy_correctness

def hold(original,trainer,config,binding):
 result=original(trainer)
 root=pathlib.Path(config['evidence_root']);outer=pathlib.Path(config['actual_outer_receipt_path'])
 if not outer.is_file():raise RuntimeError('Actual outer receipt absent')
 request={'operation':'TERMINAL_HELD','identity':identity(os.getpid()),'job':os.environ['SLURM_JOB_ID'],'actual_outer_evidence':{'path':str(outer),'sha256':sha(outer)},'effective_config_sha256':sha(root/'effective-runtime-config.json'),'held_epoch':time.time()}
 atomic(root/'trainer-terminal-held.json',request)
 with socket.socket(socket.AF_UNIX) as sock:
  sock.settimeout(14400);sock.connect(binding['socket']);pid,uid,_=struct.unpack('3i',sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
  if uid!=os.getuid() or identity(pid)!=binding['server_identity']:raise RuntimeError('Terminal parent peer changed')
  send(sock,dict(request,capability=binding['capability']));response=read(sock)
  if response.get('status')!='AUTHORIZED_OWN_SHUTDOWN' or response.get('held_sha256')!=sha(root/'trainer-terminal-held.json'):raise RuntimeError('Unknown terminal ACK; no retry')
  atomic(root/'trainer-terminal-authorization-received.json',{'received_epoch':time.time(),**response})
 return result

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

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--binding',required=True);a=ap.parse_args()
 c=json.loads(pathlib.Path(a.config).read_text());binding=json.loads(pathlib.Path(a.binding).read_text());candidate=pathlib.Path(c['candidate']).resolve();here=pathlib.Path(__file__).resolve().parent
 check_config(c);check_environment();check_source_v4(c,read_native=True);check_device_correctness(c);check_entropy_correctness(c)
 if sha(c['source_manifest']['path'])!=c['source_manifest']['sha256']:raise ValueError('Streamed sourceV4 manifest bytes drift')
 os.chdir(candidate);sys.path.insert(0,str(candidate));sys.path.insert(0,str(here));os.environ['PYTHONPATH']=str(here)+os.pathsep+os.environ.get('PYTHONPATH','')
 from hydra import compose,initialize_config_dir
 with initialize_config_dir(config_dir=str(candidate/'verl/trainer/config'),version_base=None):resolved=compose(config_name='ppo_trainer',overrides=c['training_command'][3:])
 from observed_worker import install
 import verl.workers.fsdp_workers as workers
 install(workers.ActorRolloutRefWorker)
 # Explicit value serialization keeps injected methods in actual Ray processes.
 import ray.cloudpickle
 ray.cloudpickle.register_pickle_by_value(workers)
 import verl.trainer.main_ppo as entry
 original_fit=entry.RayPPOTrainer.fit
 class HeldTrainer(entry.RayPPOTrainer):
  def fit(self):return hold(original_fit,self,c,binding)
 entry.RayPPOTrainer=HeldTrainer
 import ray
 os.environ['R1_RESOURCE_AUDIT_DIRECTORY']=str(pathlib.Path(c['evidence_root'])/'actor-resource-audit')
 env=worker_environment(os.environ)
 env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='true',NCCL_DEBUG='WARN')
 temp=pathlib.Path('/dev/shm')/('r1-'+os.environ['SLURM_JOB_ID']);temp.mkdir(mode=0o700,exist_ok=False)
 os.environ.pop('RAY_ADDRESS',None);os.environ.pop('RAY_BOOTSTRAP_ADDRESS',None)
 ray.init(address='local',num_cpus=24,include_dashboard=False,_temp_dir=str(temp),runtime_env={'env_vars':env})
 run_original_main(entry.main_task._function,resolved,ray.shutdown,c)
if __name__=='__main__':main()
