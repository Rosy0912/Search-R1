"""Exact isolated sourceV4 identity shared by preparation and actual launch."""
import pathlib,json,hashlib
SOURCE=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/tools/search-r1-32b-grpo100-source-v4')
SOURCE_SHA='12da9b40250360f5a9448d546c93949186eed2e35bf80be9250a1b5e1db338be'
UPSTREAM='e623724532d06e975b61f3b35ec7925dc790c939'
NATIVE={'path':'/dataset/shaoshitong/search-r1-rl/topd/logs/r1-latest-source-v4-cpu-3200/result.json','sha256':'8ecb69cc6bfc5ac8970625a2e7d379b3e49201937fe117b4aa3d7e47b05377d3'}
NATIVE_JOB='3200'

def validate_native_result(r):
 if not (NATIVE_JOB is not None and r.get('job')==NATIVE_JOB and r.get('tests')==26 and r.get('skips')==0 and r.get('failures')==0 and r.get('exit_code')==0 and r.get('cuda_initialized') is False and r.get('source_manifest_sha256')==SOURCE_SHA):raise ValueError('Actual latest sourceV4 native26 proof differs')
 return r

def check(c,read_native=False):
 check_fresh_base(c)
 if NATIVE_JOB is None or not all(NATIVE.values()):raise ValueError('SourceV4 native26 binding pending')
 ref=c['source_manifest']
 if pathlib.Path(c['candidate'])!=SOURCE or pathlib.Path(ref['path'])!=SOURCE/'r1-files.json' or ref['sha256']!=SOURCE_SHA:raise ValueError('Exact isolated streamed sourceV4 binding required')
 if c.get('source_upstream_commit')!=UPSTREAM or c.get('source_native_cpu_evidence')!=NATIVE:raise ValueError('Latest sourceV4 upstream/native binding required')
 if read_native:
  raw=pathlib.Path(NATIVE['path']).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=NATIVE['sha256']:raise ValueError('Native26 result bytes differ')
  validate_native_result(json.loads(raw))
 return SOURCE_SHA


def check_fresh_base(c):
 command=c['training_command']
 if len(command)<4 or command[1:3]!=['-m','verl.trainer.main_ppo']:raise ValueError('Original main_ppo command required')
 overrides={}
 for argument in command[3:]:
  if '=' not in argument:raise ValueError('Invalid original Hydra override')
  name,value=argument.split('=',1);name=name.lstrip('+')
  if name in overrides:raise ValueError('Duplicate Hydra override rejected')
  overrides[name]=value
 if overrides.get('trainer.weight_continuation_step')!='0':raise ValueError('Fresh base requires continuation step zero')
 if overrides.get('actor_rollout_ref.model.actor_checkpoint_path','null')!='null':raise ValueError('Fresh base forbids actor checkpoint initialization')
 return True


DEVICE_SCOPE='CUDA_DEVICE_INITIALIZER_AND_ORIGINAL_ADAMW_MATH_NOT_CAPACITY_NOT_TRAINING'
WORKER_SHA='ea2ac5b3a5237e29f3c052ab97883fecc55516f24bc896a546514cbcb86981ba'
def _job(value):
 import re
 if isinstance(value,bool) or not isinstance(value,(int,str)) or re.fullmatch('[1-9][0-9]*',str(value),flags=re.ASCII) is None:raise ValueError('Invalid actual device-proof job')
 return int(value)
def validate_device_results(parent,worker,terminal):
 import re
 job=_job(parent['job'])
 if not (parent['status']=='DEVICE_PARENT_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING' and parent['source_manifest_sha256']==SOURCE_SHA and parent['worker_source_sha256']==WORKER_SHA and len(parent['leased_uuids'])==8 and len(set(parent['leased_uuids']))==8 and parent['guardian_errors']==[] and parent['cleanup']['actual_cgroup_empty'] is True and parent['cleanup']['remaining']==[] and parent['parent_cuda_initialized'] is False and parent['baseline_preserved'] is True):raise ValueError('Actual protected device parent failed')
 if not (worker['status']=='DEVICE_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING' and _job(worker['job'])==job and worker['source_manifest_sha256']==SOURCE_SHA and worker['pid']==parent['worker_identity']['pid'] and worker['forced_wrapper'] is False and worker['steps']==3 and worker['all_parameter_and_state_equal'] is True and worker['allocator_after_release']=={'allocated':0,'reserved':0} and worker['all_parameters_and_states_cpu'] is True and worker['namespace_restored'] is True and worker['hooks']==['pre','post']*3):raise ValueError('Actual CUDA initializer/math/release proof failed')
 meta=worker['initializer_metadata_after_compile_context']
 if not (meta['wrapped'] is True and meta['dynamo_disabled'] is True and meta['orig_callable_is_body'] is True and worker['production_entropy_compile_context']['dynamic'] is True and worker['production_entropy_compile_context']['invoked'] is False):raise ValueError('Natural production compile-context proof failed')
 events=worker['initializer_events']
 if len(events)!=4 or any(not event['input_device'].startswith('cuda:') or event['output_device']!='cpu' or event['dtype']!='torch.float32' for event in events):raise ValueError('Original CUDA-input CPU-initializer events differ')
 if not (_job(terminal['job_id'])==job and terminal['returncode']==0 and re.findall(r'\bJobId=([0-9]+)\b',terminal['stdout'])==[str(job)] and re.search(r'\bJobState=COMPLETED\b',terminal['stdout']) and re.search(r'\bExitCode=0:0\b',terminal['stdout'])):raise ValueError('Actual device parent terminal failed')
 return job

def check_device_correctness(c):
 proof=c.get('source_device_correctness')
 if not isinstance(proof,dict) or set(proof)!={'parent_result','independent_review'}:raise ValueError('Actual CUDA source device proof pending')
 root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
 def read(ref):
  p=pathlib.Path(ref['path'])
  if p.is_symlink() or root/'logs' not in p.resolve().parents:raise ValueError('Actual device proof must be own logs')
  raw=p.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Actual device proof SHA drift')
  return json.loads(raw)
 parent=read(proof['parent_result']);review=read(proof['independent_review'])
 if not (review['status']=='PASS' and review['scope']==DEVICE_SCOPE and review['parent_result']==proof['parent_result'] and review['worker_result']==parent['worker_result'] and review['source_manifest_sha256']==SOURCE_SHA):raise ValueError('Independent CUDA device proof differs')
 # Bind the complete independently reviewed raw proof, not just the summary.
 for name in ('foreign_raw','source_manifest','parent_manifest'):
  ref=review[name];path=pathlib.Path(ref['path'])
  if path.is_symlink() or root not in path.resolve().parents or hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Reviewed whole CUDA proof drift '+name)
 if review['source_manifest']!=c['source_manifest']:raise ValueError('CUDA proof source manifest differs')
 worker=read(parent['worker_result']);terminal=read(review['controller_terminal'])
 job=validate_device_results(parent,worker,terminal)
 if _job(review['job'])!=job:raise ValueError('Independent CUDA device job differs')
 return job
