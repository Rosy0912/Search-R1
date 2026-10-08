"""SourceV5 plus explicitly inherited unchanged optimizer proof; entropy proof separate."""
import pathlib,json,hashlib
SOURCE=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/tools/search-r1-32b-grpo100-source-v6')
SOURCE_SHA='5723faa51151b136d6d1d9c42dc69288b1c16748144c1f9af73cc45183ec49fd'
INITIALIZER_SOURCE_SHA='12da9b40250360f5a9448d546c93949186eed2e35bf80be9250a1b5e1db338be'
INITIALIZER_SOURCE=SOURCE.with_name('search-r1-32b-grpo100-source-v4')
UPSTREAM='e623724532d06e975b61f3b35ec7925dc790c939'
NATIVE={'path':'/dataset/shaoshitong/search-r1-rl/topd/logs/r1-latest-source-v4-cpu-3200/result.json','sha256':'8ecb69cc6bfc5ac8970625a2e7d379b3e49201937fe117b4aa3d7e47b05377d3'}
NATIVE_JOB='3200'

def validate_native_result(r):
 if not (NATIVE_JOB is not None and r.get('job')==NATIVE_JOB and r.get('tests')==26 and r.get('skips')==0 and r.get('failures')==0 and r.get('exit_code')==0 and r.get('cuda_initialized') is False and r.get('source_manifest_sha256')==INITIALIZER_SOURCE_SHA):raise ValueError('Actual latest sourceV4 native26 proof differs')
 return r

def check(c,read_native=False):
 check_fresh_base(c)
 if NATIVE_JOB is None or not all(NATIVE.values()):raise ValueError('SourceV4 native26 binding pending')
 ref=c['source_manifest']
 if pathlib.Path(c['candidate'])!=SOURCE or pathlib.Path(ref['path'])!=SOURCE/'r1-files.json' or ref['sha256']!=SOURCE_SHA:raise ValueError('Exact isolated entropy sourceV5 binding required')
 if c.get('source_upstream_commit')!=UPSTREAM or c.get('source_native_cpu_evidence')!=NATIVE:raise ValueError('Latest sourceV5 upstream/inherited initializer native binding required')
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
 if not (parent['status']=='DEVICE_PARENT_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING' and parent['source_manifest_sha256']==INITIALIZER_SOURCE_SHA and parent['worker_source_sha256']==WORKER_SHA and len(parent['leased_uuids'])==8 and len(set(parent['leased_uuids']))==8 and parent['guardian_errors']==[] and parent['cleanup']['actual_cgroup_empty'] is True and parent['cleanup']['remaining']==[] and parent['parent_cuda_initialized'] is False and parent['baseline_preserved'] is True):raise ValueError('Actual protected device parent failed')
 if not (worker['status']=='DEVICE_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING' and _job(worker['job'])==job and worker['source_manifest_sha256']==INITIALIZER_SOURCE_SHA and worker['pid']==parent['worker_identity']['pid'] and worker['forced_wrapper'] is False and worker['steps']==3 and worker['all_parameter_and_state_equal'] is True and worker['allocator_after_release']=={'allocated':0,'reserved':0} and worker['all_parameters_and_states_cpu'] is True and worker['namespace_restored'] is True and worker['hooks']==['pre','post']*3):raise ValueError('Actual CUDA initializer/math/release proof failed')
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
 if not (review['status']=='PASS' and review['scope']==DEVICE_SCOPE and review['parent_result']==proof['parent_result'] and review['worker_result']==parent['worker_result'] and review['source_manifest_sha256']==INITIALIZER_SOURCE_SHA):raise ValueError('Independent CUDA device proof differs')
 # Bind the complete independently reviewed raw proof, not just the summary.
 for name in ('foreign_raw','source_manifest','parent_manifest'):
  ref=review[name];path=pathlib.Path(ref['path'])
  if path.is_symlink() or root not in path.resolve().parents or hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Reviewed whole CUDA proof drift '+name)
 if review['source_manifest']!={'path':str(INITIALIZER_SOURCE/'r1-files.json'),'sha256':INITIALIZER_SOURCE_SHA}:raise ValueError('Inherited initializer CUDA proof source manifest differs')
 # Fresh current source is independently byte-pinned; inherited proof is never relabeled.
 original_manifest=json.loads((INITIALIZER_SOURCE/'r1-files.json').read_text())
 for rel in ('verl/workers/actor/dp_actor.py','verl/workers/actor/streamed_adamw.py'):
  if hashlib.sha256((INITIALIZER_SOURCE/rel).read_bytes()).hexdigest()!=original_manifest['files'][rel]['sha256']:raise ValueError('Inherited initializer original source bytes drift')
 import ast
 new_actor=ast.parse((SOURCE/'verl/workers/actor/dp_actor.py').read_text());old_actor=ast.parse((INITIALIZER_SOURCE/'verl/workers/actor/dp_actor.py').read_text())
 class Normalize(ast.NodeTransformer):
  def visit_Call(self,node):
   if isinstance(node.func,ast.Attribute) and node.func.attr=='compile':return ast.Constant('LOCAL_ENTROPY_COMPILE')
   return self.generic_visit(node)
 if ast.dump(Normalize().visit(new_actor))!=ast.dump(Normalize().visit(old_actor)):raise ValueError('Inherited initializer actor source differs beyond local entropy compile')
 if (SOURCE/'verl/workers/actor/streamed_adamw.py').read_bytes()!=(INITIALIZER_SOURCE/'verl/workers/actor/streamed_adamw.py').read_bytes():raise ValueError('Inherited initializer helper bytes differ')
 worker=read(parent['worker_result']);terminal=read(review['controller_terminal'])
 job=validate_device_results(parent,worker,terminal)
 if _job(review['job'])!=job:raise ValueError('Independent CUDA device job differs')
 return job


def check_entropy_native(c):
 ref=c.get('source_entropy_native_cpu_evidence')
 if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('Actual sourceV5 native entropy4 proof pending')
 p=pathlib.Path(ref['path']);root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
 if p.is_symlink() or root/'logs' not in p.resolve().parents:raise ValueError('Entropy CPU proof must be own logs')
 raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Actual entropy native4 proof bytes drift')
 r=json.loads(raw)
 _job(r['job'])
 if not (r.get('tests')==4 and r.get('skips')==0 and r.get('failures')==0 and r.get('exit_code')==0 and r.get('cuda_initialized') is False and r.get('source_manifest_sha256')==SOURCE_SHA):raise ValueError('Actual sourceV5 entropy native4 failed')
 return r

ENTROPY_SCOPE='CUDA_ENTROPY_VALUE_GRADIENT_AND_NO_AUTOTUNE_BENCH_NOT_CAPACITY_NOT_TRAINING'
ENTROPY_OPTIONS={k:False for k in ('triton.autotune_pointwise','max_autotune','max_autotune_pointwise','coordinate_descent_tuning','dynamic_scale_rblock')}
ENTROPY_FILE_SHA='e6f0578b0f52b530d52e9fc69968108d67badeb8374f18f39258e3550ac35454'
ENTROPY_ACTOR_SHA='a0569434298344cedbad7fe184995e732c20c4c32aa62f2fd284f1be36f64a44'

ENTROPY_COMPILED_POLICY={'actual_float32_value_atol':1e-4,'actual_float32_value_rtol':1e-5,'actual_float32_gradient_atol':1e-7,'actual_float32_gradient_rtol':1e-4,'actual_bfloat16_max_reference_nextafter_step':1,'eager_comparison':'REPORT_ONLY_NOT_CORE_ACCEPTANCE'}

def validate_entropy_case(case):
 import math
 def finite_stats(r,fields):
  for field in fields:
   v=r[field]
   if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('Nonfinite/invalid entropy comparison statistic')
 for name in ('candidate_vs_original_compiled_value','candidate_vs_original_compiled_gradient','candidate_vs_eager_value','candidate_vs_eager_gradient'):
  r=case[name];grad=name.endswith('gradient');core='original_compiled' in name
  if r['elements']!=(case['length']*case['vocab'] if grad else case['length']):raise ValueError('Full-element entropy comparison incomplete')
  finite_stats(r,('max_absolute_delta','max_relative_delta'))
  if not core:
   # Complete original eager diagnostics remain public; they do not replace or
   # weaken the candidate-vs-original-compiled core comparison.
   finite_stats(r,('atol','rtol'))
   if not isinstance(r['all_elements_passed'],bool):raise ValueError('Eager diagnostic outcome missing')
   continue
  dtype_key='gradient_dtype' if grad else 'output_dtype'
  dtype=case['candidate'][dtype_key]
  if dtype!=case['original_compiled'][dtype_key] or r['all_elements_passed'] is not True:raise ValueError('Compiled entropy dtype/numerical comparison failed')
  if dtype=='torch.float32':
   finite_stats(r,('atol','rtol'))
   if (r['atol'],r['rtol'])!=((1e-7,1e-4) if grad else (1e-4,1e-5)):raise ValueError('Original compiled FP32 tolerance differs')
  elif dtype=='torch.bfloat16':
   finite_stats(r,('max_reference_adjacent_steps',))
   if r['criterion']!='ACTUAL_REFERENCE_BF16_NEXTAFTER_CLOSED_NEIGHBOR_INTERVAL_NO_EXTRA_ABSOLUTE_FLOOR' or type(r['nonadjacent_elements']) is not int or r['nonadjacent_elements']!=0 or type(r['allowed_reference_adjacent_steps']) is not int or r['allowed_reference_adjacent_steps']!=1 or r['max_reference_adjacent_steps']>1:raise ValueError('Original compiled BF16 closed neighbor comparison failed')
   if 'atol' in r or 'rtol' in r:raise ValueError('BF16 neighbor criterion cannot gain absolute tolerance floor')
  else:raise ValueError('Unexpected actual compiled entropy dtype')
 return True

def validate_entropy_results(parent,worker,terminal):
 import math,re
 job=_job(parent['job'])
 if not (parent['status']=='ENTROPY_PARENT_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING' and parent['source_manifest_sha256']==SOURCE_SHA and len(parent['leased_uuids'])==8 and len(set(parent['leased_uuids']))==8 and parent['guardian_errors']==[] and parent['cleanup']['actual_cgroup_empty'] is True and parent['cleanup']['remaining']==[] and parent['parent_cuda_initialized'] is False and parent['baseline_preserved'] is True):raise ValueError('Actual protected entropy parent failed')
 if not (worker['status']=='ENTROPY_DEVICE_CORRECTNESS_AND_BENCH_SUPPRESSION_PASS_NOT_CAPACITY_NOT_TRAINING' and _job(worker['job'])==job and worker['pid']==parent['worker_identity']['pid'] and worker['source_manifest_sha256']==SOURCE_SHA and worker['options']==ENTROPY_OPTIONS and worker['actual_device']['uuid'] in parent['leased_uuids'] and worker['allocator_after_release']=={'allocated':0,'reserved':0}):raise ValueError('Actual entropy worker failed')
 if worker['compiled_comparison_policy']!=ENTROPY_COMPILED_POLICY:raise ValueError('Predeclared compiled/eager comparison scope differs')
 identity=worker['source_identity']
 if identity['entropy_file_sha256']!=ENTROPY_FILE_SHA or identity['actor_source_sha256']!=ENTROPY_ACTOR_SHA:raise ValueError('Actual entropy source identity differs')
 expected=[(4,17,'float32'),(5,17,'bfloat16'),(2,152064,'float32'),(3,152064,'bfloat16'),(6144,152064,'float32'),(6143,152064,'float32'),(6144,152064,'float32'),(6144,152064,'bfloat16'),(6143,152064,'bfloat16'),(6144,152064,'bfloat16')]
 cases=worker['cases']
 if [(r['length'],r['vocab'],r['input_dtype']) for r in cases]!=expected:raise ValueError('Actual full-length/dynamic/repeated entropy cases missing')
 for case in cases:validate_entropy_case(case)
 events=worker['events'];pre=[e for e in events if e['kind']=='ACTUAL_PRECOMPILE' and e['scope']=='candidate']
 if not pre or any(e['kind'] in ('DO_BENCH_CALL','TUNING_ROUTE_CALL') and e['scope']=='candidate' for e in events):raise ValueError('Candidate benchmark suppression unproved')
 flags=('autotune_pointwise','max_autotune','max_autotune_pointwise','dynamic_scale_rblock')
 if any(e['launchers_after']!=1 or any(e['inductor_meta'].get(k) is not False for k in flags) for e in pre):raise ValueError('Actual single-launcher/four-explicit-flag entropy metadata missing')
 # Torch2.4 codegen emits the coordinate-descent metadata key only when enabled.
 # Its absence is source-derived evidence, never fabricated raw false metadata.
 if identity.get('codegen_source_sha256')!='2f0105a59f6e341c2e8a695e02e192e2603a6ef31e6e914c8b6dd33b9a46235a' or any('coordinate_descent_tuning' in e['inductor_meta'] for e in pre):raise ValueError('Pinned disabled coordinate-descent metadata provenance differs')
 if not worker['generated_files']:raise ValueError('Actual generated entropy kernel files missing')
 if not (_job(terminal['job_id'])==job and terminal['returncode']==0 and re.findall(r'\bJobId=([0-9]+)\b',terminal['stdout'])==[str(job)] and re.search(r'\bJobState=COMPLETED\b',terminal['stdout']) and re.search(r'\bExitCode=0:0\b',terminal['stdout'])):raise ValueError('Actual entropy proof terminal failed')
 return job

def check_entropy_correctness(c):
 proof=c.get('source_entropy_correctness')
 if not isinstance(proof,dict) or set(proof)!={'parent_result','independent_review'}:raise ValueError('Actual sourceV5 entropy CUDA proof pending')
 root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
 def read(ref,parse=True):
  if set(ref)!={'path','sha256'}:raise ValueError('Canonical entropy proof ref required')
  path=pathlib.Path(ref['path'])
  if path.is_symlink() or root not in path.resolve().parents:raise ValueError('Entropy proof must be own ROOT')
  raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Actual entropy proof bytes drift')
  return json.loads(raw) if parse else raw
 parent=read(proof['parent_result']);review=read(proof['independent_review'])
 if not (review['status']=='PASS' and review['scope']==ENTROPY_SCOPE and review['source_manifest_sha256']==SOURCE_SHA and review['parent_result']==proof['parent_result'] and review['worker_result']==parent['worker_result']):raise ValueError('Independent entropy proof binding differs')
 native=parent['native_cpu_four'];native_ref={k:native[k] for k in ('path','sha256')}
 if native_ref!=c.get('source_entropy_native_cpu_evidence'):raise ValueError('Entropy native4 proof run differs')
 cpu=check_entropy_native(c);read(native['log'],parse=False)
 if review['actual_native_cpu_contracts']!=4:raise ValueError('Actual entropy CPU4 count differs')
 worker=read(parent['worker_result']);terminal=read(review['controller_terminal']);job=validate_entropy_results(parent,worker,terminal)
 if _job(review['job'])!=job or _job(cpu['job'])!=job:raise ValueError('Independent entropy/CPU proof job differs')
 for name in ('foreign_raw','source_manifest'):read(review[name],parse=False)
 parent_manifest=read(review['parent_manifest'])
 if parent_manifest.get('worker.py')!=parent['worker_source_sha256']:raise ValueError('Actual entropy worker source differs from reviewed parent manifest')
 if review['source_manifest']!=c['source_manifest']:raise ValueError('Reviewed entropy source manifest differs')
 for rec in worker['generated_files']:
  raw=read({k:rec[k] for k in ('path','sha256')},parse=False)
  if len(raw)!=rec['size']:raise ValueError('Generated entropy kernel size differs')
 return job
