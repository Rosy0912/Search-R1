"""Source-isolated tests; before/after origin evidence is mandatory, never GPU proof."""
from pathlib import Path
import sys,os,types,json,hashlib,importlib,subprocess,struct
CRITICAL=('verl','search_r1','verl.protocol','verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos','search_r1.baselines.distillation','search_r1.baselines.objectives','search_r1.baselines.full_state','search_r1.baselines.checkpoint_gate','search_r1.baselines.gradient_norm','search_r1.baselines.actor_metrics','search_r1.baselines.recovery')
SOURCE='3f85dabc7aa05bc45ec7af015d65d9fa4a4c17e4'
STABILITY_TESTS=('tests/baselines/test_actor_metrics.py','tests/baselines/test_bapo_stability_patch.py','tests/baselines/test_e1_checkpoint_gate.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def namespace_origins(repo,load_critical=True):
 if load_critical:
  for name in CRITICAL:importlib.import_module(name)
 result={}
 for name,module in list(sys.modules.items()):
  if name not in ('verl','search_r1')and not name.startswith(('verl.','search_r1.')):continue
  file=getattr(module,'__file__',None);assert file,'Missing actual source file '+name
  path=Path(file).resolve();assert repo in path.parents and path.is_file(),'Outside repository import '+name+': '+str(path)
  for directory in getattr(module,'__path__',[]):assert repo in Path(directory).resolve().parents,'Outside package search path '+name
  result[name]=dict(path=str(path),sha256=sha(path))
 if load_critical:assert set(CRITICAL)<=set(result)
 return result
def code_digest(code):
 # Marshal ref/intern flags can change after execution. Hash immutable semantic fields.
 def stable(value):
  if isinstance(value,types.CodeType):
   fields=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags','co_code','co_consts','co_names','co_varnames','co_filename','co_name','co_qualname','co_firstlineno','co_lnotab','co_linetable','co_exceptiontable','co_freevars','co_cellvars')
   return {'code':{name:stable(getattr(value,name))for name in fields if hasattr(value,name)}}
  if isinstance(value,bytes):return {'bytes':value.hex()}
  if isinstance(value,str):return {'str':value}
  if isinstance(value,float):return {'float_bits':struct.pack('>d',value).hex()}
  if isinstance(value,complex):return {'complex':[stable(value.real),stable(value.imag)]}
  if isinstance(value,tuple):return {'tuple':[stable(v)for v in value]}
  if isinstance(value,frozenset):return {'frozenset':sorted((stable(v)for v in value),key=lambda v:json.dumps(v,sort_keys=True))}
  if value is Ellipsis:return {'ellipsis':True}
  if value is None or type(value)in (int,bool):return {type(value).__name__:value}
  raise AssertionError('Unsupported actual code constant '+str(type(value)))
 return hashlib.sha256(json.dumps(stable(code),sort_keys=True,separators=(',',':')).encode()).hexdigest()
FUNCTIONS={
 'verl.workers.actor.dp_actor':('DataParallelPPOActor.update_policy','DataParallelPPOActor.compute_log_prob','DataParallelPPOActor._optimizer_step'),
 'verl.trainer.ppo.core_algos':('kl_penalty','compute_policy_loss'),
 'search_r1.baselines.distillation':('global_sum_count','global_token_count','global_token_denominator'),
 'search_r1.baselines.objectives':('row_denominator','weighted_policy_loss'),
 'search_r1.baselines.actor_metrics':('validate_actor_metric_topology','merge_actor_metric_records','aggregate_sequence_mean_actor_metrics'),
   'search_r1.baselines.full_state':('state_digest','atomic_save'),
 'search_r1.baselines.checkpoint_gate':('validate_release',),
 'search_r1.baselines.recovery':('validate_manifest','validate_native','validate_driver'),}
def function_origins(repo):
 result={}
 for module,names in FUNCTIONS.items():
  loaded=importlib.import_module(module);expected=repo.joinpath(*module.split('.')).with_suffix('.py')
  for dotted in names:
   fn=loaded
   for part in dotted.split('.'):fn=getattr(fn,part)
   code=fn.__code__;path=Path(code.co_filename).resolve()
   assert path==expected and repo in path.parents,'Outside function code '+module+'.'+dotted+': '+str(path)
   assert fn.__module__==module
   result[module+'.'+dotted]=dict(path=str(path),sha256=sha(path),co_filename=code.co_filename,co_firstlineno=code.co_firstlineno,function_module=fn.__module__,qualname=fn.__qualname__,code_sha256=code_digest(code))
 return result
def isolate_repo(repo):
 repo=Path(repo).resolve();assert repo.is_dir()and (repo/'verl/__init__.py').is_file()and (repo/'search_r1/__init__.py').is_file()
 # Never silently reload an old cached module; refuse contaminated startup.
 namespace_origins(repo,False)
 sys.path[:]=[str(repo)]+[p for p in sys.path if p!=str(repo)]
 # Spawned multiprocessing/Gloo interpreters receive the same actual namespace priority.
 os.environ['PYTHONPATH']=str(repo)
 importlib.invalidate_caches();return repo
if __name__=='__main__':
 repo=isolate_repo(Path.cwd());assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE
 assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
 before=namespace_origins(repo);before_functions=function_origins(repo)
 stability_tests={name:sha(repo/name)for name in STABILITY_TESTS}
 print('STABILITY_TEST_FILES_JSON='+json.dumps(stability_tests,sort_keys=True),flush=True)
 print('CPU_IMPORT_ORIGINS_BEFORE_JSON='+json.dumps(dict(repo=str(repo),source_commit=SOURCE,modules=before,functions=before_functions),sort_keys=True),flush=True)
 if sys.argv[1:]==['--origins-only']:
  after=namespace_origins(repo);after_functions=function_origins(repo);assert before==after and before_functions==after_functions
  print('CPU_IMPORT_ORIGINS_AFTER_JSON='+json.dumps(dict(repo=str(repo),source_commit=SOURCE,modules=after,functions=after_functions),sort_keys=True),flush=True)
  raise SystemExit(0)
 assert not sys.argv[1:]
 os.environ.pop('SEARCH_R1_ASSETS',None);os.environ['GLOO_SOCKET_IFNAME']='lo0'if sys.platform=='darwin'else'lo'
 for name,directory in [('tests',repo/'tests'),('tests.baselines',repo/'tests/baselines')]:
  module=types.ModuleType(name);module.__path__=[str(directory)];sys.modules[name]=module
 import pytest
 exit_code=pytest.main(['-q','tests/baselines','tests/reproduction','tests/recovery'])
 after=namespace_origins(repo);after_functions=function_origins(repo)
 assert before_functions==after_functions,'Function implementation changed during tests '+json.dumps({k:[before_functions[k],after_functions[k]] for k in before_functions if before_functions[k]!=after_functions[k]},sort_keys=True)
 assert all(after[name]==value for name,value in before.items()),'Loaded source origin or bytes changed during tests'
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE
 assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
 assert stability_tests=={name:sha(repo/name)for name in STABILITY_TESTS}
 print('STABILITY_TEST_FILES_JSON='+json.dumps(stability_tests,sort_keys=True),flush=True)
 print('CPU_IMPORT_ORIGINS_AFTER_JSON='+json.dumps(dict(repo=str(repo),source_commit=SOURCE,modules=after,functions=after_functions),sort_keys=True),flush=True)
 raise SystemExit(exit_code)
