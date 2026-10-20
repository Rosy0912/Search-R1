"""Hash-bound process-local layout adapters; original scientific files untouched."""
import ast,copy,importlib.abc,importlib.machinery,inspect,json,os,sys,textwrap
from contextvars import ContextVar
from pathlib import Path
from .contract import active,read_plan,projected_config,reshard_original_eight_rows,sha
_installed=False
_projecting=ContextVar('cgrts16_projecting',default=False)
def _project_call(original,config):
 if not active() or _projecting.get():return original(config)
 projected=projected_config(config);token=_projecting.set(True)
 try:return original(projected)
 finally:_projecting.reset(token)
def _check(module):
 plan=read_plan();rel=str(Path(module.__file__).relative_to(Path(plan['source'])))
 assert sha(module.__file__)==plan['parent_source_files'][rel],rel
def _contract(module):
 original=module.validate_recursive_config
 def wrapped(config):return _project_call(original,config)
 module.validate_recursive_config=wrapped
def _controller(module):
 original=module.validate_behavior_mu_config
 def wrapped(config):return _project_call(original,config)
 module.validate_behavior_mu_config=wrapped
def _ledger(module):
 old=module.validate_ledger;src=textwrap.dedent(inspect.getsource(old))
 assert src.count("entry['rows'] != 16")==1 and src.count('range(320)')==1
 changed=src.replace("entry['rows'] != 16","entry['rows'] != 8").replace('range(320)','range(160)')
 original_ast=ast.parse(src);new_ast=ast.parse(changed)
 def normalize(tree):
  tree=copy.deepcopy(tree)
  for n in ast.walk(tree):
   if isinstance(n,ast.Constant) and type(n.value) is int and n.value in (8,160):n.value={8:16,160:320}[n.value]
  return ast.dump(tree,include_attributes=False)
 assert normalize(new_ast)==ast.dump(original_ast,include_attributes=False)
 space=dict(module.__dict__);exec(compile(new_ast,module.__file__,'exec'),space);replacement=space['validate_ledger']
 module.validate_ledger=lambda ledger,writes,allow_zero=True: replacement(ledger,writes,allow_zero) if active() else old(ledger,writes,allow_zero)
def _runtime(module):
 original=module.runtime_env_vars
 def wrapped():
  v=original()
  if active():v.update({k:os.environ[k] for k in ['CGRTS16_PARALLEL_PLAN','PYTHONPATH']})
  return v
 module.runtime_env_vars=wrapped
def _lease(module):
 old=module.validate_gpu_lease
 def wrapped(config):
  if not active():return old(config)
  projected_config(config);module.validate_inherited_gpu_leases(8)
  from .runtime import verify_node_records
  verify_node_records(read_plan())
 module.validate_gpu_lease=wrapped
def _local(module):
 old=module.check_local_runtime
 def wrapped(config,ray_nodes=None):
  if not active():return old(config,ray_nodes)
  projected_config(config)
  from .runtime import validate_cluster_boundary
  return validate_cluster_boundary(config,ray_nodes)
 module.check_local_runtime=wrapped
def _trainer(module):
 cls=module.RayPPOTrainer;oldfit=cls.fit;src=textwrap.dedent(inspect.getsource(oldfit))
 assert src.count("len(receipts) != 8")==1 and src.count("{r['rank'] for r in receipts} != set(range(8))")==1
 changed=src.replace('len(receipts) != 8','len(receipts) != self.actor_rollout_wg.world_size').replace("{r['rank'] for r in receipts} != set(range(8))","{r['rank'] for r in receipts} != set(range(self.actor_rollout_wg.world_size))")
 # Only startup rank cardinality changes; execute the entire original fit body.
 assert changed.replace('len(receipts) != self.actor_rollout_wg.world_size','len(receipts) != 8').replace("{r['rank'] for r in receipts} != set(range(self.actor_rollout_wg.world_size))","{r['rank'] for r in receipts} != set(range(8))")==src
 marker="        if current_chain_resume:\n"
 assert changed.count(marker)==1
 inserted="        from reproduction.cgrts16_parallel_v1.runtime import admit_full_initialization\n        admit_full_initialization(self)\n"
 changed=changed.replace(marker,inserted+marker)
 space=dict(module.__dict__);exec(compile(ast.parse(changed),module.__file__,'exec'),space);cls.fit=space['fit']
 oldbalance=cls._balance_batch
 def balance(self,batch,metrics,logging_prefix='global_seqlen'):
  if not active():return oldbalance(self,batch,metrics,logging_prefix)
  assert self.actor_rollout_wg.world_size==16 and len(batch)==2560
  from types import SimpleNamespace
  import torch
  oldbalance(SimpleNamespace(actor_rollout_wg=SimpleNamespace(world_size=8)),batch,metrics,logging_prefix)
  batch.reorder(torch.tensor(reshard_original_eight_rows(len(batch))))
  metrics['layout/original_eight_global_optimizer_groups_preserved']=1
 cls._balance_batch=balance
def _evaluation(module):
 original=module.batch_answer_metrics
 def wrapped(data,tokenizer):
  rows=original(data,tokenizer)
  if active():
   from .runtime import observe_generation
   observe_generation(data,tokenizer,rows)
  return rows
 module.batch_answer_metrics=wrapped
CALLBACKS={'search_r1.baselines.tree_fresh200':_contract,'search_r1.baselines.controller':_controller,
 'search_r1.baselines.tree_guard_continue':_ledger,'search_r1.baselines.runtime_audit':_runtime,
 'search_r1.baselines.preflight':_lease,'search_r1.baselines.local_runtime':_local,
 'verl.trainer.ppo.ray_trainer':_trainer,'search_r1.baselines.evaluation':_evaluation}
class Loader(importlib.abc.Loader):
 def __init__(self,original,name):self.original,self.name=original,name
 def __getattr__(self,name):return getattr(self.original,name)
 def create_module(self,spec):return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
 def exec_module(self,module):
  self.original.exec_module(module);_check(module);CALLBACKS[self.name](module)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname not in CALLBACKS:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  if spec and spec.loader:spec.loader=Loader(spec.loader,fullname)
  return spec
def install():
 global _installed
 if _installed:return
 assert active();read_plan();_installed=True;sys.meta_path.insert(0,Finder())
 for name,callback in CALLBACKS.items():
  if name in sys.modules:_check(sys.modules[name]);callback(sys.modules[name])
def worker_setup():install()
