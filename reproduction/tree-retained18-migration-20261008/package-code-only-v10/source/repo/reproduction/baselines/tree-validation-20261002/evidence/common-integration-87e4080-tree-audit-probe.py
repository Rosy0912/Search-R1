import pathlib,tempfile,os,json,torch,numpy as np,ast,subprocess,inspect
from unittest.mock import patch
from tests.baselines.test_paper_metric_logging import run
from search_r1.baselines import runtime_audit as audit
root=pathlib.Path.cwd();git=lambda ref,p:subprocess.check_output(['git','show',ref+':'+p]).decode();base='f4f457df55fbfe99691d466d5c6f1d3fd7175c65'
def funcs(text,class_name):
 tree=ast.parse(text);return {n.name:ast.dump(n,include_attributes=False) for n in next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name).body if isinstance(n,ast.FunctionDef)}
p='search_r1/baselines/controller.py';old=funcs(git(base,p),'Controller');new=funcs((root/p).read_text(),'Controller');names=['generate','audit_actor_batch','main_f1_reward','compute_advantages','after_actor','stage_validation','save'];assert all(old[n]==new[n] for n in names)
class Strip(ast.NodeTransformer):
 def visit_Assign(self,n):
  if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='audit':return None
  return n
 def visit_If(self,n):
  if isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='audit':return None
  if isinstance(n.test,ast.BoolOp) and any(isinstance(x,ast.Compare) and isinstance(x.left,ast.Name) and x.left.id=='audit' for x in n.test.values):return None
  if isinstance(n.test,ast.Compare) and any(isinstance(x,ast.Tuple) and [getattr(y,'value',None) for y in x.elts]==['grpo','gigpo'] for x in n.test.comparators):return None
  return self.generic_visit(n)
p='verl/workers/actor/dp_actor.py';assert ast.dump(Strip().visit(ast.parse((root/p).read_text())),include_attributes=False)==ast.dump(ast.parse(git(base,p)),include_attributes=False)
state=torch.random.get_rng_state().clone();off,wo=run(1,'tree_grpo');end=torch.random.get_rng_state().clone()
with tempfile.TemporaryDirectory() as td:
 torch.random.set_rng_state(state)
 with patch.dict(os.environ,{'SEARCH_R1_BASELINE_AUDIT_DIR':td}),patch.object(audit,'ActorAudit',side_effect=AssertionError('Common ActorAudit must not execute on Tree')),patch.object(audit,'emit',side_effect=AssertionError('Common driver_outer must skip Tree')):
  audit.driver_outer(object(),1,'tree_grpo');on,wn=run(1,'tree_grpo');assert torch.equal(wo,wn) and torch.equal(end,torch.random.get_rng_state()) and off==on and not list(pathlib.Path(td).iterdir())
  text=inspect.getsource(run).replace('def run(', 'def run_large(').replace('ppo_mini_batch_size=2','ppo_mini_batch_size=256').replace('(2,1)','(256,1)').replace('(2,2)','(256,2)').replace("np.asarray(['q','q'])","np.asarray(['q']*256)")
  ns=dict(run.__globals__);exec(text,ns);small,ws=ns['run_large'](8,'tree_grpo');large,wl=ns['run_large'](64,'tree_grpo');assert torch.equal(ws,wl)
  for key in ['actor/pg_loss','actor/entropy_loss','actor/kl_loss','actor/pg_clipfrac','actor/ppo_kl']:np.testing.assert_allclose(small[key],large[key],atol=1e-7)
out={'head':'87e4080fe94869c9f1978b9abf7b3931c0141f54','actor_ast_equals_f4_after_new_grpo_gigpo_audit_removed':True,'controller_tree_methods_ast_equal_f4':names,'real_tree_actor_common_audit_on_off_parameter_metrics_rng_bitwise_equal':True,'common_actor_audit_and_driver_outer_skip_tree_no_io':True,'toy_real_actor_256rows_micro8_vs64_objective_parameter_equal':True,'limitations':'CPU real actor/TensorDict and fake model only; micro64 shared GPU physical recipe needs new actual3B smoke, not production bitwise numerical equivalence'}
(root/'reproduction/baselines/tree-validation-20261002/evidence/common-integration-87e4080-tree-audit-probe.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
