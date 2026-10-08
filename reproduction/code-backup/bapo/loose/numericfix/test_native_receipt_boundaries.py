"""Exercise actual verifier AST rank validation with synthetic CPU evidence only."""
import ast,json,math,tempfile,unittest
from pathlib import Path
source=Path('work/numericfix/remote/verify_hardware_receipt.py').read_text()
tree=ast.parse(source);verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
loop=next(n for n in verify.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='rank')
fn=ast.FunctionDef(name='validate',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='root'),ast.arg(arg='run'),ast.arg(arg='seen')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[loop],decorator_list=[])
module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));ns={'Path':Path,'json':json,'math':math,'SOURCE':'1cceda2f7e201de3f68c876bb9a3b895563c20ce'};exec(compile(module,'actual-hardware-verifier-rank-block','exec'),ns)
class RankEvidence(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.run=self.root/'runs/new';self.folder=self.run/'numeric-fixture';self.folder.mkdir(parents=True);self.seen=set()
  cases=[{'case':case,'orig':orig,'offload':offload,'norm':1.0} for case in ['normal_native_and_fullvector_oracle','finite_gradient_fp32_square_overflow','one_rank_nonsharded_nan_collective_rejection','empty_one_rank_sharded_gradients'] for orig in [False,True] for offload in [False,True]]
  cases += [{'case':case,'orig':orig,'norm':0.} for case in ['pure_NO_SHARD_all_empty_gradients','pure_NO_SHARD_one_rank_inf_rejection']for orig in [False,True]]
  cases += [{'case':'gpu_low_var_kl_saturated_forward_backward_fp64_oracle','points':408,'max_value_error':1e-6,'max_grad_error':1e-6},{'case':'gpu_low_var_kl_nonfinite_inputs_refused','count':3}]
  for rank in range(8):
   p=self.folder/('rank-'+str(rank)+'.json');p.write_text(json.dumps({'status':'PASS','source_commit':ns['SOURCE'],'rank':rank,'world':8,'records':cases,'gradient_norm_file':str(self.root/'repo/search_r1/baselines/gradient_norm.py'),'core_algos_file':str(self.root/'repo/verl/trainer/ppo/core_algos.py')}));self.seen.add(str(p))
 def mutate(self,fun):
  p=self.folder/'rank-0.json';value=json.loads(p.read_text());fun(value);p.write_text(json.dumps(value))
 def validate(self):ns['validate'](self.root,self.run,self.seen)
 def test_positive_synthetic_evidence(self):self.validate()
 def test_old_source_rejected(self):
  self.mutate(lambda r:r.update(source_commit='b09736d9d1b487a5c134e7a358761e04613cea32'))
  with self.assertRaises(ValueError):self.validate()
 def test_missing_rank_binding_rejected(self):
  self.seen.remove(str(self.folder/'rank-7.json'))
  with self.assertRaises(ValueError):self.validate()
 def test_duplicate_case_rejected(self):
  self.mutate(lambda r:r['records'].__setitem__(0,r['records'][1]))
  with self.assertRaises(ValueError):self.validate()
 def test_missing_point_rejected(self):
  self.mutate(lambda r:r['records'][-2].update(points=407))
  with self.assertRaises(ValueError):self.validate()
 def test_nan_oracle_error_rejected(self):
  self.mutate(lambda r:r['records'][-2].update(max_grad_error=float('nan')))
  with self.assertRaises(ValueError):self.validate()
 def test_error_outside_tolerance_rejected(self):
  self.mutate(lambda r:r['records'][-2].update(max_value_error=1e-3))
  with self.assertRaises(ValueError):self.validate()
 def test_missing_nonfinite_case_rejected(self):
  self.mutate(lambda r:r['records'][-1].update(count=2))
  with self.assertRaises(ValueError):self.validate()
 def test_wrong_imported_source_rejected(self):
  self.mutate(lambda r:r.update(core_algos_file='/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097/repo/verl/trainer/ppo/core_algos.py'))
  with self.assertRaises(ValueError):self.validate()
if __name__=='__main__':unittest.main(verbosity=2)
