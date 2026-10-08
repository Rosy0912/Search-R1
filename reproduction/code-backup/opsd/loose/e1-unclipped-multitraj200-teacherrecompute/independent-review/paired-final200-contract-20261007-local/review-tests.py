from pathlib import Path
import sys,types,unittest,copy,ast
D=Path(sys.argv[1])
p=D/'test_paired_eval_contract.py'
m=types.ModuleType('independent_candidate_tests');m.__file__=str(p)
exec(compile(p.read_bytes(),str(p),'exec',dont_inherit=True),m.__dict__)
c=m.c
class IndependentBoundaryTests(unittest.TestCase):
 def test_teacher_role_cannot_follow_hf200_actor(self):
  v=m.fixture();m.assign(v['roles']['actual_hf200']['resolved_config'],'baselines.distillation.teacher_path',c.HF200)
  with self.assertRaises(ValueError):c.validate_control_inputs(v)
 def test_coefficient_bool_does_not_pass_zero_and_proposed_stage_copy_isolated(self):
  v=m.fixture();m.assign(v['roles']['official_base']['resolved_config'],'algorithm.kl_ctrl.kl_coef',False)
  with self.assertRaises(ValueError):c.validate_control_inputs(v)
  a=c.proposed_role_stage(m.fixture(),'official_base');a['generation_budget']['native_validation_sampling']['temperature']=99
  b=c.proposed_role_stage(m.fixture(),'actual_hf200')
  self.assertEqual(b['generation_budget']['native_validation_sampling']['temperature'],0.0)
 def test_pure_module_has_no_io_process_or_production_entry(self):
  tree=ast.parse((D/'paired_eval_contract.py').read_bytes())
  imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
  self.assertEqual(set(imports),{'copy','hashlib','json','pathlib','re'})
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
  self.assertFalse(any(n.func.id in {'open','exec','eval','compile','__import__'} for n in calls))
  self.assertFalse(any(isinstance(n,ast.If) and '__name__' in ast.unparse(n.test) for n in tree.body))
s=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(m.ContractTests),unittest.defaultTestLoader.loadTestsFromTestCase(IndependentBoundaryTests)])
r=unittest.TextTestRunner(verbosity=2).run(s)
sys.exit(not r.wasSuccessful())
