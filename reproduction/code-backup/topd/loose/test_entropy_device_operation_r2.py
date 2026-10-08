"""Operation admission contracts using actual failure bytes and rendered queue guard."""
import ast,json,pathlib,unittest
HERE=pathlib.Path(__file__).resolve().parent
OP=HERE/'run_entropy_device_v1_r1_final_once_r2.py'
tree=ast.parse(OP.read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets));remote=ast.literal_eval(assignment.value.func.value)
class Contracts(unittest.TestCase):
 def test_actual_failed3209_physical_schema(self):
  c=json.loads(pathlib.Path('outputs/remote-evidence/standard-r1-profile3209-small/owned-cleanup.json').read_text());self.assertTrue(c['cleanup']['actual_cgroup_empty']);self.assertEqual(c['own_gpu_records_remaining'],[])
  result=json.loads(pathlib.Path('outputs/remote-evidence/standard-r1-profile3209-small/train-result.json').read_text());self.assertEqual(result['status'],'FAILED');self.assertIs(result['cleanup_passed'],False)
 def test_exact_queue_running_rejected(self):
  tr=ast.parse(remote);node=next(n for n in ast.walk(tr) if isinstance(n,ast.Assert) and 'queue.splitlines' in ast.unparse(n));compiled=compile(ast.Module(body=[node],type_ignores=[]),'actual_queue','exec')
  for queue in ('3200|entropy-device-proof|RUNNING','3200|r1-32b-grpo-profile|RUNNING'):
   with self.assertRaises(AssertionError):exec(compiled,{'queue':queue})
  exec(compiled,{'queue':''})
 def test_explicit_submit_only(self):
  source=OP.read_text();self.assertIn("assert '--submit' in sys.argv",source);self.assertLess(source.index("if '--render-only'"),source.index("r=send(REMOTE"))
 def test_unknown_unconditional_before_submit(self):
  self.assertLess(remote.index("status='UNKNOWN_BEFORE_SUBMISSION'"),remote.index("['sbatch'"));self.assertIn("d['status']=='SUBMITTED'",remote)
 def test_bound_observer_not_guard_budget(self):
  self.assertIn('time.monotonic()+1350',remote);self.assertIn('timeout=min(5,max(.1,deadline-time.monotonic()))',remote);self.assertIn("assert not terminal.exists()",remote)
 def test_all_source_dependencies_and_whole_archives(self):
  for s in ('d9c495d8f2eb5bf9b1ff5680b6a8c0c2626d1e5375e92ae4971a78a9bf3dac70','cc9445265a3835e1623c0de9d95380717f0e44d4a303d45e3fe1f559431e1605','72c5fd163b9cbe6bfb502d098b7bc14e7dfa386e5c008ffd7a202eebd14b57b4','26ab1b6ade6f1d385c3381941a4346ba8c136c3b9b65405d4c7d32242cbafcad'):
   self.assertIn(s,remote if s[:4]!='26ab' else (pathlib.Path('work/entropy-device-proof-v1-r1-final/worker.py').read_text()))
if __name__=='__main__':unittest.main()
