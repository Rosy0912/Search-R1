"""CPU-only source contracts; not actual CUDA/model/memory proof."""
import tempfile,unittest.mock,enum,json,ast,importlib.util,pathlib,sys,unittest
import torch
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('entropy_device_worker_contract',HERE/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
class Contracts(unittest.TestCase):
 def test_bf16_exact_and_one_neighbor(self):
  ref=torch.tensor([0.,1.,-1.,12.],dtype=torch.bfloat16);up=torch.nextafter(ref,torch.full_like(ref,float('inf')))
  self.assertTrue(worker.adjacent_difference(ref,ref)['all_elements_passed']);self.assertTrue(worker.adjacent_difference(up,ref)['all_elements_passed'])
 def test_bf16_two_neighbor_rejected(self):
  ref=torch.tensor([1.],dtype=torch.bfloat16);up=torch.nextafter(ref,torch.full_like(ref,float('inf')));twice=torch.nextafter(up,torch.full_like(up,float('inf')))
  r=worker.adjacent_difference(twice,ref);self.assertFalse(r['all_elements_passed']);self.assertEqual(r['nonadjacent_elements'],1)
 def test_power_of_two_two_lower_neighbors_rejected(self):
  ref=torch.tensor([1.],dtype=torch.bfloat16);down=torch.nextafter(ref,torch.full_like(ref,float('-inf')));twice=torch.nextafter(down,torch.full_like(down,float('-inf')));self.assertFalse(worker.adjacent_difference(twice,ref)['all_elements_passed'])
 def test_zero_has_no_extra_floor(self):
  ref=torch.zeros(1,dtype=torch.bfloat16);up=torch.nextafter(ref,torch.full_like(ref,float('inf')));self.assertTrue(worker.adjacent_difference(up,ref)['all_elements_passed']);self.assertFalse(worker.adjacent_difference(up*2,ref)['all_elements_passed'])
 def test_fp32_gradient_full_element_drift(self):
  ref=torch.ones(1024*1024+1,dtype=torch.float32);other=ref.clone();other[-1]=1.1;self.assertFalse(worker.compiled_difference(other,ref,True)['all_elements_passed'])
 def test_gpu_production_and_dynamic_repeated_cases(self):
  tree=ast.parse((HERE/'worker.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cases' for t in n.targets));cases=ast.literal_eval(node.value)
  for dtype in ('float32','bfloat16'):
   self.assertEqual(cases.count((6144,152064,dtype)),1);self.assertIn((6143,152064,dtype),cases);self.assertEqual(len(cases),4)
 def test_parent_online_leases_excludes_aggregate(self):
  tree=ast.parse((HERE/'parent.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='leases');body=ast.unparse(fn);self.assertIn('lease_ownership(row)',body);self.assertNotIn('validate_inherited_gpu_leases(',body)
  source=ast.unparse(tree);self.assertEqual(source.count('validate_inherited_gpu_leases(expected_count=8)'),1)
 def test_parent_cpu_four_before_cuda_child(self):
  s=(HERE/'parent.py').read_text();self.assertLess(s.index("cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES=''"),s.index('child=owned.spawn'));self.assertIn("cpu['tests']==4",s)
 def test_monitor_before_context_wait_and_cleanup_finally(self):
  s=(HERE/'parent.py').read_text();self.assertLess(s.index('thread.start()'),s.index("wait_file(folder/'child-context-ready.json'"));self.assertIn('owned.cleanup()',s);self.assertIn('guardian_join_error',s)
 def test_actual_dtype_selects_comparison(self):
  ref=torch.ones(4,dtype=torch.float32);self.assertEqual(worker.compiled_difference(ref,ref,False)['atol'],1e-4)
 def test_actual_metadata_enum_set_projection(self):
  class Hint(enum.Enum):REDUCTION=1
  raw={'dynamic_scale_rblock':False,'reduction_hint':Hint.REDUCTION,'autotune_hints':{Hint.REDUCTION}}
  projection=worker.metadata_projection(raw);encoded=json.dumps(projection);decoded=json.loads(encoded)
  self.assertIs(decoded['dynamic_scale_rblock'],False);self.assertEqual(decoded['reduction_hint']['__actual_repr__'],repr(Hint.REDUCTION));self.assertEqual(decoded['autotune_hints']['__actual_type__'],'set');self.assertIs(raw['reduction_hint'],Hint.REDUCTION)
 def test_positive_negative_power_boundary_both_directions(self):
  for value in (1.,-1.):
   for direction in (float('inf'),float('-inf')):
    ref=torch.tensor([value],dtype=torch.bfloat16);one=torch.nextafter(ref,torch.full_like(ref,direction));two=torch.nextafter(one,torch.full_like(one,direction));self.assertTrue(worker.adjacent_difference(one,ref)['all_elements_passed']);self.assertFalse(worker.adjacent_difference(two,ref)['all_elements_passed'])
 def test_cold_real_helper_reset_before_construct_and_unique_cache(self):
  calls=[]
  class Fn:
   def get_compiler_config(self):return worker.OPTIONS
  def source():calls.append('construct');return None,Fn(),Fn(),{}
  with tempfile.TemporaryDirectory() as directory,unittest.mock.patch.object(torch._dynamo,'reset',side_effect=lambda:calls.append('reset')),unittest.mock.patch.object(worker,'source_functions',side_effect=source):
   events=[];worker.cold_compiled('candidate',pathlib.Path(directory),0,events);self.assertEqual(calls,['reset','construct']);self.assertEqual(events[0]['actual_options'],worker.OPTIONS)
   with self.assertRaises(FileExistsError):worker.cold_compiled('candidate',pathlib.Path(directory),0,events)
 def test_cold_wrong_actual_config_rejects(self):
  class Fn:
   def get_compiler_config(self):return dict(worker.OPTIONS,dynamic_scale_rblock=True)
  with tempfile.TemporaryDirectory() as directory,unittest.mock.patch.object(torch._dynamo,'reset'),unittest.mock.patch.object(worker,'source_functions',return_value=(None,Fn(),Fn(),{})):
   with self.assertRaisesRegex(RuntimeError,'Actual cold candidate'):worker.cold_compiled('candidate',pathlib.Path(directory),0,[])
 def test_frontcheck_missing_actual_dispatch_and_metadata_fail(self):
  with self.assertRaisesRegex(RuntimeError,'actual backend missing'):worker.candidate_frontcheck([])
 def test_no_cuda_initialized_by_contracts(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
