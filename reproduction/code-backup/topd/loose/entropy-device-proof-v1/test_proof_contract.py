"""CPU-only source contracts; not actual CUDA/model/memory proof."""
import ast,importlib.util,pathlib,sys,unittest
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
 def test_zero_has_no_extra_floor(self):
  ref=torch.zeros(1,dtype=torch.bfloat16);up=torch.nextafter(ref,torch.full_like(ref,float('inf')));self.assertTrue(worker.adjacent_difference(up,ref)['all_elements_passed']);self.assertFalse(worker.adjacent_difference(up*2,ref)['all_elements_passed'])
 def test_fp32_gradient_full_element_drift(self):
  ref=torch.ones(1024*1024+1,dtype=torch.float32);other=ref.clone();other[-1]=1.1;self.assertFalse(worker.compiled_difference(other,ref,True)['all_elements_passed'])
 def test_gpu_production_and_dynamic_repeated_cases(self):
  tree=ast.parse((HERE/'worker.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cases' for t in n.targets));cases=ast.literal_eval(node.value)
  for dtype in ('float32','bfloat16'):
   self.assertEqual(cases.count((6144,152064,dtype)),2);self.assertIn((6143,152064,dtype),cases)
 def test_parent_online_leases_excludes_aggregate(self):
  tree=ast.parse((HERE/'parent.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='leases');body=ast.unparse(fn);self.assertIn('lease_ownership(row)',body);self.assertNotIn('validate_inherited_gpu_leases(',body)
  source=ast.unparse(tree);self.assertEqual(source.count('validate_inherited_gpu_leases(expected_count=8)'),1)
 def test_parent_cpu_four_before_cuda_child(self):
  s=(HERE/'parent.py').read_text();self.assertLess(s.index("cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES=''"),s.index('child=owned.spawn'));self.assertIn("cpu['tests']==4",s)
 def test_monitor_before_context_wait_and_cleanup_finally(self):
  s=(HERE/'parent.py').read_text();self.assertLess(s.index('thread.start()'),s.index("wait_file(folder/'child-context-ready.json'"));self.assertIn('owned.cleanup()',s);self.assertIn('guardian_join_error',s)
 def test_actual_dtype_selects_comparison(self):
  ref=torch.ones(4,dtype=torch.float32);self.assertEqual(worker.compiled_difference(ref,ref,False)['atol'],1e-4)
 def test_no_cuda_initialized_by_contracts(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
