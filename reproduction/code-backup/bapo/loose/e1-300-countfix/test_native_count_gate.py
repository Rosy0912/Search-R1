"""Count-record structural refusal tests, not real NCCL execution."""
import ast,copy,hashlib,tempfile,unittest,math
from pathlib import Path
path=Path(__file__).parent/'tools/verify_hardware_receipt.py';tree=ast.parse(path.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='validate_native_counts');ns=dict(math=math,sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest());exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns);gate=ns['validate_native_counts'];SOURCE='f0e37bf68d61ec926ac225f5e1ee1f606cb09bf9'
class CountGate(unittest.TestCase):
 def fixture(self,root,rank):
  (root/'tools').mkdir(exist_ok=True);(root/'tools/native-count-fixture.py').write_bytes(b'synthetic-tool-identity')
  files={}
  for name in ('search_r1/baselines/distillation.py','search_r1/baselines/objectives.py'):
   p=root/'repo'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic-source');files[name]=ns['sha'](p)
  cases=[('one_empty_rank',[0,1,2,3,4,5,6,7],[3,4,5,6,7,8,9,10]),('global_empty_tokens',[0]*8,[3,4,5,6,7,8,9,10]),('unequal_rows_with_zero',[2,1,4,1,6,2,8,3],[0,2,3,4,5,6,7,8]),('global_empty_rows',[0]*8,[0]*8)]
  records=[dict(case=k,local_tokens=t[rank],local_rows=r[rank],global_tokens=sum(t),global_row_denominator=max(sum(r),1)/8,token_denominator=max(sum(t),1)/8,input_device='cpu',collective_device='cuda:'+str(rank),world=8)for k,t,r in cases]
  return dict(status='PASS_NATIVE_NCCL_CPU_INPUT_COUNTS_ONLY',source_commit=SOURCE,rank=rank,world=8,records=records,source_files_sha256=files,fixture_sha256=ns['sha'](root/'tools/native-count-fixture.py'))
 def test_exact_four_oracles_for_all_eight_ranks(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d)
   for rank in range(8):gate(self.fixture(root,rank),rank,SOURCE,root)
 def test_source_world_fixture_import_refusal(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);v=self.fixture(root,0)
   for key,value in [('source_commit','2175549691d6856e2b393b2834a32febe41ee56e'),('world',1),('rank',1),('fixture_sha256','0'*64),('status','PASS')]:
    bad=copy.deepcopy(v);bad[key]=value
    with self.assertRaises(AssertionError,msg=key):gate(bad,0,SOURCE,root)
   bad=copy.deepcopy(v);bad['source_files_sha256'].pop('search_r1/baselines/objectives.py')
   with self.assertRaises(AssertionError):gate(bad,0,SOURCE,root)
 def test_all_cpu_current_cuda_and_sum_denominator_boundaries(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);v=self.fixture(root,0)
   for key,value in [('input_device','cuda:0'),('collective_device','cpu'),('collective_device','cuda:1'),('global_tokens',999),('token_denominator',999),('global_row_denominator',999),('local_tokens',1),('local_rows',999),('world',1)]:
    bad=copy.deepcopy(v);bad['records'][0][key]=value
    with self.assertRaises(AssertionError,msg=key):gate(bad,0,SOURCE,root)
   for kind in ('missing','duplicate'):
    bad=copy.deepcopy(v)
    if kind=='missing':bad['records'].pop()
    else:bad['records'][1]=copy.deepcopy(bad['records'][0])
    with self.assertRaises(AssertionError):gate(bad,0,SOURCE,root)
 def test_native_count_invocation_before_actual_state_actor(self):
  text=(Path(__file__).parent/'tools/hardware.sh').read_text();a=text.index('$ROOT/tools/native-count-fixture.py')
  self.assertLess(a,text.index('$ROOT/tools/native-state-fixture.py'));self.assertLess(a,text.index('$ROOT/tools/native-actor-fixture.py'))
if __name__=='__main__':unittest.main()
