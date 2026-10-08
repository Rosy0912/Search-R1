"""Recorder process-prefix counterexamples with isolated temp stubs, no target proof."""
import ast,pathlib,tempfile,unittest,types,json,hashlib,time,re,os,sys
from pathlib import Path
runner=Path(__file__).parent/'run-target-cpu-gate.py';SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
class Prefix(unittest.TestCase):
 def execute(self,root,prefix):
  tree=ast.parse(runner.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  class RootBinding(ast.NodeTransformer):
   def visit_Constant(self,n):
    if n.value=='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r4-20261004':return ast.copy_location(ast.Constant(str(root)),n)
    return n
  tree=ast.fix_missing_locations(RootBinding().visit(tree))
  class Processes:
   STDOUT=-2
   calls=[]
   def check_output(self,args,**kwargs):self.calls.append(args);return SOURCE+'\n'if args[-1]=='HEAD'else''
   def run(self,args,**kwargs):
    self.calls.append(args);assert args[0]==str(root/'env/bin/python');assert kwargs['cwd']==root/'repo';assert kwargs['env']['PYTHONPATH']==str(root/'repo')
    kwargs['stdout'].write('2 passed in 0.01s\n');return types.SimpleNamespace(returncode=0)
  process=Processes();prefixsys=types.SimpleNamespace(prefix=prefix,exit=lambda code:None)
  ns=dict(pathlib=pathlib,sys=prefixsys,subprocess=process,os=os,time=time,json=json,re=re,hashlib=hashlib)
  exec(compile(tree,'actual-recorder-ast','exec'),ns);return process
 def test_system_prefix_refused_before_any_marker_log_or_record_write(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'logs').mkdir()
   with self.assertRaisesRegex(AssertionError,'exact newROOT/env interpreter'):self.execute(root,'/usr')
   self.assertEqual(list((root/'logs').iterdir()),[])
 def test_correct_prefix_producer_records_actual_prefix_and_child_command(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'logs').mkdir();(root/'tools').mkdir();(root/'tools/cpu-tests.py').write_text('ISOLATED_TEST_STUB_NOT_TARGETPROOF')
   self.execute(root,str(root/'env'));v=json.loads((root/'logs/cpu-gate.json').read_text())
   self.assertEqual(v['sys_prefix'],str(root/'env'));self.assertEqual(v['passed'],2);self.assertEqual(v['exit_code'],0)
   self.assertEqual(v['log_sha256'],hashlib.sha256((root/'logs/cpu-gate.log').read_bytes()).hexdigest())
if __name__=='__main__':unittest.main()
