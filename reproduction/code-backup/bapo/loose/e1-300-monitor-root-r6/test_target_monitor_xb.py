"""Local deployment-boundary stubs; never an actual target proof."""
import ast,base64,hashlib,json,os,pathlib,re,socket,subprocess,sys,tempfile,time,types,unittest
from pathlib import Path
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004';SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
FILE=Path(__file__).parent/'run-target-monitor-cpu-xb.py'
class XB(unittest.TestCase):
 def execute(self,root,prefix=None,source=SOURCE):
  tree=ast.parse(FILE.read_text());tree.body=[x for x in tree.body if not isinstance(x,(ast.Import,ast.ImportFrom))]
  class Binding(ast.NodeTransformer):
   def visit_Constant(self,n):return ast.copy_location(ast.Constant(str(root)),n)if n.value==ROOT else n
  tree=ast.fix_missing_locations(Binding().visit(tree))
  class Process:
   STDOUT=-2
   calls=[]
   def check_output(self,args,**kw):return source+'\n'if args[-1]=='HEAD'else''
   def run(self,args,**kw):
    self.calls.append(args);assert args[0]==str(root/'env/bin/python');assert kw['env']['CUDA_VISIBLE_DEVICES']=='';assert kw['cwd']==root/'repo'
    kw['stdout'].write('Ran 22 tests in 0.01s\n\nOK\n');return types.SimpleNamespace(returncode=0)
  process=Process();ns=dict(base64=base64,hashlib=hashlib,json=json,os=os,pathlib=pathlib,re=re,subprocess=process,sys=types.SimpleNamespace(prefix=prefix or str(root/'env'),executable=str(root/'env/bin/python')),time=time,socket=types.SimpleNamespace(gethostname=lambda:'h100-5'))
  try:exec(compile(tree,'actual-xb-runner-ast','exec'),ns)
  except SystemExit as e:self.assertEqual(e.code,0)
  return process
 def root(self,path):r=Path(path).resolve();(r/'repo').mkdir();return r
 def test_prefix_fails_before_any_write(self):
  with tempfile.TemporaryDirectory()as d:
   root=self.root(d)
   with self.assertRaisesRegex(AssertionError,'Recorder'):self.execute(root,'/usr')
   self.assertEqual({p.name for p in root.iterdir()},{'repo'})
 def test_wrong_source_before_write(self):
  with tempfile.TemporaryDirectory()as d:
   root=self.root(d)
   with self.assertRaises(AssertionError):self.execute(root,source='wrong')
   self.assertEqual({p.name for p in root.iterdir()},{'repo'})
 def test_exclusive_existing_evidence(self):
  with tempfile.TemporaryDirectory()as d:
   root=self.root(d);dest=root/'temporary-evidence/monitor22-actual-linux-20261004';dest.mkdir(parents=True);keep=dest/'original';keep.write_text('keep')
   with self.assertRaisesRegex(AssertionError,'Existing'):self.execute(root)
   self.assertEqual(keep.read_text(),'keep');self.assertEqual(len(list(dest.iterdir())),1)
 def test_symlink_evidence_parent_rejected(self):
  with tempfile.TemporaryDirectory()as d,tempfile.TemporaryDirectory()as external:
   root=self.root(d);(root/'temporary-evidence').symlink_to(external)
   with self.assertRaises(AssertionError):self.execute(root)
   self.assertEqual(list(Path(external).iterdir()),[])
 def test_exact_bytes_record_and_only_new_directory(self):
  with tempfile.TemporaryDirectory()as d:
   root=self.root(d);process=self.execute(root);self.assertEqual(len(process.calls),1);dest=root/'temporary-evidence/monitor22-actual-linux-20261004';record=json.loads((dest/'monitor-cpu-gate.json').read_text());self.assertEqual(record['actual_tests'],22)
   for p,sha in record['files_sha256'].items():self.assertEqual(hashlib.sha256((dest/p).read_bytes()).hexdigest(),sha)
   self.assertEqual({p.name for p in root.iterdir()},{'repo','temporary-evidence'});self.assertFalse((root/'tools').exists())
if __name__=='__main__':unittest.main()
