"""Durable actual-exit binding contract tests; no remote or GPU action."""
import ast,copy,hashlib,json,pathlib,tempfile,unittest
BASE=pathlib.Path(__file__).parent;P=BASE/'deployment-tools/verify_launch.py';tree=ast.parse(P.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_cpu_log_binding');ns=dict(pathlib=pathlib,json=json,hashlib=hashlib);exec(compile(ast.Module(body=[fn],type_ignores=[]),'durable','exec'),ns);validate=ns['validate_cpu_log_binding'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Test(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name).resolve();(self.root/'logs').mkdir();self.source='4'*40;self.log=self.root/'logs/cpu-gate-durable.log';self.log.write_text('second run true log');self.first=self.root/'logs/cpu-gate.log';self.first.write_text('retained first observation');self.exit=self.root/'logs/cpu-gate-durable-exit.json';self.actual=dict(source_commit=self.source,scope='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE',exit_code=0,training_submitted=False,log_sha256=sha(self.log));self.rebind()
 def tearDown(self):self.tmp.cleanup()
 def rebind(self):
  self.exit.write_text(json.dumps(self.actual));self.approval={key:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))for key,p in [('cpu_gate_log',self.log),('cpu_gate_exit_record',self.exit)]};self.record=dict(exit_code=0,passed=395,failed=0,skipped=0,log_sha256=sha(self.log),exit_record_sha256=sha(self.exit))
 def check(self):return validate(self.root,self.source,self.record,self.approval)
 def test_valid_durable_and_first_not_modified(self):self.assertEqual(self.check(),self.log);self.assertEqual(self.first.read_text(),'retained first observation')
 def test_wrong_exit_source_scope_failure_alias_log_after_rebind(self):
  for k,v in [('exit_code',1),('exit_code',False),('source_commit','old'),('scope','PASS_GPU'),('training_submitted',True),('log_sha256',sha(self.first))]:
   with self.subTest(key=k):
    old=copy.deepcopy(self.actual);self.actual[k]=v;self.rebind()
    with self.assertRaises(AssertionError):self.check()
    self.actual=old
 def test_wrong_primary_log_and_bytes_record_hash(self):
  for key,value in [('path',str(self.first)),('bytes',True),('bytes',1),('sha256','0'*64)]:
   self.rebind();self.approval['cpu_gate_log'][key]=value
   with self.subTest(key=key),self.assertRaises(AssertionError):self.check()
  self.rebind();self.record['exit_record_sha256']='0'*64
  with self.assertRaises(AssertionError):self.check()
 def test_record_symlink_duplicate_exit_and_bool_counts(self):
  target=self.root/'other-exit.json';target.write_text(self.exit.read_text());self.exit.unlink();self.exit.symlink_to(target)
  with self.assertRaises(AssertionError):self.check()
  self.exit.unlink();self.rebind();raw=self.exit.read_text();self.exit.write_text(raw[:-1]+',"exit_code":0}');self.approval['cpu_gate_exit_record'].update(bytes=self.exit.stat().st_size,sha256=sha(self.exit));self.record['exit_record_sha256']=sha(self.exit)
  with self.assertRaisesRegex(AssertionError,'Duplicate'):self.check()
  self.rebind();self.record['skipped']=False
  with self.assertRaises(AssertionError):self.check()
 def test_original_gate_preserved_except_explicit_primary_log_route(self):
  s=P.read_text();a=s.index('def validate_cpu_log_binding');b=s.index('phase=sys.argv[1]',a);s=(s[:a]+s[b:]).replace('cpu_log=validate_cpu_log_binding(root,source,c,r)\ntext=cpu_log.read_text()',"assert sha(root/'logs/cpu-gate.log')==c['log_sha256']\ntext=(root/'logs/cpu-gate.log').read_text()");self.assertEqual(s,(BASE/'verify-launch-durable-before.py.txt').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
