"""Mocked runtime/record contract tests only; no Linux/GPU acceptance."""
import ast,copy,hashlib,json,pathlib,tempfile,types,unittest
from unittest.mock import patch
BASE=pathlib.Path(__file__).parent
P=BASE/'deployment-tools/verify_launch.py';tree=ast.parse(P.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_runtime_executable')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Test(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name).resolve();(self.root/'logs').mkdir();(self.root/'tools').mkdir();(self.root/'env/bin').mkdir(parents=True);self.binary=self.root/'real-python';self.binary.write_bytes(b'mock binary, not executed');(self.root/'env/bin/python').symlink_to(self.binary);self.capture=self.root/'tools/capture-runtime-executable.py';self.capture.write_text('bound capture tool mock');self.source='4'*40;st=self.binary.stat();self.record=dict(source_commit=self.source,root=str(self.root),sys_prefix=str(self.root/'env'),executable=dict(resolved_path=str(self.binary),device=st.st_dev,inode=st.st_ino),executable_sha256=sha(self.binary),host='h100-5',scope='ACTUAL_LINUX_INTERPRETER_IDENTITY_NOT_GPU_OR_LAUNCH_ACCEPTANCE');self.path=self.root/'logs/runtime-executable.json';self.manifest=dict(source_commit=self.source,files={'capture-runtime-executable.py':sha(self.capture)})
  self.fake_sys=types.SimpleNamespace(platform='linux',prefix=str(self.root/'env'),executable=str(self.root/'env/bin/python'))
  self.fake_pathlib=types.SimpleNamespace(Path=lambda p:self.binary if str(p)=='/proc/self/exe'else pathlib.Path(p))
  self.ns=dict(pathlib=self.fake_pathlib,json=json,hashlib=hashlib,sys=self.fake_sys,sha=sha);exec(compile(ast.Module(body=[fn],type_ignores=[]),'validator','exec'),self.ns);self.save()
 def tearDown(self):self.tmp.cleanup()
 def save(self):
  self.path.write_text(json.dumps(self.record,sort_keys=True,indent=2)+'\n');self.approval=dict(runtime_executable=dict(path=str(self.path),bytes=self.path.stat().st_size,sha256=sha(self.path)))
 def check(self):
  with patch('os.uname',return_value=types.SimpleNamespace(nodename='h100-5')):return self.ns['validate_runtime_executable'](self.root,self.source,self.approval,self.manifest)
 def test_valid_mock_record_and_venv_symlink(self):self.assertEqual(self.check()['binary_sha256'],sha(self.binary))
 def test_binding_path_bytes_sha_alias_and_semantic_mismatch(self):
  for key,value in [('path','/outside/raw.json'),('bytes',True),('bytes',1),('sha256','0'*64)]:
   with self.subTest(key=key):
    old=copy.deepcopy(self.approval);self.approval['runtime_executable'][key]=value
    with self.assertRaises((AssertionError,FileNotFoundError)):self.check()
    self.approval=old
 def test_rebound_record_identity_fields_rejected(self):
  for key,value in [('source_commit','old'),('root','/old'),('sys_prefix','/old/env'),('host','other'),('scope','PASS_GPU'),('executable_sha256','0'*64)]:
   with self.subTest(key=key):
    old=copy.deepcopy(self.record);self.record[key]=value;self.save()
    with self.assertRaises(AssertionError):self.check()
    self.record=old
 def test_executable_inode_device_path_alias_rejected(self):
  for key,value in [('inode',True),('inode',1),('device',True),('device',-1),('resolved_path','relative'),('resolved_path','/other/python')]:
   with self.subTest(key=key):
    old=copy.deepcopy(self.record);self.record['executable'][key]=value;self.save()
    with self.assertRaises(AssertionError):self.check()
    self.record=old
 def test_current_runtime_platform_prefix_executable_mismatch(self):
  for key,value in [('platform','darwin'),('prefix','/foreign'),('executable','/missing/python')]:
   old=getattr(self.fake_sys,key);setattr(self.fake_sys,key,value)
   with self.subTest(key=key),self.assertRaises((AssertionError,FileNotFoundError)):self.check()
   setattr(self.fake_sys,key,old)
 def test_proc_executable_mismatch(self):
  foreign=self.root/'foreign-binary';foreign.write_text('foreign');self.fake_pathlib.Path=lambda p:foreign if str(p)=='/proc/self/exe'else pathlib.Path(p)
  with self.assertRaises(AssertionError):self.check()
 def test_record_symlink_and_duplicate_keys_refused(self):
  raw=self.path.read_bytes();other=self.root/'other.json';other.write_bytes(raw);self.path.unlink();self.path.symlink_to(other)
  with self.assertRaises(AssertionError):self.check()
  self.path.unlink();self.path.write_bytes(raw[:-2]+b',"host":"h100-5"}\n');self.approval['runtime_executable'].update(bytes=self.path.stat().st_size,sha256=sha(self.path))
  with self.assertRaisesRegex(AssertionError,'Duplicate'):self.check()
 def test_tool_manifest_hash_and_binary_replacement_refused(self):
  self.manifest['files']['capture-runtime-executable.py']='0'*64
  with self.assertRaises(AssertionError):self.check()
  self.manifest['files']['capture-runtime-executable.py']=sha(self.capture);self.binary.write_bytes(b'changed inplace same inode')
  with self.assertRaises(AssertionError):self.check()
 def test_old_launch_gate_byte_exact_preserved(self):
  text=P.read_text();a=text.index('def validate_runtime_executable');b=text.index("phase=sys.argv[1]",a);without=(text[:a]+text[b:]).replace('\nvalidate_runtime_executable(root,source,r,m)','').replace('cpu_log=validate_cpu_log_binding(root,source,c,r)\ntext=cpu_log.read_text()',"assert sha(root/'logs/cpu-gate.log')==c['log_sha256']\ntext=(root/'logs/cpu-gate.log').read_text()");self.assertEqual(without,(BASE/'verify-launch-runtime-before.py.txt').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
