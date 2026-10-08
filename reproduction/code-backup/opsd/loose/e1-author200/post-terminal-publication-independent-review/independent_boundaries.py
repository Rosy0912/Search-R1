"""Independent local fixtures. No remote or production publication."""
import base64,contextlib,hashlib,io,json,os,pathlib,subprocess,sys,tempfile,types,unittest
from unittest import mock
BASE=pathlib.Path(__file__).resolve().parents[1]
PUBLISHER=BASE/'post-terminal-publication-candidate/publish_post_terminal_integrity_once.py'
PIN='d57b3838d5518337eaae2aa2230d8d961e41b6940e18e7396b33d19c769a56f0'
raw=PUBLISHER.read_bytes()
assert hashlib.sha256(raw).hexdigest()==PIN
p=types.ModuleType('independently_pinned_publication');p.__file__=str(PUBLISHER)
exec(compile(raw,p.__file__,'exec',dont_inherit=True),p.__dict__)
class IndependentBoundaries(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name).resolve()
  self.packet=p.prepare_packet();self.mraw,self.members=p.validate_packet(self.packet)
 def tearDown(self):
  for d in self.root.rglob('*'):
   if d.is_dir() and not d.is_symlink():d.chmod(0o700)
  self.tmp.cleanup()
 def test_dead_allocation_rejected_before_new_directory(self):
  with mock.patch.object(p.os,'getuid',return_value=0),mock.patch.object(p.socket,'gethostname',return_value='h100-10'),mock.patch.object(p,'compiled_module',return_value=object()),mock.patch.object(p,'source_anchor',side_effect=ValueError('original allocation is gone')),mock.patch.object(p,'publish_files',side_effect=AssertionError('must never publish')):
   with self.assertRaisesRegex(ValueError,'gone'):p.publish_packet(self.packet)
 def test_source_change_after_publication_is_not_success(self):
  before={'source_files':{'x.py':{'sha256':'a'*64,'bytes':1}}};after={'source_files':{'x.py':{'sha256':'b'*64,'bytes':1}}}
  trace=[]
  def fake_publish(*a):trace.append('publication-fixture-only');return []
  with mock.patch.object(p.os,'getuid',return_value=0),mock.patch.object(p.socket,'gethostname',return_value='h100-10'),mock.patch.object(p,'compiled_module',return_value=object()),mock.patch.object(p,'source_anchor',side_effect=[before,after]),mock.patch.object(p,'publish_files',side_effect=fake_publish):
   with self.assertRaisesRegex(ValueError,'Source anchors changed'):p.publish_packet(self.packet)
  self.assertEqual(trace,['publication-fixture-only'])
 def test_foreign_uid_refused_before_helper(self):
  with mock.patch.object(p.os,'getuid',return_value=5),mock.patch.object(p.socket,'gethostname',return_value='h100-10'),mock.patch.object(p,'compiled_module',side_effect=AssertionError('must not load')):
   with self.assertRaisesRegex(ValueError,'Only own live'):p.publish_packet(self.packet)
 def run_main_fixture(self,rc,stdout):
  calls=[]
  def remote(*a,**k):calls.append(a);return types.SimpleNamespace(stdout=stdout,stderr=b'actual fixture stderr',returncode=rc)
  args=[p.__file__,'--deploy-reviewed','--publisher-sha256',PIN]
  with mock.patch.object(p,'W',self.root),mock.patch.object(p,'prepare_packet',return_value=self.packet),mock.patch.object(p,'compiled_module',return_value=types.SimpleNamespace(remote=remote)),mock.patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()) as capture:
   with self.assertRaises((ValueError,KeyError)):p.main()
  self.assertEqual(capture.getvalue(),'');self.assertEqual(len(calls),1)
  output=self.root/'post-terminal-integrity-deployment-b8d8ff319cec-originals'
  self.assertEqual((output/'publish-stdout.original.json').read_bytes(),stdout)
  self.assertEqual((output/'publish-stderr.original').read_bytes(),b'actual fixture stderr')
  self.assertEqual(json.loads((output/'publish-exit-original.json').read_bytes())['actual_returncode'],rc)
 def test_exit_zero_malformed_payload_has_originals_but_no_success(self):self.run_main_fixture(0,b'{}')
 def test_boolean_exit_does_not_impersonate_zero(self):self.run_main_fixture(False,b'{}')
 def test_nonzero_return_object_retains_exact_exit(self):self.run_main_fixture(-15,b'{}')
 def test_missing_explicit_deploy_flag_refuses_before_packet(self):
  with mock.patch.object(sys,'argv',[p.__file__,'--publisher-sha256',PIN]),mock.patch.object(p,'prepare_packet',side_effect=AssertionError('must not prepare')):
   with self.assertRaisesRegex(ValueError,'explicitly'):p.main()
 def test_duplicate_or_nonfinite_json_rejects(self):
  for body in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'):
   with self.subTest(body=body),self.assertRaises(ValueError):p.unique(body)
if __name__=='__main__':unittest.main(verbosity=2)
