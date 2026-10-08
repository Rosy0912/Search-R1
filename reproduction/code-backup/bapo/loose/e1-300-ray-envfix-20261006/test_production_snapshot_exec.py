"""Execute actual capture against synthetic /proc files, without processes/signals."""
import importlib.util,pathlib,tempfile,types,unittest
from unittest.mock import patch
D=pathlib.Path(__file__).parent/'deployment-tools';spec=importlib.util.spec_from_file_location('sampler_capture',D/'actual-title-dependency-preflight.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Capture(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.base=pathlib.Path(self.tmp.name).resolve();self.pid=57;self.proc=self.base/str(self.pid);self.proc.mkdir();self.a=self.base/'python-a';self.a.write_bytes(b'python-a');self.b=self.base/'python-b';self.b.write_bytes(b'python-b');(self.proc/'exe').symlink_to(self.a);(self.proc/'stat').write_text('57 (worker) '+' '.join(['S']+['0']*18+['999']+['0']*5));(self.proc/'cmdline').write_bytes(b'ray::IDLE\0\0');self.Path=pathlib.Path;self.stat=pathlib.Path.stat
 def tearDown(self):self.tmp.cleanup()
 def factory(self,arg,*args,**kwargs):return self.base if arg=='/proc' else self.Path(arg,*args,**kwargs)
 def capture(self,marker=lambda pid:{}):
  with patch.object(m,'pathlib',types.SimpleNamespace(Path=self.factory)),patch.object(m,'process_markers',side_effect=marker):return m.production_snapshot(self.pid,{})
 def test_stable_capture_succeeds(self):
  s=self.capture();self.assertEqual(s['start_ticks'],999);self.assertEqual(s['executable']['resolved_path'],str(self.a));self.assertEqual(s['uid'],self.proc.stat().st_uid)
 def test_same_pid_same_epoch_different_exe_reject(self):
  def change(pid):(self.proc/'exe').unlink();(self.proc/'exe').symlink_to(self.b);return {}
  with self.assertRaisesRegex(ValueError,'physical executable'):self.capture(change)
 def test_same_path_replaced_inode_reject(self):
  def change(pid):self.a.unlink();self.a.write_bytes(b'replaced-inode');return {}
  # Ensure an inode cannot be immediately recycled by keeping an open descriptor.
  with self.a.open('rb')as held:
   with self.assertRaisesRegex(ValueError,'physical executable'):self.capture(change)
 def test_same_epoch_changed_uid_reject(self):
  state={'changed':False}
  def stat(path,*args,**kwargs):
   if path==self.proc:return types.SimpleNamespace(st_uid=2000 if state['changed']else 1000)
   return self.stat(path,*args,**kwargs)
  def change(pid):state['changed']=True;return {}
  with patch.object(self.Path,'stat',stat):
   with self.assertRaisesRegex(ValueError,'UID'):self.capture(change)
 def test_epoch_changed_reject(self):
  def change(pid):(self.proc/'stat').write_text('57 (worker) '+' '.join(['S']+['0']*18+['1000']+['0']*5));return {}
  with self.assertRaisesRegex(ValueError,'epoch'):self.capture(change)
if __name__=='__main__':unittest.main()
