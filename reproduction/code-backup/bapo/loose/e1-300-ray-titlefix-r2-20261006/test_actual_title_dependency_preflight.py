import ast,base64,copy,importlib.util,pathlib,unittest
B=pathlib.Path(__file__).parent;P=B/'deployment-tools/actual-title-dependency-preflight.py';spec=importlib.util.spec_from_file_location('preflight',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def snap(self,title='ray::IDLE'):
  raw=(title+'\0\0').encode();return dict(pid=123,start_ticks=456,uid=1000,executable=dict(resolved_path='/usr/bin/python3.10',device=1,inode=2,sha256='a'*64),cmdline_base64=base64.b64encode(raw).decode(),command=[title,'',''])
 def test_real_nul_shape_accepts(self):m.validate_snapshot(self.snap(),self.snap('ray::WorkerDict'),'ray::WorkerDict')
 def test_same_pid_epoch_executable_and_uid_required(self):
  for key,value in [('pid',124),('start_ticks',457),('uid',1001),('executable',dict(resolved_path='/bin/sleep',device=1,inode=2,sha256='a'*64))]:
   a=self.snap();a[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.validate_snapshot(self.snap(),a,'ray::IDLE')
 def test_inode_and_binary_change_rejected(self):
  for key,value in [('inode',3),('device',3),('sha256','b'*64)]:
   a=self.snap();a['executable'][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.validate_snapshot(self.snap(),a,'ray::IDLE')
 def test_nonempty_argv_tail_rejected(self):
  a=self.snap();a['command']=['ray::IDLE','extra'];a['cmdline_base64']=base64.b64encode(b'ray::IDLE\0extra').decode()
  with self.assertRaises(ValueError):m.validate_snapshot(self.snap(),a,'ray::IDLE')
 def test_raw_ack_mismatch_rejected(self):
  a=self.snap();a['cmdline_base64']=base64.b64encode(b'ray::WorkerDict\0').decode()
  with self.assertRaises(ValueError):m.validate_snapshot(self.snap(),a,'ray::IDLE')
 def test_unexpected_title_rejected(self):
  with self.assertRaises(ValueError):m.validate_snapshot(self.snap(),self.snap(),'ray::WorkerDict')
 def test_exact_minimal_fixture_delta(self):
  old=pathlib.Path('work/e1-300-ray-titlefix-20261006/deployment-tools/identity-linux-fixture.py').read_text();new=(B/'deployment-tools/identity-linux-fixture.py').read_text();newroot=m.ROOT;oldroot='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006';expected=old.replace(oldroot,newroot).replace(' import setproctitle\n',' import ray\n import setproctitle\n').replace("code='import setproctitle,time;","code='import ray,setproctitle,time;");self.assertEqual(new,expected)
 def test_no_ray_init_or_signal_or_install_calls(self):
  tree=ast.parse(P.read_text());bad={'kill','terminate','pidfd_send_signal','setenv','system'}
  self.assertFalse(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in bad for n in ast.walk(tree)))
  self.assertNotIn('ray.init(',P.read_text());self.assertNotIn('pip install',P.read_text())
if __name__=='__main__':unittest.main(verbosity=2)
