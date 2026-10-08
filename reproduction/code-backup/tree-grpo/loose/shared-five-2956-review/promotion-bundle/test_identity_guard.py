import ast,pathlib,types,unittest
SOURCE=pathlib.Path(__file__).resolve().parents[1]/'resource_guard_identity.py'
def load():
 tree=ast.parse(SOURCE.read_text()); funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('has_job','process_identity','fd_pid','own')]
 ns=dict(pathlib=pathlib,os=types.SimpleNamespace(pidfd_open=lambda pid:pid+100,close=lambda fd:None),job='123',known_own={},zombie_proofs={})
 exec(compile(ast.Module(body=funcs,type_ignores=[]),str(SOURCE),'exec'),ns);return ns
class Checks(unittest.TestCase):
 def setUp(self):
  self.n=load();self.state='R';self.start=42;self.member=True;self.fdpid=7
  self.n['process_identity']=lambda pid:(self.state,self.start)
  self.n['fd_pid']=lambda fd:self.fdpid
  self.n['pathlib']=types.SimpleNamespace(Path=lambda *args:types.SimpleNamespace(read_text=lambda:'0::/job_123/step_0' if self.member else '0::/job_456/step_0'))
 def test_exact_segment(self):
  self.assertFalse(self.n['has_job']('0::/job_1234/step_0'));self.assertTrue(self.n['has_job']('0::/job_123/step_0'))
 def register(self):self.assertTrue(self.n['own'](7))
 def test_original_z(self):
  self.register();self.member=False;self.state='Z';self.assertTrue(self.n['own'](7));self.assertEqual(self.n['zombie_proofs'][7]['start_ticks'],42)
 def test_live_outside(self):
  self.register();self.member=False;self.assertFalse(self.n['own'](7))
 def test_reused_z(self):
  self.register();self.member=False;self.state='Z';self.start=43;self.assertFalse(self.n['own'](7))
 def test_reaped_fd(self):
  self.register();self.member=False;self.state='Z';self.fdpid=-1;self.assertFalse(self.n['own'](7))
 def test_unknown_z(self):
  self.member=False;self.state='Z';self.assertFalse(self.n['own'](7))
 def test_missing_clears(self):
  self.register();self.member=False;self.state='Z';self.n['own'](7)
  def absent(pid):raise FileNotFoundError()
  self.n['process_identity']=absent;self.assertIsNone(self.n['own'](7));self.assertFalse(self.n['zombie_proofs'])
 def test_unreadable_refuses(self):
  def bad(pid):raise PermissionError('unknown')
  self.n['process_identity']=bad
  with self.assertRaises(PermissionError):self.n['own'](7)
 def test_registration_race(self):
  calls=iter([('R',42),('R',43)]);self.n['process_identity']=lambda pid:next(calls)
  self.assertIsNone(self.n['own'](7));self.assertFalse(self.n['known_own'])
 def test_stop_even_append_failure(self):
  tree=ast.parse(SOURCE.read_text());handler=tree.body[-1].handlers[0];sent=[]
  def append(_):raise OSError('disk failure')
  ns=dict(append=append,time=types.SimpleNamespace(time=lambda:0),job='123',traceback=types.SimpleNamespace(format_exc=lambda:'failed'),subprocess=types.SimpleNamespace(run=lambda argv,check:sent.append(argv)))
  wrapper=ast.Try(body=[ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError',ctx=ast.Load()),args=[],keywords=[]),cause=None)],handlers=[handler],orelse=[],finalbody=[])
  with self.assertRaises(OSError):exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(SOURCE),'exec'),ns)
  self.assertEqual(sent,[['scancel','123']])
 def test_fresh_sm_classification(self):
  tree=ast.parse(SOURCE.read_text());loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='sample']
  self.assertEqual(len(loops),1);self.assertIsInstance(loops[0].body[0],ast.Assign);self.assertEqual(ast.unparse(loops[0].body[0].value),'own(sample.pid)')
if __name__=='__main__':unittest.main(verbosity=2)
