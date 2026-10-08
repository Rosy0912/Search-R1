"""Actual rendered admission byte contracts, no remote/GPU execution."""
import ast,pathlib,json,hashlib,unittest
RENDER=pathlib.Path('outputs/streamed-adamw-device-v2-operation-render.py')
class Contract(unittest.TestCase):
 def test_actual_cleanup_real_fields(self):
  raw=json.loads(pathlib.Path('outputs/remote-evidence/standard-r1-profile3197-small/owned-cleanup.json').read_text())
  tree=ast.parse(RENDER.read_text()); assertion=next(n for n in ast.walk(tree) if isinstance(n,ast.Assert) and "c['own_gpu_records_remaining']" in ast.unparse(n))
  exec(compile(ast.Module(body=[assertion],type_ignores=[]),'real-admission','exec'),{'c':raw})
  raw=dict(raw,own_gpu_records_remaining=[{'pid':1}])
  with self.assertRaises(AssertionError):exec(compile(ast.Module(body=[assertion],type_ignores=[]),'real-admission','exec'),{'c':raw})
 def test_unknown_record_never_resubmits(self):
  tree=ast.parse(RENDER.read_text());outer=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='record.exists()')
  assertion=outer.body[1]
  for status in ['UNKNOWN_BEFORE_SUBMISSION','FAILED']:
   with self.assertRaises(AssertionError):exec(compile(ast.Module(body=[assertion],type_ignores=[]),'unknown','exec'),{'d':{'status':status,'archive_sha256':'same'},'F':{'archive_sha256':'same'}})
 def test_immutable_terminal_short_circuit_before_scontrol(self):
  s=RENDER.read_text();self.assertLess(s.index("if d.get('terminal')"),s.index("['scontrol','show','job'"));self.assertIn("assert not terminal.exists()",s)
 def test_all_full_sources_and_actual8_entry(self):
  s=RENDER.read_text();self.assertIn('len(members)==339',s);self.assertIn('cc9445265a3835e',s);self.assertIn('4861cb7cd20fe9',s)
  entry=pathlib.Path('work/streamed-adamw-device-parent-v2/probe.sbatch').read_text();self.assertIn('--gres=gpu:8',entry);self.assertIn('-m search_r1.baselines.preflight --',entry)
if __name__=='__main__':unittest.main()
