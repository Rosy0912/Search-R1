"""CPU-only isolated collector routing fixtures; no remote execution or GPU acceptance."""
import base64,contextlib,hashlib,io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
H=Path(__file__).resolve().parents[1]
FILES=('render_control_tools.py','deploy_source.py','deploy_tools.py','deploy_native_cpu.py','submit_reviewed.py','publish_gpu_init.py','publish_first5_release.py','watch_actual_scheduler.py','collect_pre_gpu.py','snapshot_actual.py','transport.py','finalize_runtime_tools.py')
class Interfaces(unittest.TestCase):
 def load(self,name):
  fake=types.ModuleType('transport');fake.remote=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Unexpected actual remote call'))
  module=types.ModuleType('fixture_'+name);module.__file__=str(H/(name+'.py'))
  with mock.patch.dict(sys.modules,{'transport':fake}):exec(compile(Path(module.__file__).read_bytes(),module.__file__,'exec'),module.__dict__)
  return module
 def exercise_collector(self,name,foreign=False):
  module=self.load(name)
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary).resolve();new=root/'HNEW';old=root/'OLD';new.mkdir();old.mkdir();module.HERE=new
   (new/'actual-job-id.txt').write_text('99999\n') # Explicit synthetic CLI identity only.
   binding=(old if foreign else new)/'synthetic-binding.json'
   binding.write_text(json.dumps({'remote_root':'/cpu-only-unlaunched/synthetic','source_commit':'0'*40}))
   output=root/'output';calls=[]
   ready=b'{"scope":"CPU_FIXTURE_NOT_ACTUAL_READY"}'
   record={'scope':'CPU_FIXTURE_NOT_ACTUAL_JOB','job_id':'99999','run_root':'/cpu-only-unlaunched/synthetic/runs/formal-99999','files':{},'live':{}}
   if name=='collect_pre_gpu':record['files']['approval-ready.json']={'base64':base64.b64encode(ready).decode(),'sha256':hashlib.sha256(ready).hexdigest()}
   def remote(command,**kwargs):
    calls.append(command)
    self.assertIn('/cpu-only-unlaunched/synthetic/runs/formal-99999',command)
    self.assertNotIn('/runs/formal-3299',command)
    return types.SimpleNamespace(stdout=json.dumps(record).encode(),stderr=b'CPU fixture remote stub; no command executed')
   module.remote=remote
   argv=[name,'--output',str(output),'--binding',str(binding)]
   with mock.patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()) as printed:
    if foreign:
     with self.assertRaises(AssertionError):module.main()
     self.assertEqual(calls,[]);self.assertFalse(output.exists())
    else:
     module.main();self.assertEqual(len(calls),1)
     self.assertEqual((output/'transport.stderr.original').read_bytes(),b'CPU fixture remote stub; no command executed')
     self.assertEqual(json.loads((output/'actual-observation.json').read_bytes())['job_id'],'99999')
     self.assertNotIn('INITIALIZATION_APPROVED',printed.getvalue())
 def test_collect_new_explicit_binding(self):self.exercise_collector('collect_pre_gpu')
 def test_collect_foreign_binding_rejected_before_remote(self):self.exercise_collector('collect_pre_gpu',True)
 def test_snapshot_new_explicit_binding(self):self.exercise_collector('snapshot_actual')
 def test_snapshot_foreign_binding_rejected_before_remote(self):self.exercise_collector('snapshot_actual',True)
 def test_local_gate_and_renderer_use_new_source_checkout(self):
  self.assertIn("path=HERE.parent/'opsd-unclipped-multitraj200-nativehead/search_r1/baselines/checkpoint_gate.py'",(H/'publish_first5_release.py').read_text())
  self.assertIn("REPO=HERE.parent/'opsd-unclipped-multitraj200-nativehead'",(H/'render_control_tools.py').read_text())
 def test_all_control_syntax_only_and_public_key_readonly_delta(self):
  for name in FILES:compile((H/name).read_bytes(),str(H/name),'exec',dont_inherit=True)
  source=(H/'deploy_source.py').read_text();self.assertNotIn("publish(ROOT+'/review-signing/public.json'",source)
  self.assertIn("p.stat().st_uid==0",source);self.assertIn("hashlib.sha256(p.read_bytes()).hexdigest()",source)
if __name__=='__main__':unittest.main(verbosity=2)
