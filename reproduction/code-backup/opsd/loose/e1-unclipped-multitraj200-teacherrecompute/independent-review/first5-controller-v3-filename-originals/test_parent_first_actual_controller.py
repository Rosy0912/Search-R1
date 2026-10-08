import importlib.util,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
H=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute')
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class T(unittest.TestCase):
 def pair(self,old):
  path=H/('first5_cpu_inspection_once_candidate-v2.original.py' if old else 'first5_cpu_inspection_once_candidate.py');m=load('parent',path)
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)
   child="import importlib.util,time,sys;from pathlib import Path;time.sleep(.25);p="+repr(str(path))+";s=importlib.util.spec_from_file_location('child',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.OUTPUT=Path("+repr(d)+");m.context=lambda c:{'scope':'LOCAL_FIXTURE_CONTEXT'};m.process=lambda pid,*a,**k:{'pid':pid,'scope':'LOCAL_FIXTURE_IDENTITY'};m.srun_command=lambda *a:[sys.executable,'-c','print(\"LOCAL_SRUN_FIXTURE\")'];sys.exit(m.controller({},Path('fixture'), 'a'*64))"
   with patch.object(m,'process',lambda pid,*a,**k:{'pid':pid,'scope':'LOCAL_FIXTURE_IDENTITY'}):rc=m.wait_child([sys.executable,'-c',child],out,'controller')
   e=json.loads((out/'controller-exit.original.json').read_bytes());self.assertEqual(e['returncode'],rc);self.assertTrue(e['waited'])
   if old:
    self.assertNotEqual(rc,0);self.assertIn('FileExistsError',(out/'controller-stderr.original').read_text());self.assertFalse((out/'srun-exit.original.json').exists())
   else:
    self.assertEqual(rc,0)
    for n in ['controller-start.original.json','controller-child-start.original.json','srun-child-start.original.json','srun-exit.original.json']:self.assertTrue((out/n).is_file())
    own=json.loads((out/'controller-start.original.json').read_bytes());parent=json.loads((out/'controller-child-start.original.json').read_bytes());self.assertEqual(parent['child_pid'],own['process']['pid']);self.assertEqual(json.loads((out/'srun-exit.original.json').read_bytes())['returncode'],0)
 def test_old_parent_first_actual_controller_failure(self):self.pair(True)
 def test_new_parent_first_actual_controller_nested_wait(self):self.pair(False)
if __name__=='__main__':unittest.main(verbosity=2)
