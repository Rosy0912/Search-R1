"""Real local subprocess writes; mocked owner metadata is not Slurm evidence."""
import importlib.util,json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
H=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('current',H/'first5_cpu_inspection_once_candidate.py')
old=load('old_v2',H/'first5_cpu_inspection_once_candidate-v2.original.py')
class Delta(unittest.TestCase):
 def actual_pair(self,module):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);childcode='from pathlib import Path;import json,time;d=Path('+repr(d)+');(d/"controller-start.original.json").open("xb").write(b"{\\\"scope\\\":\\\"REAL_LOCAL_CHILD_CONTEXT\\\"}");(d/"child-ready").write_text("ready");time.sleep(.08)'
   def metadata(pid):
    until=time.monotonic()+3
    while not (out/'child-ready').exists():
     if time.monotonic()>until:raise RuntimeError('Child did not actually write context')
     time.sleep(.003)
    return {'pid':pid,'scope':'LOCAL_OWNER_MOCK_NOT_SLURM'}
   with patch.object(module,'process',metadata):
    if module is old:
     with self.assertRaises(FileExistsError):module.wait_child([sys.executable,'-c',childcode],out,'controller')
    else:self.assertEqual(module.wait_child([sys.executable,'-c',childcode],out,'controller'),0)
   self.assertEqual(json.loads((out/'controller-start.original.json').read_bytes())['scope'],'REAL_LOCAL_CHILD_CONTEXT')
   self.assertEqual(json.loads((out/'controller-exit.original.json').read_bytes())['returncode'],0)
   if module is m:self.assertEqual(json.loads((out/'controller-child-start.original.json').read_bytes())['scope'],m.SCOPE)
 def test_old_v2_real_collision_preserved(self):self.actual_pair(old)
 def test_v3_parent_child_context_both_preserved(self):self.actual_pair(m)
 def test_only_metadata_filename_bytes_changed(self):
  self.assertEqual((H/'first5_cpu_inspection_once_candidate-v2.original.py').read_bytes().replace(b"prefix+'-start.original.json'",b"prefix+'-child-start.original.json'"),(H/'first5_cpu_inspection_once_candidate.py').read_bytes())
if __name__=='__main__':unittest.main(verbosity=2)
