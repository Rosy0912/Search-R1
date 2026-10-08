"""Local isolated durability-order fixtures only; not actual Slurm proof."""
from pathlib import Path
import importlib.util,tempfile,unittest,json
from unittest.mock import patch
H=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('candidate',H/'first5_cpu_inspection_once_candidate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Delta(unittest.TestCase):
 def test_file_fsync_precedes_parent_fsync(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'raw';events=[]
   with patch.object(m.os,'fsync',side_effect=lambda fd:events.append('file-fsync')),patch.object(m,'sync_dir',side_effect=lambda p:events.append(('dir-fsync',str(p)))):m.write_once(p,{'fixture':True})
   self.assertEqual(events,['file-fsync',('dir-fsync',str(p.parent))]);self.assertTrue(p.exists())
 def test_launch_parent_fsync_precedes_popen(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d)/'run';e=run/'evidence';e.mkdir(parents=True);out=e/'once';events=[]
   a=type('A',(),dict(job_id='3304',controller_sha256='a'*64,checkpoint_manifest_sha256='b'*64,pause_nonce='c'*64,driver_pid=999,driver_start_ticks='123'))()
   def pop(*args,**kwargs):events.append('Popen');raise RuntimeError('LOCAL_DURABILITY_FIXTURE_STOP_BEFORE_REAL_LAUNCH')
   with patch.object(m,'RUN',run),patch.object(m,'OUTPUT',out),patch.object(m,'self_pin',lambda *a:None),patch.object(m,'physical',lambda p,*a:p),patch.object(m,'context',lambda *a:{'scope':'LOCAL_MOCK'}),patch.object(m,'stable_raw',lambda p,*a:p.read_bytes()),patch.object(m,'sync_dir',side_effect=lambda p:events.append(('dir',str(p)))),patch.object(m.subprocess,'Popen',side_effect=pop):
    with self.assertRaisesRegex(RuntimeError,'LOCAL_DURABILITY'):m.start(a)
   self.assertIn(('dir',str(e)),events);self.assertIn(('dir',str(out)),events);self.assertEqual(events[-1],'Popen');self.assertTrue((out/'launch-original.json').exists())
 def test_sync_directory_real_local_fixture(self):
  with tempfile.TemporaryDirectory() as d:m.sync_dir(Path(d))
if __name__=='__main__':unittest.main(verbosity=2)
