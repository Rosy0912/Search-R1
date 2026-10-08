import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'collect-numeric-terminal.CANDIDATE.py'
s=importlib.util.spec_from_file_location('terminal_collector',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
BASE=P.parents[1];ACTUAL=BASE/'evidence/fresh200-numeric-diag3311-slurm-terminal.actual.json'
class Terminal(unittest.TestCase):
 def actual(self):return json.loads(ACTUAL.read_text())
 def test_actual_terminal_115_seconds(self):self.assertEqual(m.terminal(self.actual())[2],115)
 def test_running_rejected(self):
  v=self.actual();v['stdout']=v['stdout'].replace('JobState=COMPLETED','JobState=RUNNING')
  with self.assertRaisesRegex(ValueError,'JobState'):m.terminal(v)
 def test_boolean_job_rejected(self):
  v=self.actual();v['job_id']=True
  with self.assertRaisesRegex(ValueError,'SlurmJSON'):m.terminal(v)
 def test_wrong_nonce_rejected(self):
  v=self.actual();v['stdout']=v['stdout'].replace(m.NONCE,'a'*32)
  with self.assertRaisesRegex(ValueError,'Comment'):m.terminal(v)
 def test_requeue_rejected(self):
  v=self.actual();v['stdout']=v['stdout'].replace('Requeue=0','Requeue=1')
  with self.assertRaisesRegex(ValueError,'Requeue'):m.terminal(v)
 def test_duplicate_json_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'x';p.write_text('{"job_id":3311,"job_id":3307}')
   with self.assertRaisesRegex(ValueError,'Duplicate'):m.read(p)
if __name__=='__main__':unittest.main()
