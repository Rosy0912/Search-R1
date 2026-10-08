import unittest,pathlib,importlib.util,json,copy
B=pathlib.Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('c',B/'check-formal-allocation.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
w=json.loads((B.parent/'evidence/ray-envfix-formal3292-existing-wait-child-env.json').read_text());stage=w['stage_sha256'];pid=w['proofs'][0]['parent_before']['pid']
class T(unittest.TestCase):
 def check(self,x):return m.validate_waiting_environment(x,3292,stage,pid)
 def test_actual(self):self.assertTrue(self.check(w))
 def test_identity_negative(self):
  for key,val in [('job_id',3284),('source_commit','old'),('run_root','/foreign'),('signals_sent',1),('processes_spawned',1),('stage_sha256','0'*64)]:
   x=copy.deepcopy(w);x[key]=val
   with self.subTest(key=key),self.assertRaises(AssertionError):self.check(x)
 def test_epoch_and_uid(self):
  for key in ['start_ticks','uid','ppid']:
   x=copy.deepcopy(w);x['proofs'][0]['child_after'][key]+=1
   with self.subTest(key=key),self.assertRaises(AssertionError):self.check(x)
 def test_rebound_foreign_exe(self):
  x=copy.deepcopy(w)
  for tag in ['child_before','child_after']:x['proofs'][0][tag]['executable']['resolved_path']='/foreign/sleep'
  with self.assertRaises(AssertionError):self.check(x)
 def test_marker_raw_rebind(self):
  x=copy.deepcopy(w)
  for tag in ['child_before','child_after']:x['proofs'][0][tag]['environment']['SPT_NOENV']='0'
  with self.assertRaises(AssertionError):self.check(x)
 def test_parent_wrong_stage(self):
  x=copy.deepcopy(w)
  for tag in ['parent_before','parent_after']:x['proofs'][0][tag]['command']=['bash','foreign','formal300']
  with self.assertRaises(AssertionError):self.check(x)
 def test_bool_pid(self):
  x=copy.deepcopy(w)
  for tag in ['child_before','child_after']:x['proofs'][0][tag]['pid']=True
  with self.assertRaises(AssertionError):self.check(x)
if __name__=='__main__':unittest.main()
