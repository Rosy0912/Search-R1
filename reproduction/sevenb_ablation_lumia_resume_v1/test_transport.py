import copy,json,unittest
from pathlib import Path
from .contract import BASE,GROUPS,STEPS,SCIENCE,MODEL,DATA,project_config,validate_plan,mapped_manifest
class TransportTests(unittest.TestCase):
 def fixture(self,g='budget-5'):
  return dict(group=g,job='zhangjunming-7b-abl-'+GROUPS[g]+'-lresume1',root=str(BASE/'runs'/g),source_commit=SCIENCE,actual_update_limit=150,scheduler_scale=200,warmup_rounds=57,parent_step=STEPS[g],fresh_from_common_initialization=False,allocation={'gpus':8,'cpus':64,'memory_gib':900,'partition':'ADA6000,L40S','min_gpu_memory_mib':44000},no_auto_retry=True,science_unchanged=True,world_size=8,model=MODEL,data_root=DATA)
 def test_all_five(self):
  for g in GROUPS:self.assertEqual(validate_plan(self.fixture(g)),BASE/'runs'/g)
 def test_wrong_step(self):
  p=self.fixture();p['parent_step']+=1
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_fresh_reset(self):
  p=self.fixture();p['fresh_from_common_initialization']=True
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_wrong_world(self):
  p=self.fixture();p['world_size']=16
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_wrong_science(self):
  p=self.fixture();p['source_commit']='0'*40
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_wrong_allocation(self):
  p=self.fixture();p['allocation']['memory_gib']=600
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_wrong_root(self):
  p=self.fixture();p['root']+='/old'
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_wrong_scheduler(self):
  p=self.fixture();p['scheduler_scale']=150
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_wrong_target(self):
  p=self.fixture();p['actual_update_limit']=200
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_no_retry(self):
  p=self.fixture();p['no_auto_retry']=False
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_no_combined_condition(self):
  p=self.fixture();p['group']='budget-10'
  with self.assertRaises(AssertionError):validate_plan(p)
if __name__=='__main__':unittest.main()
