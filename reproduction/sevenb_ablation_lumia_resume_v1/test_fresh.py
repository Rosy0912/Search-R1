import copy,unittest
from .test_transport import TransportTests
from .fresh_contract import BASE,validate_plan
class FreshTests(unittest.TestCase):
 def fixture(self):
  p=TransportTests().fixture();p.update(root=str(BASE/'fresh-runs'/'budget-5'),job='zhangjunming-7b-abl-b5-lfresh1',protocol='sevenb_ablation_lumia_fresh150_after_human_resume_v1',fresh_from_common_initialization=True,parent_step=0,parent_checkpoint=None,superseded_condition_state='TERMINATED',parent_uid='9a93ac8f-a015-4129-9ccf-ff00c1ea9561',run_id='a'*32);return p
 def test_fresh_full_budget(self):self.assertEqual(validate_plan(self.fixture()),BASE/'fresh-runs'/'budget-5')
 def test_no_old_round_credit(self):
  p=self.fixture();p['parent_step']=45
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_no_cold_resume_mix(self):
  p=self.fixture();p['parent_checkpoint']='old-native'
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_no_live_duplication(self):
  p=self.fixture();p['superseded_condition_state']='RUNNING'
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_old_namespace_refused(self):
  p=self.fixture();p['root']=str(BASE/'runs'/'budget-5')
  with self.assertRaises(AssertionError):validate_plan(p)
