import importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'publish-launch-approval.CANDIDATE.py'
s=importlib.util.spec_from_file_location('new_publisher',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
BASE=P.parents[2]
class Overlay(unittest.TestCase):
 def review(self):return json.loads((BASE/'outputs/fresh200/3312-CONFIG-COUNTER-REPAIR-4FILE-INDEPENDENT-REVIEW.json').read_text())
 def test_actual_four_file_overlay(self):
  v=self.review();got=m.merge_new_source_overlay({},v,set(v['files']));self.assertEqual(set(got),set(v['files']))
 def test_unreviewed_forward_change_refused(self):
  v=self.review()
  with self.assertRaisesRegex(ValueError,'Only two'):m.merge_new_source_overlay({},v,set(v['files'])|{'verl/workers/actor/dp_actor.py'})
 def test_wrong_ancestor_refused(self):
  v=self.review();v['base_commit']=m.AEC_SOURCE
  with self.assertRaisesRegex(ValueError,'unresolved'):m.merge_new_source_overlay({},v,set(v['files']))
 def test_native_acceptance_claim_refused(self):
  v=self.review();v['native_acceptance']=True
  with self.assertRaisesRegex(ValueError,'unresolved'):m.merge_new_source_overlay({},v,set(v['files']))
 def test_numeric_ancestor_is_explicit(self):
  self.assertNotEqual(m.SOURCE,m.ANCESTOR_SOURCE)
  self.assertEqual(m.NUMERIC_ROOT,m.ANCESTOR_ROOT/'numeric-diagnostic-3310-20261006')
 def test_new_actual_prep_cannot_use_prior_receipts(self):
  from unittest.mock import patch
  with patch.object(m,'CPU_RAW_SHA',None):
   with self.assertRaisesRegex(ValueError,'binding not ready'):m.build_approval('0'*64)
if __name__=='__main__':unittest.main()
