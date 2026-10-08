import copy,importlib.util,json,unittest
from unittest.mock import patch
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'publish-launch-approval.CANDIDATE.py'
s=importlib.util.spec_from_file_location('behavior_publisher',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ROOT=Path(__file__).resolve().parents[3]
class ConditionalPublisher(unittest.TestCase):
 def overlay(self):return json.loads((ROOT/'outputs/fresh200/BEHAVIOR-POLICY-12FILE-OVERLAY-INDEPENDENT-REVIEW.json').read_text())
 def test_real_complete_twelve_file_overlay(self):
  value=self.overlay();result=m.merge_behavior_overlay({'unchanged':'x'},value,set(value['files']))
  self.assertEqual(len(value['files']),12);self.assertEqual(result['unchanged'],'x')
  self.assertEqual(result['search_r1/baselines/behavior_policy.py'],value['files']['search_r1/baselines/behavior_policy.py']['source_sha256'])
 def test_four_file_boundary_cannot_cover_behavior(self):
  value=self.overlay()
  with self.assertRaisesRegex(ValueError,'twelve-file'):m.merge_behavior_overlay({},value,set(list(value['files'])[:4]))
 def test_unreviewed_extra_source_rejected(self):
  value=self.overlay()
  with self.assertRaisesRegex(ValueError,'twelve-file'):m.merge_behavior_overlay({},value,set(value['files'])|{'unreviewed.py'})
 def test_legacy_base_source_rejected(self):
  value=self.overlay();value['base_commit']='5121a8d9349eadb5a81119cc4573165338e041bd'
  with self.assertRaisesRegex(ValueError,'overlay unresolved'):m.merge_behavior_overlay({},value,set(value['files']))
 def test_modified_file_requires_base_hash(self):
  value=self.overlay();name=next(n for n,v in value['files'].items()if v['status']=='M');value['files'][name]['base_sha256']=None
  with self.assertRaisesRegex(ValueError,'base binding'):m.merge_behavior_overlay({},value,set(value['files']))
 def test_candidate_cannot_admit_before_numeric(self):
  with patch.object(m,'NUMERIC_REVIEW_SHA',None):
   with self.assertRaisesRegex(ValueError,'UNBOUND'):m.build_approval('0'*64)
 def terminal(self):
  nonce='a'*32;root=m.ROOT/'numeric-diagnostic-3310-20261006'
  return dict(schema='ACTUAL_ZERO_ADAM_NUMERIC_TERMINAL_EVIDENCE_V1',root=str(root),source_commit=m.SOURCE,job_id=3311,scheduler_state='COMPLETED',exit_code=0,restarts=0,requeue=0,protocol='BAPO_ZERO_ADAM_NUMERIC_DIAGNOSTIC_V1',phase='numeric-zero-adam-replay',node='h100-5',nonce=nonce,run_root=str(root/'runs'/f'numeric-zero-adam-replay-{nonce}-3311'),allocated_gpu_count=8,computed_gpu_count=1)
 def test_prior_aec_diagnostic_not_new_source_acceptance(self):
  value=self.terminal();value['source_commit']=m.AEC_SOURCE
  with self.assertRaisesRegex(ValueError,'New-source'):m.validate_numeric_interpretation(value,{}, {})
 def test_nonterminal_diagnostic_cannot_authorize_firstfive(self):
  value=self.terminal();value['scheduler_state']='RUNNING'
  with self.assertRaisesRegex(ValueError,'completed terminal'):m.validate_numeric_interpretation(value,{}, {})
 def test_boolean_compute_count_rejected(self):
  value=self.terminal();value['computed_gpu_count']=True
  with self.assertRaisesRegex(ValueError,'allocation'):m.validate_numeric_interpretation(value,{}, {})
 def test_old3310_receipt_does_not_cover_new_numeric(self):
  value=self.terminal();value['job_id']=3310
  with self.assertRaisesRegex(ValueError,'completed terminal'):m.validate_numeric_interpretation(value,{}, {})
if __name__=='__main__':unittest.main()
