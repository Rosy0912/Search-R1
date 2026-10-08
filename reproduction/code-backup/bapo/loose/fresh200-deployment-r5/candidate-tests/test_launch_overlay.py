"""Local schema counterexamples; no actual R5 remote admission."""
import copy,importlib.util,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'publish-launch-approval.CANDIDATE-UNBOUND.py'
spec=importlib.util.spec_from_file_location('r5_launch_candidate',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class OverlayTests(unittest.TestCase):
 def setUp(self):
  self.off={'verl/utils/fsdp_utils.py','search_r1/baselines/fresh200_contract.py','tests/utils/test_fsdp_optimizer_transfer.py'}
  self.bound={'search_r1/llm_agent/sampling_trace.py','tests/test_fresh200_sampling_trace.py','tests/test_fresh200_sampling_token_boundary.py','tests/baselines/test_fresh200_bapo_sampling_chain.py'}
  self.overlay=dict(status='PASS_CODE_ONLY_OFFLOAD_FIX_INDEPENDENT_REVIEW',base_source='379b14c65e127cd3a71524f7783deaebafbbdb41',must_fix=[],native_acceptance=False,files_sha256={x:'offload'for x in self.off})
  self.boundary=dict(status='PASS_CODE_ONLY_TOKEN_BOUNDARY_FIX_INDEPENDENT_REVIEW',base_source_commit='5121a8d9349eadb5a81119cc4573165338e041bd',mustfix=[],native_acceptance=False,files_sha256={x:'boundary'for x in self.bound})
 def merge(self,b=None,o=None,c=None):return m.merge_source_overlay({'untouched':'original'},o or self.overlay,self.off,b or self.boundary,c if c is not None else self.bound)
 def test_unbound_actual_records_refused(self):
  with self.assertRaisesRegex(ValueError,'binding not ready'):m.build_approval('a'*64)
 def test_exact_three_overlay_chain(self):
  result=self.merge();self.assertEqual(result['untouched'],'original');self.assertEqual(set(result),self.off|self.bound|{'untouched'})
 def test_prefix_unreviewed_file_refused(self):
  with self.assertRaises(ValueError):self.merge(c=self.bound|{'verl/workers/actor/dp_actor.py'})
 def test_missing_boundary_file_refused(self):
  b=copy.deepcopy(self.boundary);b['files_sha256'].pop('tests/test_fresh200_sampling_token_boundary.py')
  with self.assertRaises(ValueError):self.merge(b=b)
 def test_old_base_boundary_refused(self):
  b=copy.deepcopy(self.boundary);b['base_source_commit']='379b14c65e127cd3a71524f7783deaebafbbdb41'
  with self.assertRaises(ValueError):self.merge(b=b)
 def test_native_claim_not_code_review(self):
  b=copy.deepcopy(self.boundary);b['native_acceptance']=True
  with self.assertRaises(ValueError):self.merge(b=b)
if __name__=='__main__':unittest.main()
