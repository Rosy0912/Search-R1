import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]/'work/tree-speed-repair-20261008/nested-branch-source';sys.path.insert(0,str(ROOT))
import test_tree_behavior_mu_data_cpu as fixture
from search_r1.baselines.tree_raw_tokens import capture_native_emissions,fingerprint,semantic_spans
from search_r1.baselines.tree_behavior_mu import TENSOR,validate_behavior_mu_batch
STOP=dict(protocol='vllm_native_eos_and_stop_set_v1',primary_eos_token_id=151645,source='detached_sampling_params_clone_native_update_from_generation_config_v0_6_3',primary_eos_source='native_input_preprocessor_get_eos_token_id_no_lora',generation_config_sha256='a'*64,generation_config_eos_token_ids=[151645],sampling_stop_token_ids=[],sampling_all_stop_token_ids=[151645],effective_stop_token_ids=[151645])
from verl.trainer.main_ppo import RewardManager
class RawMu(unittest.TestCase):
 def test_real_projection_scoring_mu_compose_equivalence(self):
  f=fixture.BehaviorMuDataTests();f.setUp()
  try:
   with patch.object(fixture,'capture_native_emissions',lambda *a,**k:capture_native_emissions(*a,**k,native_stopping=STOP)):
    s=f.state();child=s.branch();before=fingerprint(s.raw_ledger)
    scored,src=f.engine.reward_batch([s,child]);composed,_=f.engine.compose([s,child])
    self.assertNotIn(TENSOR,scored.batch);self.assertIn(TENSOR,composed.batch)
    self.assertEqual(semantic_spans(scored[0],f.tok),semantic_spans(composed[0],f.tok))
    self.assertEqual(semantic_spans(scored[1],f.tok),semantic_spans(composed[1],f.tok))
    self.assertEqual(fingerprint(s.raw_ledger),before);self.assertEqual(s.raw_ledger,child.raw_ledger)
    result=f.validate(composed);self.assertTrue(result)
    for data in (scored,composed):
     data.non_tensor_batch['data_source']=np.array(['nq','nq'])
     data.non_tensor_batch['reward_model']=np.array([{'ground_truth':{'target':['Paris']}},{'ground_truth':{'target':['Paris']}}],dtype=object)
    rm=RewardManager(f.tok,0)
    self.assertEqual(rm(scored).sum(-1).tolist(),rm(composed).sum(-1).tolist())
    self.assertEqual(rm(scored).sum(-1).tolist(),[1.,1.])
  finally:f.tearDown()
if __name__=='__main__':unittest.main(verbosity=2)
