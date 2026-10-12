import copy,os,unittest
from unittest.mock import patch
import search_r1.baselines.recursive_transport
import test_tree_behavior_mu_data_cpu as f
import search_r1.baselines.tree_raw_tokens as raw
from search_r1.baselines.tree_behavior_mu import BehaviorMuProofCache,validate_behavior_mu_batch,runtime_provenance

class ProofCache(unittest.TestCase):
 def setUp(self):
  self.f=f.BehaviorMuDataTests();self.f.setUp()
  stop=dict(protocol='vllm_native_eos_and_stop_set_v1',primary_eos_token_id=151645,source='detached_sampling_params_clone_native_update_from_generation_config_v0_6_3',primary_eos_source='native_input_preprocessor_get_eos_token_id_no_lora',generation_config_sha256=raw.fingerprint({}),generation_config_eos_token_ids=[],sampling_stop_token_ids=[],sampling_all_stop_token_ids=[151645],effective_stop_token_ids=[151645])
  self.p=patch.object(f,'capture_native_emissions',side_effect=lambda *a,**k:raw.capture_native_emissions(*a,native_stopping=stop,**k));self.p.start()
 def tearDown(self):self.p.stop();self.f.tearDown()
 def test_compose_and_driver_share_proof_without_changing_output(self):
  import torch
  states=[self.f.state(),self.f.state(call='second')]
  original=self.f.output(states)
  cache=BehaviorMuProofCache();self.f.engine.behavior_mu_proof_cache=cache
  cached=self.f.output(states)
  self.assertEqual(cache.misses,2);self.assertEqual(cache.hits,2)
  for key in original.batch:self.assertTrue(torch.equal(original.batch[key],cached.batch[key]))
  for key in original.non_tensor_batch:
   self.assertEqual(original.non_tensor_batch[key].tolist(),cached.non_tensor_batch[key].tolist())
  expected=runtime_provenance(os.environ)
  validate_behavior_mu_batch(cached,expected_provenance=expected,proof_cache=cache)
  self.assertEqual(cache.misses,2);self.assertEqual(cache.hits,4)
  bad=copy.deepcopy(cached);bad.batch['sampled_old_log_probs'][0,-1]=1
  with self.assertRaises(ValueError):validate_behavior_mu_batch(bad,expected_provenance=expected,proof_cache=cache)
 def test_resealed_receipt_cannot_mutate_cached_native_proof(self):
  from search_r1.baselines.tree_behavior_mu import FIELD
  cache=BehaviorMuProofCache();self.f.engine.behavior_mu_proof_cache=cache
  data=self.f.output()
  binding=data.non_tensor_batch[FIELD][0]
  binding['action_receipts'][0]['sample_id']='mutated-after-compose'
  binding['binding_sha256']=raw.fingerprint({k:v for k,v in binding.items() if k!='binding_sha256'})
  with self.assertRaisesRegex(ValueError,'Sampled action/context/identity binding changed'):
   validate_behavior_mu_batch(data,expected_provenance=runtime_provenance(os.environ),proof_cache=cache)
 def test_repeat_and_permutation_keep_tensor_mu_mask_checks(self):
  data=self.f.output([self.f.state(),self.f.state(call='second')]);cache=BehaviorMuProofCache()
  expected=runtime_provenance(os.environ)
  first=validate_behavior_mu_batch(data,expected_provenance=expected,proof_cache=cache)
  second=validate_behavior_mu_batch(data,expected_provenance=expected,proof_cache=cache)
  self.assertEqual(first,second);self.assertEqual(cache.hits,2)
  # Use actual tiny DataProto permutation path (all row fields move together).
  perm=data[[1,0]]
  self.assertEqual(validate_behavior_mu_batch(perm,expected_provenance=expected,proof_cache=cache),first)
  for key in ('sampled_old_log_probs','info_mask'):
   bad=copy.deepcopy(data);bad.batch[key][0,-1]=1
   with self.assertRaises(ValueError):validate_behavior_mu_batch(bad,expected_provenance=expected,proof_cache=cache)
 def test_modified_ledger_and_tokenizer_cannot_reuse_cached_proof(self):
  data=self.f.output();cache=BehaviorMuProofCache();e=runtime_provenance(os.environ)
  validate_behavior_mu_batch(data,expected_provenance=e,proof_cache=cache)
  bad=copy.deepcopy(data);bad.non_tensor_batch[raw.FIELD][0]['raw_ledger'][0]['observation_token_ids']=[6]
  with self.assertRaises(ValueError):validate_behavior_mu_batch(bad,expected_provenance=e,proof_cache=cache)
  self.f.tok.backend_normalizer={'changed':True}
  with self.assertRaises(ValueError):validate_behavior_mu_batch(data,expected_provenance=e,proof_cache=cache)
if __name__=='__main__':unittest.main()
