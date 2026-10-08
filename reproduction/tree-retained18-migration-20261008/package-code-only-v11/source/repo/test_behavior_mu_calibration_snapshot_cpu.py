"""Actual zero-write snapshot helper on CPU; no native/FSDP/model acceptance."""
from types import SimpleNamespace as N
import unittest
import numpy as np
import torch
from search_r1.baselines.sampling_consistency import calibration_monitor_snapshot
from search_r1.baselines.policy_safety_runtime import PolicySafetyRuntime
from search_r1.baselines.tree_behavior_mu import FIELD

class CalibrationSnapshotTests(unittest.TestCase):
    def fixture(self):
        mu=torch.tensor([[-.25,0.],[-.5,0.]],dtype=torch.float32)
        full=torch.tensor([[-.75,float('nan')],[-1.,float('nan')]])
        monitor={'old_log_probs':mu,'sampled_old_log_probs':mu.clone(),'actor_old_log_probs':full,
                 'responses':torch.tensor([[1,0],[2,0]]),'attention_mask':torch.tensor([[1,1,0],[1,1,0]]),
                 'loss_mask':torch.tensor([[1,0],[1,0]])}
        frozen=torch.tensor([[-1.5,float('nan')],[-2.,float('nan')]])
        safety=N(monitor=monitor,actor_monitor_beta=frozen,indices=torch.tensor([2,0]),
                 effective_mask=PolicySafetyRuntime.effective_mask)
        data=N(non_tensor_batch={FIELD:np.array([{'id':'first'},{'id':'second'},{'id':'third'}],dtype=object)})
        return safety,data
    def test_actual_mu_full_beta_frozen_partition_beta_and_row_order_remain_separate(self):
        safety,data=self.fixture(); saved=calibration_monitor_snapshot(safety,data,True)
        self.assertEqual(saved['old_log_probs'].dtype,torch.float32)
        self.assertEqual(saved['actor_old_log_probs'][0,0].item(),-.75)
        self.assertEqual(saved['actor_same_partition_beta_monitor'][0,0].item(),-1.5)
        self.assertEqual([x['id'] for x in saved['behavior_mu_source_row_bindings']],['third','first'])
        safety.actor_monitor_beta[0,0]=-8.; safety.monitor['old_log_probs'][0,0]=-9.
        self.assertEqual(saved['actor_same_partition_beta_monitor'][0,0].item(),-1.5)
        self.assertEqual(saved['old_log_probs'][0,0].item(),-.25)
    def test_padding_nan_kept_as_evidence_does_not_refuse_valid_snapshot(self):
        safety,data=self.fixture(); saved=calibration_monitor_snapshot(safety,data,True)
        self.assertTrue(torch.isnan(saved['actor_same_partition_beta_monitor'][:,1]).all())
    def test_missing_bad_shape_or_nonfinite_valid_frozen_beta_refuses(self):
        for value in [None,torch.zeros(1,2),torch.ones(2,2,dtype=torch.int64),torch.full((2,2),float('nan'))]:
            safety,data=self.fixture(); safety.actor_monitor_beta=value
            with self.assertRaises(ValueError):calibration_monitor_snapshot(safety,data,True)
    def test_legacy_profile_has_no_extra_behavior_source_requirements(self):
        safety,data=self.fixture(); safety.actor_monitor_beta=None
        data.non_tensor_batch={}; saved=calibration_monitor_snapshot(safety,data,False)
        self.assertNotIn('actor_same_partition_beta_monitor',saved)
        self.assertNotIn('behavior_mu_source_row_bindings',saved)
        self.assertTrue(torch.equal(saved['old_log_probs'],safety.monitor['old_log_probs']))
if __name__=='__main__':unittest.main()
