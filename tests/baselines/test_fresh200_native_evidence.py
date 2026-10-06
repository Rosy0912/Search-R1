import copy
import math
import unittest
import torch
from search_r1.baselines.fresh200_native_evidence import replay_matrices, verify_native_sample, verify_independent_review


class NativeReplayCounterexamples(unittest.TestCase):
    def payload(self):
        return dict(denominator_role='actual_native_stochastic_sampling_policy',hf_probability_role='unchanged_hf_recomputation_diagnostic',
            frozen_hf_log_probs=torch.tensor([[-2.5,-2.5],[-2.5,-2.5]],dtype=torch.float64),old_log_probs=torch.tensor([[-2.,-2.],[-2.,-2.]],dtype=torch.float64),
            current_log_probs=torch.tensor([[-.8,-2.],[-2.,-2.]],dtype=torch.float64),
            advantages=torch.tensor([[-1.,1.],[-1.,-1.]],dtype=torch.float64),
            mask=torch.tensor([[1.,0.],[1.,1.]],dtype=torch.float64),
            sampling_log_probs=torch.tensor([[-2.,-2.],[-2.,-2.]],dtype=torch.float64))
    def test_independent_sequence_weighted_recompute(self):
        replay=replay_matrices(self.payload(),2)
        self.assertAlmostEqual(replay['negative_tail_weight_fraction'],.5)
        self.assertAlmostEqual(replay['sampled_old_policy_kl'],(math.exp(1.2)-1-1.2)/2)
        self.assertEqual(replay['effective_token_count'],3)
    def test_roles_and_hf_diagnostic_are_required(self):
        for key in ('denominator_role','hf_probability_role','frozen_hf_log_probs'):
            bad=self.payload();bad.pop(key)
            with self.assertRaises(ValueError):replay_matrices(bad,2)
        bad=self.payload();bad['frozen_hf_log_probs']=bad['frozen_hf_log_probs'].long()
        with self.assertRaises(ValueError):replay_matrices(bad,2)
    def test_boolean_claims_without_matrices_do_not_admit(self):
        with self.assertRaises(ValueError):replay_matrices(dict(cuda_probability_parity=True,context_audit=True,fresh_model=True),2)
        with self.assertRaises(ValueError):verify_native_sample(dict(scope='full_frozen_rollout',global_mini_batch_size=128,native_shards=[]),{},'discovery',{128:'/run',256:'/other'})
    def test_sampled_probability_change_is_rejected(self):
        bad=self.payload();bad['sampling_log_probs'][0,0]-=.01
        with self.assertRaises(ValueError):replay_matrices(bad,2)
    def test_nonfinite_or_nonbinary_rejected(self):
        for field in ('old_log_probs','current_log_probs','advantages','sampling_log_probs','frozen_hf_log_probs'):
            bad=self.payload();bad[field][0,0]=float('nan')
            with self.assertRaises(ValueError):replay_matrices(bad,2)
        bad=self.payload();bad['mask'][0,0]=.5
        with self.assertRaises(ValueError):replay_matrices(bad,2)
    def test_false_weights_and_scope_rejected(self):
        bad=self.payload();bad['weights']=torch.ones((2,2),dtype=torch.float64)
        with self.assertRaises(ValueError):replay_matrices(bad,2)
        with self.assertRaises(ValueError):replay_matrices(self.payload(),16)
    def test_full_write_index_coverage_is_required(self):
        from search_r1.baselines.fresh200_native_evidence import validate_diagnostic_ledger
        for mini in (128,256):
            expected=2560//mini
            ledger=dict(status='DIAGNOSTIC_COMPLETE_SEQUENCE',role='discovery',global_mini_batch_size=mini,
                optimizer_writes=expected,expected_optimizer_steps=expected,formal_outer_updates=0,
                scheduler_outer_steps=0,rank=0,native_process={},stop_reason=None,actual_optimizer_lr=[1e-6],scheduler_cached_last_lr=[1e-6/57],
                write_ledger=[dict(write_index=i) for i in range(expected)],
                events=[dict(diagnostic_write_index=i,scope=scope,phase=phase) for i in range(expected)
                    for scope in ('full_frozen_rollout','full_current_minibatch') for phase in ('pre_adam','post_adam')])
            validate_diagnostic_ledger(ledger,mini,'discovery',0,{})
            for mutation in ('one_write','missing_post','duplicate_index','formal_count','tiny_lr','cached_lr_claim'):
                bad=copy.deepcopy(ledger)
                if mutation=='one_write':bad['optimizer_writes']=1
                if mutation=='missing_post':bad['events'].pop()
                if mutation=='duplicate_index':bad['write_ledger'][-1]['write_index']=0
                if mutation=='formal_count':bad['formal_outer_updates']=1
                if mutation=='tiny_lr':bad['actual_optimizer_lr']=[1e-6/57]
                if mutation=='cached_lr_claim':bad['scheduler_cached_last_lr']=[1e-6]
                with self.assertRaises(ValueError):validate_diagnostic_ledger(bad,mini,'discovery',0,{})

    def test_self_declared_independent_review_rejected(self):
        with self.assertRaises(ValueError):verify_independent_review(dict(decision='PASS_BAPO_NATIVE_REPLAY'),{}, {})

if __name__=='__main__':unittest.main()
