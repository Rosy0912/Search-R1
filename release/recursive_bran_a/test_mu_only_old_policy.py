"""Targeted real CPU probability/gradient checks, without HF/Ray/GPU claims."""
import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_tree_behavior_mu_actor_cpu as f
import search_r1.baselines.tree_raw_tokens as raw
from search_r1.baselines.policy_safety_runtime import PolicySafetyRuntime, PolicySafetyRefusal, validate_actor_probability_profile


class MuOnlyTests(unittest.TestCase):
    def setUp(self):
        self.factory = f.collector_fixture.BehaviorMuDataTests()
        self.factory.setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, RECURSIVE_A_OMIT_FULL_BETA='1', SEARCH_R1_TREE_RANK_AUDIT_DIR=self.tmp.name)
        self.env.start()
        # The reused synthetic fixture predates explicit native stop-set capture.
        # Supply its actual synthetic EOS contract, without relaxing production validation.
        stopping = dict(protocol='vllm_native_eos_and_stop_set_v1', primary_eos_token_id=151645,
            source='detached_sampling_params_clone_native_update_from_generation_config_v0_6_3',
            primary_eos_source='native_input_preprocessor_get_eos_token_id_no_lora',
            generation_config_sha256=raw.fingerprint({}), generation_config_eos_token_ids=[],
            sampling_stop_token_ids=[], sampling_all_stop_token_ids=[151645], effective_stop_token_ids=[151645])
        self.capture = patch.object(f.collector_fixture, 'capture_native_emissions',
            side_effect=lambda *args, **kwargs: raw.capture_native_emissions(*args, native_stopping=stopping, **kwargs))
        self.capture.start()
        self.cfg = f.config()
        self.cfg.update(use_kl_loss=True, kl_loss_coef=0., kl_loss_type='kl')

    def tearDown(self):
        self.capture.stop()
        self.env.stop()
        self.tmp.cleanup()
        self.factory.tearDown()

    def data(self, omitted):
        data = f.make_data(self.factory, self.cfg)
        data.batch['ref_log_prob'] = torch.full_like(data.batch['old_log_probs'], -.8)
        data.batch['advantages'][1::2].fill_(-1.)
        if omitted:
            del data.batch['old_log_probs']
            del data.batch['actor_old_log_probs']
            del data.meta_info['temperature']
            data.meta_info['tree_full_actor_beta_available'] = False
            f.trainer['prepare_behavior_mu_old_policy'](data, self.cfg)
        return data

    def test_same_mu_ppo_loss_gradient_and_real_adam_update(self):
        full, omitted = self.data(False), self.data(True)
        f.trainer['validate_final_behavior_mu_batch'](omitted, self.cfg)
        results = []
        for data in (full, omitted):
            actor = f.Actor(self.cfg)
            self.assertTrue(validate_actor_probability_profile(actor, data, self.cfg.policy_safety))
            current = actor.actor_module(torch.ones(len(data), 1)).expand_as(data.batch['responses'])
            mask = PolicySafetyRuntime.effective_mask(data.batch)
            loss = (f.dual_clip_terms(data.batch['old_log_probs'], current, data.batch['advantages'], mask)
                    * data.batch['policy_weights']).sum() / len(data)
            actor.actor_optimizer.zero_grad()
            loss.backward()
            grad = actor.actor_module.weight.grad.clone()
            actor.actor_optimizer.step()
            state = actor.actor_optimizer.state[actor.actor_module.weight]
            self.assertEqual(int(state['step']), 1)
            self.assertTrue(torch.isfinite(grad).all())
            results.append((loss.detach(), grad, actor.actor_module.weight.detach().clone(),
                            state['exp_avg'].clone(), state['exp_avg_sq'].clone()))
        for lhs, rhs in zip(*results):
            self.assertTrue(torch.equal(lhs, rhs))
        self.assertNotIn('actor_old_log_probs', omitted.batch)
        self.assertTrue(torch.equal(omitted.batch['old_log_probs'], omitted.batch['sampled_old_log_probs']))
        self.assertEqual(omitted.meta_info['temperature'], 1.)

    def test_sampler_safety_still_runs_and_missing_full_beta_metrics_are_null(self):
        data = self.data(True)
        actor = f.Actor(self.cfg)
        runtime = PolicySafetyRuntime(actor, data.batch, data.meta_info, self.cfg.policy_safety, source_data=data)
        decision = runtime.measure('before_optimizer', publish=False)
        self.assertTrue(decision['allowed'])
        self.assertEqual(decision['probability_denominator'], 'actual_behavior_mu')
        self.assertFalse(decision['full_actor_beta_available'])
        for key in ('global_monitor_mu_full_partition_beta_max_abs_delta',
                    'global_monitor_actor_full_partition_max_abs_delta'):
            self.assertIsNone(decision[key])
        self.assertEqual(decision['global_monitor_actor_repeat_max_abs_delta'], 0.)
        self.assertGreater(runtime.monitor_native_forward_calls, 0)
        strict = f.config(target_old_policy_kl=1e-8)
        strict.update(use_kl_loss=True, kl_loss_coef=0.)
        runtime = PolicySafetyRuntime(f.Actor(strict), data.batch, data.meta_info, strict.policy_safety, source_data=data)
        self.assertFalse(runtime.measure('before_optimizer', publish=False)['allowed'])

    def test_unsupported_or_fabricated_beta_and_nonzero_kl_are_rejected(self):
        for error in ('fabricated', 'nonzero_kl', 'unauthorized', 'missing_mu'):
            data = self.data(True)
            cfg = f.config()
            cfg.update(use_kl_loss=True, kl_loss_coef=0., kl_loss_type='kl')
            actor = f.Actor(cfg)
            if error == 'fabricated':
                data.batch['actor_old_log_probs'] = data.batch['sampled_old_log_probs'].clone()
            if error == 'nonzero_kl':
                actor.config.kl_loss_coef = .1
            if error == 'missing_mu':
                del data.batch['sampled_old_log_probs']
            with self.subTest(error=error):
                if error == 'unauthorized':
                    with patch.dict(os.environ, RECURSIVE_A_OMIT_FULL_BETA='0'), self.assertRaises(PolicySafetyRefusal):
                        validate_actor_probability_profile(actor, data, actor.config.policy_safety)
                else:
                    with self.assertRaises(PolicySafetyRefusal):
                        validate_actor_probability_profile(actor, data, actor.config.policy_safety)
                self.assertEqual(len(actor.calls), 0)


if __name__ == '__main__':
    unittest.main()
