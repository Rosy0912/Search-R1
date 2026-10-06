"""New protocol tests; not run until the separate static review approves execution."""
import copy
import pickle
import unittest
from omegaconf import OmegaConf

from search_r1.baselines.fresh200 import (CALIBRATION_SCOPE, CHECKPOINT_STEPS, GOAL_UPDATED_AT,
    calibration_initialization, driver_flow_state, restore_driver_flow, fixed_monitor_layout, select_fixed_monitor, stop_step, validate_calibration_initialization,
    validate_initial_workers, validate_protocol)
from search_r1.baselines.policy_safety import (PolicySafetyRejected, calibrated_or_candidate_rejection,
                                             matched_policy_rejection)


def config():
    return OmegaConf.create(dict(
        actor_token_identity='sampled_action_prefix_v1',
        data=dict(train_batch_size=512),
        algorithm=dict(kl_ctrl=dict(kl_coef=0)),
        actor_rollout_ref=dict(actor=dict(ppo_mini_batch_size=128, ppo_micro_batch_size=16, ppo_epochs=1,
            optim=dict(lr=1e-6, lr_warmup_steps=57, lr_warmup_steps_ratio=0, total_training_steps=200),
            entropy_coeff=0, use_kl_loss=True, kl_loss_coef=0, kl_loss_type='low_var_kl', clip_ratio=.2,
            grad_clip=1, fresh200_protocol=True, preserve_partial_policy_safety_state=True,
            joint_loss_scaling=True, actor_token_identity='sampled_action_prefix_v1',
            policy_safety=dict(calibration_scope=CALIBRATION_SCOPE, enabled=True, dual_clip=3.,
                kl_log_bound=30., monitor_first_global_rows=256)),
            rollout=dict(n_agent=5, n=1, tensor_model_parallel_size=1, log_prob_micro_batch_size=32,
                actor_token_identity='sampled_action_prefix_v1'),
            ref=dict(log_prob_micro_batch_size=32, actor_token_identity='sampled_action_prefix_v1')),
        trainer=dict(total_training_steps=200, acceptance_step=5, save_freq=50, test_freq=50,
            val_before_train=True, instruct300=dict(enabled=True), resume=dict(enabled=False, source=None),
            trajectory_logging=True, n_gpus_per_node=8, nnodes=1,
            fresh200=dict(enabled=True, goal_updated_at=GOAL_UPDATED_AT, initial_outer_step=0,
                initial_adam_steps=0, initial_scheduler_step=0, initial_data_cursor=0,
                checkpoint_steps=list(CHECKPOINT_STEPS), preserve_partial_updates=True))))


class Protocol(unittest.TestCase):
    def test_current_goal_and_normal_accounting(self):
        value = config()
        validate_protocol(value)
        self.assertEqual(stop_step(value), 200)
        self.assertEqual(512 * 5 // value.actor_rollout_ref.actor.ppo_mini_batch_size, 20)
        self.assertEqual(2560 // 8 // (16 // 8), 160)

    def test_superseded_restore300_mini256_and_reference_kl_rejected(self):
        cases = [('trainer.total_training_steps', 300), ('actor_rollout_ref.actor.ppo_mini_batch_size', 256),
            ('trainer.instruct300.restore_checkpoint', '/old/step25'),
            ('trainer.instruct300.restore_manifest_sha256', 'a' * 64),
            ('trainer.quality_resume.enabled', True), ('trainer.resume.source', '/old/run'),
            ('actor_rollout_ref.actor.kl_loss_coef', .001), ('algorithm.kl_ctrl.kl_coef', .001),
            ('actor_rollout_ref.actor.optim.lr_warmup_steps', 0),
            ('actor_rollout_ref.actor.optim.initial_logical_step', 25),
            ('trainer.fresh200.initial_adam_steps', False)]
        for key, value in cases:
            with self.subTest(key=key):
                bad = config()
                OmegaConf.update(bad, key, value, force_add=True)
                with self.assertRaises(ValueError): validate_protocol(bad)

    def test_raw_actor_reference_rollout_identities_must_agree(self):
        bad = config()
        bad.actor_rollout_ref.ref.actor_token_identity = 'legacy_canonical'
        with self.assertRaises(ValueError): validate_protocol(bad)

    def test_legacy_target_remains300_when_explicit_fresh200_disabled(self):
        value = config()
        value.trainer.fresh200.enabled = False
        self.assertEqual(stop_step(value), 300)

    def test_calibration_initialization_rejects_old_restore_and_adam(self):
        value = calibration_initialization()
        validate_calibration_initialization(value)
        for key, wrong in [('restore_checkpoint', '/old/step25'), ('initial_adam_steps', 500),
                ('ppo_mini_batch_size', 256), ('reference_kl_coefficient', .001),
                ('initial_data_cursor', False)]:
            with self.subTest(key=key):
                bad = dict(value, **{key: wrong})
                with self.assertRaises(ValueError): validate_calibration_initialization(bad)

    def test_actual_worker_zero_state_and_fixed_weight_identity(self):
        actors, refs = [], []
        for rank in range(8):
            common = dict(rank=rank, pid=100 + rank, start_ticks=1000 + rank,
                official_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
                origin_kind='OFFICIAL_MODEL_INITIALIZATION', model_path='/actual/official',
                official_bf16_parameter_sha256=str(rank) * 64)
            refs.append(dict(common, role='reference'))
            actors.append(dict(common, role='actor', optimizer_state_entries=6, optimizer_populated_slots=0, scheduler_last_epoch=0,
                scheduler_step_count=1, learning_rates=[0.], local_mini_batch_size=16,
                local_micro_batch_size=2, reference_kl_coefficient=0.))
        validate_initial_workers(actors, refs)
        for key, wrong in [('optimizer_populated_slots', 1), ('scheduler_last_epoch', 25),
                ('official_bf16_parameter_sha256', 'bad')]:
            bad = copy.deepcopy(actors)
            bad[5][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError): validate_initial_workers(bad, refs)

    def test_candidate_kl_is_diagnostic_and_tail_unset(self):
        stats = dict(rejected_numeric_count=0, weight_mass=1, sampled_old_policy_kl=.02,
            negative_tail_fraction_count=.9)
        control = dict(mode='calibration', calibration_scope=CALIBRATION_SCOPE, candidate_old_policy_kl=.01)
        self.assertEqual(calibrated_or_candidate_rejection(stats, control), 'CALIBRATION_CANDIDATE_OLD_POLICY_TOKEN_KL')
        stats['sampled_old_policy_kl'] = .009
        self.assertIsNone(calibrated_or_candidate_rejection(stats, control))

    def test_typed_rejection_roundtrip_retains_partial_writes(self):
        value = PolicySafetyRejected('guard', dict(actual_optimizer_writes=7, outer_step=3))
        restored = pickle.loads(pickle.dumps(value))
        self.assertEqual(restored.record, value.record)
        self.assertEqual(str(restored), str(value))

    def test_unknown_message_is_not_a_matched_allrank_refusal(self):
        self.assertIsNone(matched_policy_rejection(RuntimeError('POLICY_SAFETY_REJECTED: unknown text')))
        cause = PolicySafetyRejected('matched test fixture', dict(actual_optimizer_writes=7))
        wrapper = RuntimeError('Ray fixture')
        wrapper.cause = cause
        self.assertIs(matched_policy_rejection(wrapper), cause)

    def test_real_ppo_splitter_twenty_minis_monitor_complete_256_rows(self):
        import torch
        from tensordict import TensorDict
        ids = torch.arange(320).view(320, 1)
        keys = ['responses', 'old_log_probs', 'advantages', 'policy_weights', 'step_ids',
            'actor_action_prefix_ids', 'actor_action_tokens', 'actor_action_lengths',
            'actor_action_prefix_lengths', 'actor_action_output_offsets', 'actor_action_count']
        batch = TensorDict({key: ids.clone() for key in keys}, batch_size=[320])
        monitor, expected_minis = select_fixed_monitor(batch, 16, 8, 256)
        minis = batch.split(16)
        self.assertEqual(len(minis), expected_minis)
        self.assertEqual(expected_minis, 20)
        self.assertEqual(len(monitor), 32)
        self.assertEqual(len(monitor)*8, 256)
        self.assertTrue(torch.equal(torch.cat([mini['responses'] for mini in minis]), ids))
        for key in keys:
            self.assertTrue(torch.equal(monitor[key], ids[:32]))
        monitor['responses'].fill_(-1)
        self.assertTrue(torch.equal(batch['responses'], ids))
        # The older256 mini path retains its10 steps but the fixed monitor is
        # the same complete trajectories, not a mini-dependent selection.
        old_monitor, old_minis = select_fixed_monitor(batch, 32, 8, 256)
        self.assertEqual(old_minis, 10)
        self.assertTrue(torch.equal(old_monitor['responses'], ids[:32]))

    def test_incomplete_rollout_or_nondivisible_monitor_rejected(self):
        for args in ((319, 16, 8, 256), (320, 17, 8, 256), (320, 16, 8, 255)):
            with self.subTest(args=args), self.assertRaises(ValueError): fixed_monitor_layout(*args)

    def test_actual_single_process_iterator_reload_does_not_fetch_next_row_or_draw_rng(self):
        import torch
        from types import SimpleNamespace
        from search_r1.baselines.recovery import RecoverableSampler
        sampler = RecoverableSampler(2048)
        loader = torch.utils.data.DataLoader(list(range(2048)), batch_size=512,
            sampler=sampler, num_workers=0, drop_last=True)
        iterator = iter(loader)
        first = next(iterator)
        trainer = SimpleNamespace(fresh200_active_train_iterator=iterator,
            recovery_sampler=sampler, validation_trace_serial=3)
        saved = driver_flow_state(trainer)
        self.assertEqual(saved['iterator_fields']['_num_yielded'], 1)
        self.assertEqual(saved['data_cursor'], 512)
        before_rng = torch.get_rng_state().clone()
        receipt = restore_driver_flow(trainer, saved)
        self.assertTrue(receipt['same_live_iterator'])
        self.assertEqual(driver_flow_state(trainer), saved)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertEqual(sampler.cursor, 512)
        self.assertIs(trainer.fresh200_active_train_iterator, iterator)
        second = next(iterator)
        self.assertEqual(sampler.cursor, 1024)
        self.assertEqual(len(set(first.tolist()) & set(second.tolist())), 0)
        trainer.validation_trace_serial = 4
        with self.assertRaises(ValueError): restore_driver_flow(trainer, saved)


if __name__ == '__main__': unittest.main()
