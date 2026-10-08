"""Actual CPU optimizer checks and pure provenance rejection tests; not eight GPUs."""
import copy
import unittest

import torch

from reproduction.common8.formal_contract import FORMAL_COUNTING, FORMAL_SAFEGUARDS, SAFETY_MODE, consumed_lr
from reproduction.common8.formal_inline_safety import validate_optimizer_state, verify_method
from reproduction.common8.policy_calibration_plan import document_sha256
from search_g1.policy_safety import monitor


def guarded_opportunity_fixture():
    """Actual CPU statistics, with fabricated eight-rank records for pure checking."""
    import hashlib
    from reproduction.common8.safety_calibration_evidence import MONITOR_TENSORS, MONITORING_SCOPE, OBJECTIVE_WEIGHT_DEFINITION
    def sha(value): return hashlib.sha256(value.encode()).hexdigest()
    attempt_id = 'a' * 32; gate = sha('gate'); plan = sha('plan')
    mask = torch.zeros((32, 5), dtype=torch.float64); weights = torch.zeros_like(mask)
    advantages = torch.ones_like(mask)
    for row in range(32):
        length = [2, 3, 4, 5][row % 4]; mask[row, :length] = 1
        pair_length = 5 if row % 4 < 2 else 9
        weights[row, :length] = 1. / pair_length
        if row % 4 in (0, 2): advantages[row] = -1
    reports = [monitor(torch.full_like(mask, delta), advantages, mask, weights,
        c=3., kl_log_bound=30., target_kl=.2, negative_tail_limit=.01) for delta in (0., 1.)]
    actual = dict(step=1, run_id='component', attempt_id=attempt_id, attempt_index=1,
        optimizer_writes_before=0, optimizer_writes_after=1, opportunities_processed=20,
        behavior_probability_gate_sha256=gate)
    binding = dict(schema_version=1, role='production', policy_safety_mode=SAFETY_MODE,
        run_id='component', calibration_sha256=plan, outer_count=1, initial_optimizer_writes=0,
        candidate_thresholds=FORMAL_SAFEGUARDS, complete_attempts=[actual])
    ranks = {}
    for rank in range(8):
        rows = []; count = 0
        identity = dict(selection_indices=[rank, 106 + rank, 213 + rank, 312 + rank],
            actor_token_lengths=[2, 3, 4, 5], tensor_sha256={name: sha(name + str(rank)) for name in MONITOR_TENSORS})
        for mini in range(20):
            prior = count; written = mini == 0
            context = dict(logical_update=1, attempt_id=attempt_id, attempt_index=1,
                minibatch_index=mini, prior_optimizer_writes=prior, behavior_probability_gate_sha256=gate)
            before, after = reports[int(mini > 0)], reports[1]
            events = []
            for stage, stats, writes in (('before_minibatch', before, prior),
                                         ('after_minibatch', after, prior + int(written))):
                if stage == 'after_minibatch' and written:
                    rows.append(dict(kind='OPTIMIZER_WRITE', run_id='component', rank=rank, context=copy.deepcopy(context),
                        write_count=1, gradient_norm_before=2., gradient_norm_after=1., learning_rates=[0.]))
                rows.append(dict(kind='POLICY_MONITOR', run_id='component', rank=rank, context=copy.deepcopy(context),
                    stage=stage, optimizer_writes=writes, calibration_sha256=plan,
                    monitor_identity=copy.deepcopy(identity), rejection_count=mini + int(not stats['allowed']),
                    objective_weight_definition=OBJECTIVE_WEIGHT_DEFINITION, monitoring_scope=MONITORING_SCOPE,
                    **copy.deepcopy(stats)))
                if not stats['allowed']: events.append(dict(stage=stage, reasons=stats['reasons']))
            count += int(written)
            rows.append(dict(kind='OPPORTUNITY', run_id='component', rank=rank, context=copy.deepcopy(context),
                opportunity_index=mini, processed=True,
                status='written_with_post_guard' if written else 'skipped_guard',
                optimizer_write_succeeded=written, optimizer_writes_before=prior, optimizer_writes_after=count,
                guard_stage=events[0]['stage'], guard_reasons=events[0]['reasons'], guard_events=copy.deepcopy(events)))
        rows.append(dict(kind='OUTER_BUDGET', run_id='component', rank=rank, logical_update=1,
            attempt_id=attempt_id, attempt_index=1, local_trajectories=320, local_mini=16, local_micro=2,
            opportunities_processed=20, optimizer_writes=1, optimizer_writes_before=0, optimizer_writes_after=1,
            cumulative_writes=1))
        ranks[rank] = rows
    return ranks, binding


class InlineActualOptimizerTests(unittest.TestCase):
    def test_skipped_opportunities_leave_actual_adam_counter_and_scheduler_lr_distinct(self):
        parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32))
        optimizer = torch.optim.Adam([parameter], lr=consumed_lr(1))
        validate_optimizer_state(optimizer, 1, 0, complete=False)
        opportunities = 20; actual_writes = 0
        for opportunity in range(opportunities):
            # Actual skip decisions exercise counts; this is no physical training claim.
            if opportunity in (1, 4, 9):
                optimizer.zero_grad(); continue
            parameter.grad = torch.ones_like(parameter); optimizer.step(); actual_writes += 1
        self.assertEqual(actual_writes, 17)
        self.assertEqual(parameter.item(), 1.)  # consumed LR0 still creates real Adam steps
        self.assertEqual(optimizer.state[parameter]['step'].item(), 17)
        validate_optimizer_state(optimizer, 1, actual_writes, complete=True)
        optimizer.param_groups[0]['lr'] = consumed_lr(2)
        validate_optimizer_state(optimizer, 2, 17, complete=False)
        parameter.grad = torch.ones_like(parameter); optimizer.step()
        self.assertEqual(optimizer.state[parameter]['step'].item(), 18)
        self.assertLess(parameter.item(), 1.)
        validate_optimizer_state(optimizer, 2, 18, complete=True)

    def test_state_steps_and_wrong_lr_reject_without_mutation(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.Adam([parameter], lr=consumed_lr(1))
        parameter.grad = torch.ones_like(parameter); optimizer.step()
        before = parameter.detach().clone()
        with self.assertRaises(AssertionError):
            validate_optimizer_state(optimizer, 1, 0, complete=False)
        optimizer.param_groups[0]['lr'] = 1e-6
        with self.assertRaises(AssertionError):
            validate_optimizer_state(optimizer, 1, 1, complete=True)
        self.assertTrue(torch.equal(parameter, before))

    def test_sampling_kl_point2_brake_keeps_reference_regularization_independent(self):
        kwargs = dict(c=3., kl_log_bound=30., target_kl=.2, negative_tail_limit=.01,
                      execution_role='production')
        advantage = torch.ones((2, 3), dtype=torch.float32)
        mask = torch.ones_like(advantage)
        weights = torch.ones_like(advantage)
        admitted = monitor(torch.full_like(advantage, .4), advantage, mask, weights, **kwargs)
        rejected = monitor(torch.full_like(advantage, 1.), advantage, mask, weights, **kwargs)
        self.assertTrue(admitted['allowed'])
        self.assertGreater(admitted['sampled_old_policy_kl'], .01)
        self.assertFalse(rejected['allowed'])
        self.assertEqual(rejected['reasons'], ['old_policy_kl'])
        self.assertTrue(torch.equal(advantage, torch.ones_like(advantage)))


class InlineProspectiveMethodTests(unittest.TestCase):
    def fixture(self):
        run = dict(source_commit='a' * 40, source_sha256={'one.py': 'b' * 64}, started_unix=20.)
        method = dict(schema_version=1, role='formal_inline_training_method',
            source_commit=run['source_commit'], source_map_sha256=document_sha256(run['source_sha256']),
            created_unix=10., policy_safety_mode=SAFETY_MODE, target_updates=200, warmup_steps=57,
            same_process_after_five=True, fresh_states=True, optimizer_counting=FORMAL_COUNTING,
            safeguards=copy.deepcopy(FORMAL_SAFEGUARDS),
            negative_tail_semantics='prospective operational candidate; not calibrated health')
        return run, method

    def test_exact_prospective_point2_and_opportunity_counting(self):
        run, method = self.fixture(); verify_method(method, run)
        self.assertEqual(method['safeguards']['target_old_policy_kl'], .2)
        self.assertEqual(method['optimizer_counting'], 'all20_opportunities_processed_and_at_least_one_write')

    def test_old_threshold_retroactive_method_state_resume_and_wrong_source_reject(self):
        for key, value in (('created_unix', 21.), ('source_commit', 'c' * 40),
                           ('fresh_states', False), ('optimizer_counting', 'twenty_writes')):
            run, method = self.fixture(); method[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError): verify_method(method, run)
        run, method = self.fixture(); method['safeguards']['target_old_policy_kl'] = .01
        with self.assertRaises(AssertionError): verify_method(method, run)


class FormalOpportunityEvidenceTests(unittest.TestCase):
    def test_actual_cpu_monitor_brake_one_write_and_nineteen_skips_count_one_outer(self):
        from reproduction.common8.safety_calibration_evidence import verify_formal_segment
        rows, binding = guarded_opportunity_fixture()
        proof = verify_formal_segment(rows, binding)
        self.assertEqual(proof['complete_outer_count'], 1)
        self.assertEqual(proof['processed_opportunities_per_rank'], 20)
        self.assertEqual(proof['completed_outer_actual_writes_per_rank'], 1)
        self.assertEqual(len(proof['global_monitors']), 40)
        self.assertEqual(proof['guard_triggered_monitor_count'], 39)
        self.assertFalse(proof['production_admission'])

    def test_missing_opportunity_fake_write_rejected_prewrite_or_divergent_rank_reject(self):
        from reproduction.common8.safety_calibration_evidence import EvidenceError, verify_formal_segment
        for mutation in ('missing', 'fake_write', 'divergent', 'pre_write'):
            rows, binding = guarded_opportunity_fixture()
            if mutation == 'missing': rows[7].pop(3)
            elif mutation == 'fake_write': rows[7][3]['optimizer_write_succeeded'] = False
            elif mutation == 'divergent': rows[7][3]['guard_events'] = []
            else:
                rows[7][0].update(allowed=False, reasons=['old_policy_kl'])
            with self.subTest(mutation=mutation), self.assertRaises((EvidenceError, KeyError)):
                verify_formal_segment(rows, binding)


if __name__ == '__main__':
    unittest.main()
