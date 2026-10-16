import copy
import unittest

from release.recursive_bran_a.ablation150_v1.contracts import planned_conditions
from release.recursive_bran_a.ablation150_v1.test_runner import minimal_config
from .prepare_ablation import condition_for, derive, validate_delta


class MatchedAblationConfig(unittest.TestCase):
    def baseline(self):
        value = minimal_config()
        del value['baselines']['recursive_ablation']
        value['baselines']['local_runtime_host'] = 'gpu-l40s-2'
        value['trainer'].update(actual_update_limit=200,total_training_steps=201,
            evaluation_steps=[5,50,100,150,200],checkpoint_save_steps=[5,50,100,150,200],
            experiment_name='original',default_local_dir='/original/checkpoints')
        value['actor_rollout_ref']['actor'].update(ppo_mini_batch_size=128,
            ppo_micro_batch_size=16,ppo_epochs=1,clip_ratio=.2,grad_clip=1)
        value['actor_rollout_ref']['actor']['optim'].update(lr=1e-6,lr_warmup_steps=57)
        value['preserve_unknown_future_field'] = {'behavior_mu':'original','precision':'bf16_fp32'}
        return value

    def test_all_seven_new_conditions_preserve_scientific_settings(self):
        baseline = self.baseline()
        before = copy.deepcopy(baseline)
        groups = [r['group'] for r in planned_conditions() if not r.get('reuse')]
        self.assertEqual(len(groups),7)
        for group in groups:
            result = derive(baseline,group,'/new','gpu-l40s-1')
            self.assertEqual(result['actor_rollout_ref'],baseline['actor_rollout_ref'])
            self.assertEqual(result['preserve_unknown_future_field'],baseline['preserve_unknown_future_field'])
            self.assertEqual(result['trainer']['evaluation_steps'],[5,50,100,150])
            self.assertEqual(result['trainer']['actual_update_limit'],150)
        self.assertEqual(baseline,before)

    def test_drift_in_objective_or_microbatch_is_rejected(self):
        baseline = self.baseline()
        for field,value in [('ppo_micro_batch_size',8),('clip_ratio',.3)]:
            result = derive(baseline,'C1-root-only','/new','gpu-l40s-1')
            result['actor_rollout_ref']['actor'][field] = value
            with self.assertRaises(ValueError):
                validate_delta(baseline,result,'C1-root-only','/new','gpu-l40s-1')

    def test_root_anchored_is_explicit_and_preserves_training_contract(self):
        baseline = self.baseline()
        group = 'R1-root-anchored-recursive'
        candidate = derive(baseline, group, '/r1', 'gpu-l40s-1')
        self.assertEqual(candidate['actor_rollout_ref'], baseline['actor_rollout_ref'])
        self.assertEqual(candidate['baselines']['tree']['continuation_budget_per_question'], 10)
        self.assertEqual(candidate['trainer']['actual_update_limit'], 150)
        condition = candidate['baselines']['recursive_ablation']
        self.assertEqual(condition['variant'], 'root_anchored_recursive')
        self.assertEqual(condition['acceptance_reference'], 'initial_root')
        self.assertEqual(condition['branch_boundary'], 'all_preaction_snapshots')
        self.assertEqual(condition['parent_sampling'], 'uniform_accepted')
        self.assertNotIn(group, [row['group'] for row in planned_conditions()])
        validate_delta(baseline, candidate, group, '/r1', 'gpu-l40s-1')
        for key in ('acceptance_reference', 'branch_boundary', 'parent_sampling'):
            changed = copy.deepcopy(candidate)
            changed['baselines']['recursive_ablation'][key] = 'unregistered'
            with self.assertRaises(ValueError):
                validate_delta(baseline, changed, group, '/r1', 'gpu-l40s-1')

    def test_reused_conditions_cannot_be_prepared_as_new_training(self):
        for row in planned_conditions():
            if row.get('reuse'):
                with self.assertRaises(ValueError): condition_for(row['group'])

    def test_wrong_baseline_and_host_rejected(self):
        baseline = self.baseline()
        with self.assertRaises(ValueError): derive(baseline,'C1-root-only','/new','unknown-host')
        baseline['baselines']['question_curriculum'] = {'enabled':True}
        with self.assertRaises(ValueError): derive(baseline,'C1-root-only','/new','gpu-l40s-1')


if __name__ == '__main__':
    unittest.main()
