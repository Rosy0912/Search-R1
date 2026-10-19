"""CPU contracts for the five new7B groups and exact allocation release."""
import copy
import json
import unittest
from omegaconf import OmegaConf
from .ablations import (ALLOCATION, COMPILER, DATA, GROUPS, HOST, MODEL, ROOT,
                        VERSIONS, REVISION, derive, job_name, run_root, validate_gate, validate_plan)
from .contract import model_contract, models, make_config
from reproduction.recursive_lumia.prepare_ablation import derive as derive_3b
from release.recursive_bran_a.ablation150_v1.contracts import projected_original, validate_config_dict


def plan(group='budget-5'):
    root = run_root(group)
    return dict(protocol='sevenb_recursive_ablation150_pj_v1', group=group, job=job_name(group),
                root=str(root), output=str(root/'output'), allocation=copy.deepcopy(ALLOCATION),
                model=str(MODEL), model_revision=REVISION,
                model_files={v['file']: v['sha256'] for v in models()['base']['files']},
                data_root=str(DATA), condition=derive(group)['baselines']['recursive_ablation'],
                actual_update_limit=150, scheduler_scale=200, warmup_rounds=57,
                fresh_from_common_initialization=True, parent_checkpoint=None,
                no_auto_retry=True, no_hostname_affinity=True, compiler=copy.deepcopy(COMPILER),
                required_versions=copy.deepcopy(VERSIONS), source_commit='a'*40, run_id='b'*32,
                config_text=OmegaConf.to_yaml(OmegaConf.create(derive(group)), resolve=True))


class Ablations(unittest.TestCase):
    def test_exact_five_current_conditions_and_no_equivalent_duplicate(self):
        self.assertEqual(list(GROUPS), ['C1-root-only', 'C2-random-final', 'budget-5', 'budget-20', 'generation-2'])
        for name in ['budget-10', 'generation-1', 'generation-4', 'position-early', 'position-late', 'original']:
            with self.assertRaises(ValueError): run_root(name)

    def test_all_five_configs_roundtrip_and_stop150_schedule200(self):
        for group in GROUPS:
            p = plan(group); self.assertEqual(validate_plan(p), run_root(group))
            c = OmegaConf.create(p['config_text']); validate_config_dict(OmegaConf.to_container(c, resolve=True))
            self.assertEqual(c.trainer.actual_update_limit, 150)
            self.assertEqual(c.actor_rollout_ref.actor.optim.total_training_steps, 200)
            self.assertEqual(list(c.trainer.evaluation_steps), [5, 50, 100, 150])
            self.assertEqual(c.actor_rollout_ref.model.path, str(MODEL))

    def test_native_projection_uses_already_explicit_sevenb_admission(self):
        for group in GROUPS:
            c = OmegaConf.create(derive(group))
            self.assertFalse(model_contract(c.baselines))
            projected = OmegaConf.create(projected_original(derive(group)))
            self.assertTrue(model_contract(projected.baselines))
            self.assertEqual(projected.baselines.tree.continuation_budget_per_question, 10)

    def test_only_operational_relocation_beyond_original3b_transform(self):
        for group in GROUPS:
            base = run_root(group)
            template = OmegaConf.to_container(make_config(base, MODEL, 'gpu-l40s-2'), resolve=True)
            expected = derive_3b(template, group, base, 'gpu-l40s-2')
            actual = derive(group)
            actual['baselines']['local_runtime_host'] = expected['baselines']['local_runtime_host']
            for key in ['train_files', 'val_files', 'final_val_files']:
                actual['data'][key] = expected['data'][key]
            actual['baselines']['holdout'] = expected['baselines']['holdout']
            actual['baselines']['bapo']['stage_validation_files'] = expected['baselines']['bapo']['stage_validation_files']
            actual['trainer']['experiment_name'] = expected['trainer']['experiment_name']
            self.assertEqual(actual, expected)

    def test_each_scientific_parameter_mutation_rejected(self):
        for key, value in [('actor_rollout_ref.actor.optim.lr', 2e-6), ('data.train_batch_size', 256),
                           ('trainer.actual_update_limit', 151), ('max_turns', 5),
                           ('actor_rollout_ref.rollout.temperature', .95), ('baselines.recursive_ablation.budget', 20)]:
            p = plan(); c = OmegaConf.create(p['config_text']); OmegaConf.update(c, key, value)
            p['config_text'] = OmegaConf.to_yaml(c, resolve=True)
            with self.assertRaises(AssertionError): validate_plan(p)

    def test_wrong_resource_or_model_or_resume_rejected(self):
        for key, value in [('allocation', dict(ALLOCATION, memory_gib=600)), ('parent_checkpoint', '/old/partial'),
                           ('model_revision', 'main'), ('root', '/tmp/escape'), ('no_auto_retry', False),
                           ('model_files', {}), ('no_hostname_affinity', False)]:
            p = plan(); p[key] = value
            with self.assertRaises(AssertionError): validate_plan(p)

    def test_exact_uid_allocation_gate(self):
        p = plan(); sub = dict(job=p['job'], uid='exact-uid', plan_sha256='p')
        gate = dict(plan_sha256='p', actual=dict(name=p['job'], uid='exact-uid', state='RUNNING', resources=copy.deepcopy(ALLOCATION)))
        self.assertEqual(validate_gate(p, sub, gate)['uid'], 'exact-uid')
        for field, value in [('uid', 'wrong'), ('state', 'PENDING'), ('resources', dict(ALLOCATION, gpus=4)), ('name', 'other')]:
            bad = copy.deepcopy(gate); bad['actual'][field] = value
            with self.assertRaises(AssertionError): validate_gate(p, sub, bad)

    def test_unique_bounded_names_and_group_separation(self):
        self.assertEqual(len({job_name(g) for g in GROUPS}), 5)
        self.assertTrue(all(len(job_name(g)) <= 32 for g in GROUPS))
        self.assertEqual(len({str(run_root(g)) for g in GROUPS}), 5)
        self.assertTrue(all(str(run_root(g)).startswith(str(ROOT)+'/runs/') for g in GROUPS))


if __name__ == '__main__': unittest.main()
