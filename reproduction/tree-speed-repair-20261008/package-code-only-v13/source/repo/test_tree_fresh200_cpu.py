"""Affected fresh experiment contracts on real CPU tensors; no GPU acceptance."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace as N
import unittest
from unittest.mock import patch
import uuid

import torch
from search_r1.baselines.rollout import SearchRollout
from search_r1.baselines.tree_audit import TreeAudit
from search_r1.baselines.tree_fresh200 import (PROTOCOL, REVISION, DATA_SHA256,
    leaf_quotas, validate_fresh200_config, refuse_recovery_environment, validate_completed_outer,
    validate_fresh_sampler)
from test_tree_raw_tokens_cpu import TinyProto


class Node(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def node(value):
    return Node({k: node(v) if isinstance(v, dict) else v for k, v in value.items()})


def config():
    return node(dict(baselines=dict(training_protocol=PROTOCOL, method='tree_grpo', model_revision=REVISION,
        model_repository='Qwen/Qwen2.5-3B-Instruct', tree=dict(roots=2, rounds=1, expansions=2,
        leaves=3, leaf_quotas=[3, 2], token_preservation=dict(enabled=True), behavior_mu=dict(enabled=True)),
        holdout=dict(input_mode='prebuilt_manifest', manifest_sha256=DATA_SHA256)),
        actor_rollout_ref=dict(model=dict(path='official-same-local-path'), ref=dict(model_path='official-same-local-path',
        log_prob_micro_batch_size=32), rollout=dict(n_agent=2, n=1, best_of=1, log_prob_micro_batch_size=32, tree_raw_emission=True),
        actor=dict(ppo_mini_batch_size=128, ppo_micro_batch_size=16, ppo_epochs=1,
        ulysses_sequence_parallel_size=1, use_dynamic_bsz=False, fresh200_protocol=PROTOCOL, guard_continue_protocol="tree_guard_continue_fresh_rollout_v1",
        optim=dict(lr=1e-6, lr_warmup_steps=57, lr_warmup_steps_ratio=0, total_training_steps=200),
        entropy_coeff=0, kl_loss_coef=0, kl_loss_type='low_var_kl', clip_ratio=.2, grad_clip=1,
        policy_safety=dict(enabled=True, dual_clip=3., target_old_policy_kl=.2, sampling_probability_correction=dict(enabled=True,
        protocol='tree_behavior_mu_centered_ppo_v1')))),
        data=dict(train_batch_size=512, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500),
        algorithm=dict(kl_ctrl=dict(kl_coef=0)), trainer=dict(n_gpus_per_node=8, nnodes=1,
        total_training_steps=201, actual_update_limit=200, save_freq=50, acceptance_step=5,
        acceptance_gate_enabled=True, test_freq=5), max_turns=4, force_final_answer=False))


class CompleteEngine(SearchRollout):
    """Only model generation is replaced; production tree run/compose/audit execute."""
    def complete(self, states, meta):
        snapshots = [[] for _ in states]
        for i, state in enumerate(states):
            while state.turns < 2:
                snapshots[i].append(copy.deepcopy(state))
                step = state.turns
                state.response += [9]
                state.model_mask += [1]
                state.steps += [step]
                state.anchors += [str(step)]
                state.records += [dict(action_tokens=1, observation_tokens=0)]
                state.turns += 1
                state.rolling = state.prompt + state.response
                state.node = uuid.uuid4().hex
            state.done = True
        self.cost['trajectories'] += len(states)
        return snapshots


class Fresh200Tests(unittest.TestCase):
    def test_fresh_contract_and_exact_twenty_calls(self):
        c = config(); validate_fresh200_config(c)
        local_rows = c.data.train_batch_size * sum(leaf_quotas(c.baselines.tree)) // 8
        self.assertEqual((local_rows, c.actor_rollout_ref.actor.ppo_mini_batch_size // 8), (320, 16))
        self.assertEqual(local_rows // (c.actor_rollout_ref.actor.ppo_mini_batch_size // 8), 20)

    def test_drifted_model_split_update_or_budget_refuses(self):
        changes = [('revision', 'wrong'), ('split', 'wrong'), ('micro', 64), ('mini', 384),
                   ('kl', .001), ('warmup', 56), ('iterations', 300), ('leaves', [3, 3])]
        for field, value in changes:
            c = config()
            if field == 'revision': c.baselines.model_revision = value
            elif field == 'split': c.baselines.holdout.manifest_sha256 = value
            elif field == 'micro': c.actor_rollout_ref.actor.ppo_micro_batch_size = value
            elif field == 'mini': c.actor_rollout_ref.actor.ppo_mini_batch_size = value
            elif field == 'kl': c.actor_rollout_ref.actor.kl_loss_coef = value
            elif field == 'warmup': c.actor_rollout_ref.actor.optim.lr_warmup_steps = value
            elif field == 'iterations': c.trainer.actual_update_limit = value
            else: c.baselines.tree.leaf_quotas = value
            with self.subTest(field=field), self.assertRaises(ValueError): validate_fresh200_config(c)

    def test_recovery_environment_cannot_override_fresh(self):
        refuse_recovery_environment({'TREE_INSTRUCT_MODEL': 'local-official'})
        for name in ['TREE_RECOVERY_MANIFEST', 'TREE_RECOVERY_PARENT_OPTIMIZER_STEPS']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                refuse_recovery_environment({name: 'old-value'})

    def test_fresh_sampling_cannot_select_best_of_multiple_or_guess_default(self):
        for value in (None, 2, True, '1'):
            c = config(); c.actor_rollout_ref.rollout.best_of = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_fresh200_config(c)
        c = config(); del c.actor_rollout_ref.rollout['best_of']
        with self.assertRaises(ValueError): validate_fresh200_config(c)

    def test_actual_checkpoint_sampler_starts_uniterated_and_refuses_consumed_state(self):
        from search_r1.baselines.training_state import CheckpointBatchSampler
        sampler = CheckpointBatchSampler(169028, 512, 42, DATA_SHA256, True)
        self.assertEqual((sampler.epoch, sampler.cursor, sampler.order), (-1, 0, None))
        validate_fresh_sampler(sampler)
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 512)
        self.assertEqual((sampler.epoch, sampler.cursor), (0, 512))
        with self.assertRaises(RuntimeError): validate_fresh_sampler(sampler)

    def test_fresh200_cannot_fall_back_to_uncorrected_or_unguarded_probabilities(self):
        c = config(); validate_fresh200_config(c)
        paths = [('collector', c.baselines.tree.behavior_mu), ('raw', c.baselines.tree.token_preservation),
                 ('actor', c.actor_rollout_ref.actor.policy_safety.sampling_probability_correction),
                 ('guard', c.actor_rollout_ref.actor.policy_safety)]
        for name, target in paths:
            target.enabled = False
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'without an uncorrected fallback'):
                validate_fresh200_config(c)
            target.enabled = True

    def test_partial_outer_does_not_receive_credit(self):
        m = {'actor/optimizer_steps': 20, 'actor/skipped_empty_minibatches': 0,
             'actor/scheduler_step_before_outer': 0, 'actor/scheduler_step_after_outer': 1}
        validate_completed_outer(m, 1)
        for key, value in [('actor/optimizer_steps', 19), ('actor/skipped_empty_minibatches', 1),
                           ('actor/scheduler_step_after_outer', 0)]:
            changed = dict(m, **{key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError): validate_completed_outer(changed, 1)

    def test_actual_tree_run_generates_five_with_prefix_inheritance_and_audit(self):
        manager = N(tokenizer=N(pad_token_id=151643), config=N(max_start_length=2048),
                    tensor_fn=N(create_position_ids=lambda m: (m.cumsum(-1)-1)*m))
        engine = CompleteEngine(manager); engine.output_meta = {}
        engine.rng = N(choice=lambda nodes: nodes[-1], sample=lambda leaves, n: list(reversed(leaves))[:n])
        batch = TinyProto(dict(input_ids=torch.tensor([[8], [8], [10], [10]]),
                               attention_mask=torch.ones(4, 1, dtype=torch.long)))
        fake = ModuleType('verl'); fake.DataProto = TinyProto
        with tempfile.TemporaryDirectory() as temp, patch.dict('sys.modules', {'verl': fake}):
            engine.audit = TreeAudit(Path(temp)/'audit', temp, 1, 1, ['q0', 'q0', 'q1', 'q1'])
            result, sources = engine.run(batch, config().baselines.tree)
            self.assertEqual(len(result), 10)
            self.assertEqual(engine.cost['trajectories'], 10)
            self.assertEqual([sources.tolist().count(i) for i in range(4)], [3, 2, 3, 2])
            report = json.loads(engine.audit.path.read_text())
            self.assertEqual(report['branch_count'], 6)
            self.assertEqual(report['root_leaf_quotas'], [3, 2])
            self.assertEqual(report['relative_root_masses'], [.6, .4])
            self.assertTrue(all(x['inherited'] and x['prefix_length'] == 2 for x in report['branches']))
            self.assertTrue(all(x['leaves'] == 5 for x in report['question_groups']))
            self.assertTrue(all(x['relative_leaf_mass'] == .2 for x in report['leaves']))

    def test_legacy_rectangular_quotas_remain_unchanged(self):
        tree = node(dict(roots=2, leaves=3))
        self.assertEqual(leaf_quotas(tree), (3, 3))
        self.assertEqual(leaf_quotas(N(roots=2, leaves=3)), (3, 3))
        for value in [[3], [3, 0], [True, 2], [2, 2]]:
            tree.leaf_quotas = value
            with self.subTest(value=value), self.assertRaises(ValueError): leaf_quotas(tree)

    def test_actual_attribute_tree_preserves_exact_five_leaf_budget(self):
        manager = N(tokenizer=N(pad_token_id=151643), config=N(max_start_length=2048),
                    tensor_fn=N(create_position_ids=lambda m: (m.cumsum(-1)-1)*m))
        engine = CompleteEngine(manager); engine.output_meta = {}
        tree = N(roots=2, rounds=1, expansions=2, leaves=3, leaf_quotas=[3, 2])
        batch = TinyProto(dict(input_ids=torch.tensor([[8], [8]]),
                               attention_mask=torch.ones(2, 1, dtype=torch.long)))
        fake = ModuleType('verl'); fake.DataProto = TinyProto
        with patch.dict('sys.modules', {'verl': fake}):
            output, sources = engine.run(batch, tree)
        self.assertEqual(len(output), 5)
        self.assertEqual(engine.cost['trajectories'], 5)
        self.assertEqual([sources.tolist().count(i) for i in range(2)], [3, 2])

    def test_actual_constant_scheduler_57_outer_warmup(self):
        source = Path(__file__).parent/'verl/utils/torch_functional.py'
        tree = ast.parse(source.read_text())
        fn = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'get_constant_schedule_with_warmup']
        from torch.optim.lr_scheduler import LambdaLR
        scope = {'LambdaLR': LambdaLR, 'Optimizer': torch.optim.Optimizer}
        exec(compile(ast.Module(body=fn, type_ignores=[]), str(source), 'exec'), scope)
        p = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.Adam([p], lr=1e-6)
        scheduler = scope['get_constant_schedule_with_warmup'](optimizer, 57)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.)
        for outer in range(1, 61):
            for _ in range(20):
                optimizer.zero_grad(); p.sum().backward(); optimizer.step()
            scheduler.step()
            self.assertEqual(scheduler.last_epoch, outer)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-6 * min(outer/57, 1), places=16)
        self.assertEqual(int(optimizer.state[p]['step']), 1200)


if __name__ == '__main__': unittest.main()
