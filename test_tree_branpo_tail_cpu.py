"""Finite tail sampling and population diagnostics on real CPU tensors."""
import copy
import math
from types import SimpleNamespace as N, ModuleType
import unittest
from unittest.mock import patch
import torch
from search_r1.baselines.rollout import SearchRollout
from search_r1.baselines.advantages import tree_variance_metrics, tree_grpo, grouped_center
from test_tree_fresh200_cpu import node, config
from test_tree_raw_tokens_cpu import TinyProto


class TailEngine(SearchRollout):
    def complete(self, states, meta):
        paths = [[] for _ in states]
        for i, s in enumerate(states):
            while s.turns < self.action_count:
                paths[i].append(copy.deepcopy(s))
                step = s.turns
                action, obs = [90+step], [700+step] if step < self.action_count-1 else []
                s.response.extend(action+obs); s.model_mask.extend([1]+[0]*len(obs))
                s.steps.extend([step]+[-1]*len(obs)); s.anchors.append(str(step))
                s.records.append(dict(action_tokens=1, observation_tokens=len(obs)))
                s.raw_ledger.append(dict(action_token_ids=action, selected_logprobs=[-.25-step], observation_token_ids=obs))
                s.rolling = s.prompt+s.response
                s.observation = obs; s.turns += 1
            s.done = True
        self.cost['trajectories'] += len(states)
        return paths


class TailTests(unittest.TestCase):
    def run_tail(self, reward=0., actions=3):
        manager = N(tokenizer=N(pad_token_id=151643), config=N(max_start_length=2048),
                    tensor_fn=N(create_position_ids=lambda m: (m.cumsum(-1)-1)*m))
        engine = TailEngine(manager); engine.output_meta = {}; engine.action_count = actions
        calls = []
        def score(states):
            calls.append([(s.source, s.turns, s.branch_metadata.copy()) for s in states])
            return [reward if not callable(reward) else reward(s) for s in states]
        engine.reward_callback = score
        batch = TinyProto(dict(input_ids=torch.tensor([[8], [8]]), attention_mask=torch.ones(2,1,dtype=torch.long)))
        tree = config().baselines.tree; tree.branch_sampling = 'branpo_tail_v1'
        fake = ModuleType('verl'); fake.DataProto = TinyProto
        with patch.dict('sys.modules', {'verl': fake}):
            output, sources = engine.run(batch, tree)
        return engine, output, sources, calls

    def test_finite_same_reward_tail_then_backoff_with_retrieval_prefix(self):
        engine, output, sources, calls = self.run_tail()
        self.assertEqual(sources.tolist(), [0,0,0,1,1])
        self.assertEqual(engine.cost['trajectories'], 5)
        self.assertEqual([len(c) for c in calls], [2,2,1])
        self.assertEqual(len(set(s.trajectory for s in engine.last_states)), 5)
        b0, b1, b2 = [s.branch_metadata for s in engine.last_states if s.branch_metadata['continuation']]
        self.assertEqual([b0['action_depth'], b1['action_depth'], b2['action_depth']], [2,1,2])
        self.assertEqual(b0['inherited_observation_tokens'], 2)
        self.assertEqual(b0['inherited_raw_ledger_actions'], 2)
        self.assertEqual(engine.last_states[1].raw_ledger[:2], engine.last_states[0].raw_ledger[:2])
        self.assertEqual(engine.last_states[1].response[:4], [90,700,91,701])
        self.assertEqual(b1['reason'], 'same_reward_backoff')
        self.assertTrue(all(s.turns == 3 for s in engine.last_states))
        self.assertEqual(output.batch['info_mask'][1,-5:].tolist(), [1,0,1,0,1])
        self.assertTrue(all(b['inherited_prefix_verified'] for b in (b0,b1,b2)))
        self.assertEqual(b1['stop_reason'], 'root_quota_exhausted')

    def test_success_does_not_stop_or_select_and_contrast_keeps_tail(self):
        engine, _, _, calls = self.run_tail(lambda s: float(bool(s.branch_metadata)))
        self.assertEqual([len(c) for c in calls], [2,2,1])
        self.assertEqual(engine.last_states[2].branch_metadata['action_depth'], 2)
        self.assertEqual(engine.last_states[2].branch_metadata['reason'], 'contrast_found_reuse_tail')
        self.assertEqual([s.branch_metadata['sampled_reward'] for s in engine.last_states], [0.,1.,1.,0.,1.])

    def test_all_correct_and_earliest_snapshot_stop_finitely(self):
        engine, _, _, calls = self.run_tail(1., actions=1)
        self.assertEqual([len(c) for c in calls], [2,2,1])
        self.assertTrue(engine.last_states[2].branch_metadata['backoff_clamped'])
        self.assertEqual(engine.last_states[2].branch_metadata['action_depth'], 0)
        with self.assertRaisesRegex(ValueError, 'at least one'):
            self.run_tail(actions=0)

    def metrics(self, scores):
        metadata = [{'root_slot':0,'protocol':'branpo_tail_v1'}]*3+[{'root_slot':1,'protocol':'branpo_tail_v1'}]*2
        return tree_variance_metrics(torch.as_tensor(scores,dtype=torch.float64), ['q']*5, ['a']*3+['b']*2, metadata)

    def test_binary_mixed_moments_and_unchanged_loss_estimator(self):
        score = torch.tensor([0.,0.,1.,0.,1.],dtype=torch.float64)
        m = self.metrics(score)
        self.assertAlmostEqual(m['tree_variance/root0_raw_reward_population_variance'],2/9)
        self.assertAlmostEqual(m['tree_variance/root1_raw_reward_population_variance'],.25)
        self.assertAlmostEqual(m['tree_variance/group_raw_reward_population_variance'],.24)
        self.assertAlmostEqual(m['tree_variance/root0_intra_adv_population_variance'],2/3,places=5)
        self.assertAlmostEqual(m['tree_variance/root1_intra_adv_population_variance'],.5,places=5)
        self.assertAlmostEqual(m['tree_variance/intra_adv_population_variance_leaf_mass'],.6,places=5)
        self.assertLess(m['tree_variance/adv_variance_decomposition_abs_error'],1e-12)
        mask = torch.tensor([[1.,0.,1.]]*5,dtype=torch.float64)
        expected = grouped_center(score,['q']*5)+grouped_center(score,[('q','a')]*3+[('q','b')]*2)
        torch.testing.assert_close(tree_grpo(score[:,None],mask,['q']*5,['a']*3+['b']*2), expected[:,None]*mask)

    def test_all_zero_one_and_root_separated_intra_degeneracy(self):
        for score in ([0]*5,[1]*5):
            m = self.metrics(score)
            self.assertTrue(all(math.isfinite(v) for v in m.values()))
            self.assertEqual(m['tree_variance/intra_adv_population_variance'],0)
            self.assertEqual(m['tree_variance/group_actual_adv_population_variance'],0)
        m = self.metrics([0,0,0,1,1])
        self.assertEqual(m['tree_variance/intra_adv_population_variance'],0)
        self.assertGreater(m['tree_variance/group_actual_adv_population_variance'],0)
        self.assertEqual(m['tree_variance/same_reward_tree_fraction'],1.)
        self.assertEqual(m['tree_variance/mixed_reward_question_fraction'],1.)

    def test_training_reward_callback_uses_actual_reward_and_source_metadata(self):
        from search_r1.baselines.controller import Controller
        from search_r1.baselines.rollout import object_array
        engine, _, _, _ = self.run_tail()
        controller=Controller.__new__(Controller)
        seen=[]
        def actual_reward(data):
            seen.extend(data.non_tensor_batch['reward_model'].tolist())
            return torch.tensor([[v['ground_truth']['target']] for v in seen],dtype=torch.float32)
        controller.trainer=N(reward_fn=actual_reward)
        qa=N(non_tensor_batch={'reward_model':object_array([{'ground_truth':{'target':.2}}, {'ground_truth':{'target':.8}}])})
        original=copy.deepcopy([s.rolling for s in engine.last_states])
        fake=ModuleType('verl');fake.DataProto=TinyProto
        with patch.dict('sys.modules',{'verl':fake}):
            got=controller.tree_training_rewards(engine,qa,[engine.last_states[4],engine.last_states[1]])
        self.assertAlmostEqual(got[0],.8); self.assertAlmostEqual(got[1],.2)
        self.assertEqual([s.rolling for s in engine.last_states],original)
        self.assertEqual(len(qa.non_tensor_batch['reward_model']),2)

    def test_metric_row_permutation_invariance(self):
        score=torch.tensor([0.,0.,1.,0.,1.]);uids=['q']*5;trees=['a']*3+['b']*2
        order=[4,2,0,3,1]
        a=tree_variance_metrics(score,uids,trees)
        b=tree_variance_metrics(score[order],[uids[i] for i in order],[trees[i] for i in order])
        for key in a: self.assertAlmostEqual(a[key],b[key])


if __name__ == '__main__': unittest.main()
