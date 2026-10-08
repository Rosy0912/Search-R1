"""Analytical counterexamples and real old-verl DataProto search continuations."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from verl import DataProto
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from search_r1.baselines.advantages import gigpo, tree_grpo, topd, token_mean_ppo
from search_r1.baselines.rollout import SearchRollout, padded, select_rows
from search_r1.baselines.bapo import (BoundaryStage, boundary_scores, resample_rows, legal_action_records,
                                    decode_answers, BoundaryReward)
from search_r1.baselines.cosearch import parse_ranking, ranker_reward, semantic_groups
from search_r1.baselines.distillation import generalized_jsd, teacher_inputs


class CharTokenizer:
    pad_token_id = 0
    pad_token = '<pad>'
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(int(i)) for i in ids if int(i) != 0)
    def batch_decode(self, rows, **kwargs):
        return [self.decode(row) for row in rows]
    def __call__(self, texts, **kwargs):
        return {'input_ids': padded([self.encode(t) for t in texts], 0)}


class Worker:
    def __init__(self, tokenizer, search_only=False):
        self.tokenizer = tokenizer
        self.calls = []
        self.search_only = search_only
    def generate_sequences(self, batch):
        self.calls.append(copy.deepcopy(batch))
        texts = []
        for row in batch.batch['input_ids']:
            text = self.tokenizer.decode(row)
            texts.append('<search>capital</search>' if self.search_only or '<information>' not in text
                         else '<answer>Paris</answer>')
        return DataProto.from_dict({'responses': padded([self.tokenizer.encode(t) for t in texts], 0)},
                                   meta_info=dict(temperature=1., micro_batch_size=2, use_dynamic_bsz=False,
                                                  max_token_len=4096))
    def compute_log_prob(self, batch):
        for key in ('temperature', 'micro_batch_size', 'use_dynamic_bsz', 'max_token_len'):
            assert key in batch.meta_info
        return DataProto.from_dict({'old_log_probs': torch.zeros_like(batch.batch['responses'], dtype=torch.float32)})


def manager(search_only=False, cap=4096, turns=4, force=False, gpus=2):
    tokenizer = CharTokenizer()
    worker = Worker(tokenizer, search_only)
    config = GenerationConfig(turns, 256, cap, 500, 500, gpus, search_url='unused',
                              force_final_answer=force)
    obj = LLMGenerationManager(tokenizer, worker, config)
    queries = []
    def search(q):
        queries.extend(q)
        return ['Paris'] * len(q)
    obj.batch_search = search
    return obj, worker, queries


def batch(tokenizer, count=1):
    ids = padded([tokenizer.encode('Question: capital?')] * count, 0, True)
    return DataProto.from_dict(dict(input_ids=ids, attention_mask=(ids != 0).long(),
                                    position_ids=torch.arange(ids.shape[1]).repeat(count, 1)),
                              meta_info={'do_sample': False, 'validate': True})


class Estimators(unittest.TestCase):
    def test_empty_action_still_participates_in_anchor_mean(self):
        out = gigpo(torch.tensor([[1.], [0.]]), torch.tensor([[1.], [0.]]),
                    torch.tensor([[0], [-1]]), [['root'], ['root']], ['q', 'q'])
        torch.testing.assert_close(out, torch.tensor([[1.], [0.]]))
    def test_topd_microbatches_have_global_token_weight_and_empty_is_safe(self):
        value = torch.tensor(0., requires_grad=True)
        losses = []
        for n, a in ((1, 1.), (3, 3.), (2, 0.)):
            logp = value.expand(1, n)
            mask = torch.zeros_like(logp) if a == 0 else torch.ones_like(logp)
            loss, _, _ = token_mean_ppo(torch.zeros_like(logp), logp, torch.full_like(logp, a), mask, .2, 4.)
            losses.append(loss)
        sum(losses).backward()
        self.assertAlmostEqual(float(value.grad), -2.5)
    def test_gigpo_empty_action_time_and_uid_permutation(self):
        rewards = torch.tensor([[0., 1.], [0., 0.]])
        mask = torch.ones_like(rewards)
        steps = torch.tensor([[0, 2], [0, 2]])
        anchors = [['root', 'empty', 'one'], ['root', 'empty', 'two']]
        adv = gigpo(rewards, mask, steps, anchors, ['q', 'q'], gamma=.95)
        torch.testing.assert_close(adv[:, 0], torch.tensor([.95125, -.95125]))
        reversed_adv = gigpo(rewards.flip(0), mask, steps, anchors[::-1], ['q', 'q'], gamma=.95)
        torch.testing.assert_close(reversed_adv, adv.flip(0))
    def test_tree_independent_inter_and_intra_normalization(self):
        reward = torch.tensor([[1.], [0.], [0.], [0.]])
        out = tree_grpo(reward, torch.ones_like(reward), ['q'] * 4, ['a', 'a', 'b', 'b'])
        expected = (reward[:, 0] - .25) / (reward[:, 0].std() + 1e-6)
        expected[:2] += torch.tensor([.5, -.5]) / (math.sqrt(.5) + 1e-6)
        torch.testing.assert_close(out[:, 0], expected)
    def test_topd_exclusive_future_actor_mean_and_mask(self):
        old = torch.zeros((2, 4))
        teacher = torch.tensor([[1., 900., 2., 3.], [0., -900., 0., 0.]])
        mask = torch.tensor([[1., 0., 1., 1.], [1., 0., 1., 1.]])
        adv, returns, reward = topd(teacher, old, mask, ['q', 'q'], alpha=.2)
        r = torch.log(.2 * teacher[0, [0, 2, 3]].exp() + .8)
        torch.testing.assert_close(returns[0, [0, 2, 3]], torch.stack((r[0] + r[1:].mean(), r[1] + r[2], r[2])))
        self.assertEqual(float(adv[:, 1].abs().sum()), 0.)
        valid = adv[mask.bool()]
        self.assertAlmostEqual(float(valid.mean()), 0., places=5)
        self.assertTrue(torch.isfinite(adv).all())
        self.assertFalse(adv.requires_grad)
    def test_full_vocab_jsd_direction_clip_and_gradient_mask(self):
        s = torch.tensor([[[3., -2., 0.], [1., 1., 1.]]], requires_grad=True)
        t = torch.tensor([[[-2., 3., 0.], [5., -5., 0.]]], requires_grad=True)
        expected = torch.nn.functional.kl_div(s.log_softmax(-1), t.log_softmax(-1),
                                             log_target=True, reduction='none')
        torch.testing.assert_close(generalized_jsd(s, t, component_clip=.05), expected.clamp(max=.05).sum(-1))
        for beta in (0., .5, 1.):
            self.assertTrue(torch.isfinite(generalized_jsd(s, t, beta, component_clip=None)).all())
        generalized_jsd(s, t, component_clip=None)[0, 0].backward()
        self.assertIsNone(t.grad)
        self.assertEqual(float(s.grad[0, 1].abs().sum()), 0.)
        self.assertGreater(float(s.grad[0, 0].abs().sum()), 0.)


class Rollouts(unittest.TestCase):
    def test_inherited_context_mismatch_can_fail_before_generation(self):
        obj, worker, _ = manager(gpus=1)
        obj.config.max_start_length = 2
        with self.assertRaisesRegex(ValueError, 'rollout prefix'):
            SearchRollout(obj, context_alignment='error').run(batch(obj.tokenizer))
        self.assertEqual(worker.calls, [])
        engine = SearchRollout(obj)
        engine.run(batch(obj.tokenizer))
        self.assertGreater(engine.cost['context_mismatches'], 0)
    def test_repeated_parent_snapshot_does_not_change_sampling_weight(self):
        obj, _, _ = manager(gpus=1)
        engine = SearchRollout(obj)
        engine.rng.choice = lambda nodes: nodes[-1]
        engine.run(batch(obj.tokenizer), SimpleNamespace(rounds=2, expansions=2, leaves=3))
        self.assertEqual(engine.tree_candidate_counts, [2, 2])
    def test_real_dataproto_source_selection_and_metadata(self):
        obj, worker, queries = manager()
        engine = SearchRollout(obj)
        output, sources = engine.run(batch(obj.tokenizer, 3))
        self.assertEqual(sources.tolist(), [0, 1, 2])
        selected = select_rows(output, [2, 0])
        self.assertIsInstance(selected, DataProto)
        self.assertEqual(len(selected), 2)
        selected = selected.union(worker.compute_log_prob(selected))
        self.assertIn('old_log_probs', selected.batch)
        for call in worker.calls:
            self.assertFalse(call.meta_info['do_sample'])
            self.assertTrue(call.meta_info['validate'])
            self.assertEqual(len(call) % 2, 0)
        self.assertEqual(len(queries), 3)
        width = output.batch['responses'].shape[1]
        mask = output.batch['info_mask'][:, -width:]
        self.assertTrue(torch.all(output.batch['step_ids'][mask == 0] == -1))
        self.assertEqual(engine.cost['padded_rows'], 2)
    def test_tree_continuation_copies_prefix_without_retrieval_replay(self):
        obj, worker, queries = manager(gpus=1)
        engine = SearchRollout(obj)
        output, sources = engine.run(batch(obj.tokenizer), SimpleNamespace(rounds=2, expansions=2, leaves=3))
        self.assertEqual(len(output), 3)
        self.assertEqual(len(set(output.non_tensor_batch['trajectory_uid'])), 3)
        self.assertEqual(len(set(output.non_tensor_batch['tree_uid'])), 1)
        self.assertTrue(all(s.turns <= 4 for s in engine.last_states))
        self.assertEqual(engine.cost['trajectories'], 5)
        self.assertEqual(engine.cost['retrieval_calls'], len(queries))
        # Any continuation resumed after the shared search has no new search call.
        self.assertLess(len(queries), 5)
        initial = engine.last_states[0].branch()
        before = copy.deepcopy(engine.last_states[0].response)
        initial.response.append(123)
        self.assertEqual(engine.last_states[0].response, before)
    def test_budget_exhaustion_is_not_environment_completion(self):
        obj, worker, queries = manager(search_only=True, turns=4, gpus=1)
        output, _ = SearchRollout(obj).run(batch(obj.tokenizer))
        self.assertEqual(len(worker.calls), 4)
        self.assertEqual(output.meta_info['active_mask'], [True])
        self.assertEqual(output.meta_info['turns_stats'], [4])
    def test_response_cap_and_one_final_generation(self):
        obj, worker, queries = manager(search_only=True, cap=80, turns=2, force=True, gpus=1)
        output, _ = SearchRollout(obj).run(batch(obj.tokenizer))
        self.assertEqual(len(worker.calls), 3)
        self.assertEqual(len(queries), 2)
        self.assertLessEqual(output.batch['responses'].shape[1], 80)
        self.assertEqual(output.batch['step_ids'].shape, output.batch['responses'].shape)
        self.assertEqual(output.meta_info['turns_stats'], [3])
    def test_first_anchor_is_question_and_does_not_include_future_answer(self):
        obj, _, _ = manager(gpus=1)
        engine = SearchRollout(obj)
        output, _ = engine.run(batch(obj.tokenizer, 2))
        self.assertEqual(output.non_tensor_batch['anchors'][0][0], output.non_tensor_batch['anchors'][1][0])
        self.assertNotEqual(output.non_tensor_batch['anchors'][0][0], output.non_tensor_batch['anchors'][0][1])


class BoundaryAndRanker(unittest.TestCase):
    def test_low_diversity_bonus_high_diversity_no_bonus(self):
        stage = BoundaryStage(early=False)
        low = ["I don't know"] * 4 + ['wrong'] * 4
        scores = boundary_scores(low, [['Paris']] * 8, ['q'] * 8, stage)
        self.assertEqual(scores[:4], [.5] * 4)
        high = ["I don't know"] * 4 + ['wrong1', 'wrong2', 'wrong3', 'wrong4']
        self.assertEqual(boundary_scores(high, [['Paris']] * 8, ['q'] * 8, stage), [0.] * 8)
        low[-1] = 'Paris'
        self.assertEqual(boundary_scores(low, [['Paris']] * 8, ['q'] * 8, stage)[:4], [0.] * 4)
    def test_stage_consecutive_stalls_and_state_roundtrip(self):
        stage = BoundaryStage(patience=3)
        for value in (.1, .1, .1, .2, .2, .2):
            stage.observe(value)
        self.assertTrue(stage.early)
        stage.observe(.2)
        self.assertFalse(stage.early)
        other = BoundaryStage()
        other.load_state_dict(stage.state_dict())
        self.assertEqual(other.state_dict(), stage.state_dict())
        self.assertEqual(resample_rows(['wrong'] * 5, [['Paris']] * 5, ['q'] * 5, other), list(range(5)))
    def test_recovered_invalid_action_cannot_earn_boundary_bonus(self):
        self.assertFalse(legal_action_records([{'text': 'invalid'}, {'text': '<answer>IDK</answer>'}]))
        self.assertFalse(legal_action_records([{'text': '<answer>IDK</answer><answer>IDK</answer>'}]))
        self.assertTrue(legal_action_records([{'text': '<think>x</think><search>x</search>'},
                                             {'text': '<think>y</think><answer>Paris</answer>'}]))
    def test_ranking_bounds_duplicates_and_reward_threshold(self):
        self.assertEqual(parse_ranking('<ranking>[2,1,3]</ranking>', 5, 3), [2, 1, 3])
        for text in ('<ranking>[1,1,2]</ranking>', '<ranking>[0,1,2]</ranking>', '<ranking>[true,1,2]</ranking>'):
            self.assertIsNone(parse_ranking(text, 3, 3))
        self.assertAlmostEqual(ranker_reward(['Paris', 'Rome', 'Berlin'], [2, 3, 1], ['Paris'], 1.), 5 / 3)
        self.assertEqual(ranker_reward(['Paris', 'Rome', 'Berlin'], [2], ['Paris'], 1.), 0.)
        self.assertEqual(ranker_reward(['Rome'], [1], ['Paris'], 1.), 1.)
        self.assertEqual(ranker_reward(['Rome'], None, ['Paris'], 1.), -.2)
    def test_semantic_groups_do_not_cross_questions(self):
        events = [dict(question_uid=q, query=text) for q, text in [('a', 'who is president'),
                    ('a', 'who president is'), ('b', 'who is president'), ('a', 'capital france')]]
        keys = semantic_groups(events)
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])
        self.assertNotEqual(keys[0], keys[3])
    def test_teacher_target_alignment_with_padding_and_internal_hole(self):
        tokenizer = CharTokenizer()
        prompts = torch.tensor([[0, 65, 66], [0, 0, 67]])
        responses = torch.tensor([[68, 0, 69], [70, 0, 0]])
        attention = torch.tensor([[1, 0, 1], [1, 0, 0]])
        ids, mask, positions = teacher_inputs(tokenizer, prompts, responses, attention)
        self.assertEqual(ids[0, :4].tolist(), [65, 66, 68, 69])
        self.assertEqual(positions[0, [0, 2]].tolist(), [1, 2])
        self.assertEqual(positions[1, 0], 0)


if __name__ == '__main__':
    unittest.main()
