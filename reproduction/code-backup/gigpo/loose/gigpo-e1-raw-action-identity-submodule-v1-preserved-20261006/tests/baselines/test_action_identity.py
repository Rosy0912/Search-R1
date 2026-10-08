"""CPU counterexamples for sampled tokens and real DataProto rollout layout.

The byte tokenizer/worker are explicit CPU interfaces, not model/GPU evidence.
Actual 3289 Qwen token material remains on H100 and is bound separately.
"""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from verl import DataProto
from search_r1.baselines.action_identity import (IDENTITY_MODE, retain_sampled_action,
    sampled_ids_with_evidence, canonical_environment_text, _BYTE_DECODER)
from search_r1.baselines.rollout import SearchRollout, padded, select_rows
from search_r1.baselines.objectives import attach_policy_weights, weighted_policy_loss
from search_r1.baselines.advantages import gigpo
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager


class ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    pad_token = '<pad>'
    all_special_ids = [0, 98, 99]
    pieces = {1: b'A', 2: b'<search>', 3: b'q</search>\n', 4: b'<answer>',
              5: b'ok</answer>\n', 6: b'\xe0\xae', 7: b'\x85', 8: b'</se',
              9: b'arch>tail', 10: b'</answer></search>\n'}
    _encoder = {value: character for character, value in _BYTE_DECODER.items()}
    def token_bytes(self, token):
        return self.pieces[token] if token in self.pieces else bytes([token - 1000])
    def convert_ids_to_tokens(self, token):
        if token in self.all_special_ids:
            return {0: '<pad>', 98: '<special>', 99: '<eos>'}[token]
        return ''.join(self._encoder[byte] for byte in self.token_bytes(token))
    def encode(self, text, **kwargs):
        # Canonical tokenize intentionally chooses a different valid path.
        out = []
        while text:
            marker = next(((value.decode(), token) for token, value in self.pieces.items()
                           if value in (b'<search>', b'<answer>') and text.startswith(value.decode())), None)
            if marker:
                out.append(marker[1]); text = text[len(marker[0]):]
            else:
                char, text = text[0], text[1:]
                out.extend(1000 + byte for byte in char.encode('utf-8'))
        return out
    def decode(self, tokens, skip_special_tokens=False, **kwargs):
        pieces = []
        for token in map(int, tokens):
            if token in self.all_special_ids:
                if not skip_special_tokens:
                    pieces.append(self.convert_ids_to_tokens(token).encode())
            else:
                pieces.append(self.token_bytes(token))
        return b''.join(pieces).decode('utf-8', errors='replace')
    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]
    def __call__(self, texts, **kwargs):
        return {'input_ids': padded([self.encode(text) for text in texts], self.pad_token_id)}


class SampledWorker:
    def __init__(self, tokenizer, actions):
        self.tokenizer, self.actions, self.calls = tokenizer, actions, []
    def generate_sequences(self, batch):
        self.calls.append(copy.deepcopy(batch))
        real = self.actions[len(self.calls) - 1]
        rows = [real] * len(batch)
        responses = padded(rows, self.tokenizer.pad_token_id)
        lengths = torch.tensor(list(map(len, rows)), dtype=torch.long)
        mask = torch.arange(responses.shape[1])[None, :] < lengths[:, None]
        return DataProto.from_dict(dict(responses=responses, response_sampled_lengths=lengths,
            response_sample_mask=mask), meta_info={'temperature': 1.})


def manager(actions, mode=IDENTITY_MODE):
    tokenizer = ByteTokenizer()
    worker = SampledWorker(tokenizer, actions)
    config = GenerationConfig(4, 2048, 4096, 500, 500, 8, force_final_answer=False)
    config.actor_token_identity = mode
    obj = LLMGenerationManager(tokenizer, worker, config)
    obj.batch_search = lambda queries: ['doc'] * len(queries)
    return obj, worker


def initial_batch(tokenizer, rows=2):
    prompts = padded([tokenizer.encode('Question?')] * rows, 0, left=True)
    return DataProto.from_dict(dict(input_ids=prompts, attention_mask=torch.ones_like(prompts),
                                   position_ids=torch.arange(prompts.shape[1]).repeat(rows, 1)))


class ExactTokenPrefixes(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
    def action(self, ids, canonical=None, length=None, mask=None):
        length = len(ids) if length is None else length
        mask = [1] * length + [0] * (len(ids) - length) if mask is None else mask
        if canonical is None:
            canonical = canonical_environment_text(self.tok.decode(ids[:length], skip_special_tokens=True))
        return retain_sampled_action(self.tok, ids, length, mask, canonical)
    def test_noncanonical_first_token_preserves_raw_prefix(self):
        ids = [1000 + byte for byte in b'<search>'] + [3, 1, 99, 0]
        out = self.action(ids, length=len(ids) - 1)
        self.assertEqual(list(out.actor_ids), ids[:9])
        self.assertNotEqual(self.tok.encode(out.canonical_text), list(out.actor_ids))
        self.assertEqual(out.closing_token_index, 8)
        self.assertTrue(out.closing_inside_token)
        self.assertEqual(out.canonical_text, '<search>q</search>')
    def test_closing_across_token_boundary_keeps_atomic_suffix(self):
        out = self.action([1, 8, 9, 1, 99])
        self.assertEqual(out.actor_ids, (1, 8, 9))
        self.assertEqual(out.canonical_text, 'A</search>')
        self.assertTrue(out.closing_inside_token)
    def test_original_search_priority_is_preserved(self):
        out = self.action([4, 10, 99])
        self.assertEqual(out.selected_closing, '</search>')
        self.assertEqual(out.actor_ids, (4, 10))
        self.assertEqual(out.canonical_text, '<answer></answer></search>')
    def test_incomplete_utf8_final_token_is_not_retokenized(self):
        ids = [1] * 499 + [6]
        out = self.action(ids)
        self.assertEqual(len(out.actor_ids), 500)
        self.assertEqual(list(out.actor_ids), ids)
        self.assertGreater(len(self.tok.encode(out.canonical_text)), 500)
    def test_split_utf8_before_closing_maps_whole_tokens(self):
        out = self.action([6, 7, 8, 9, 99])
        self.assertEqual(out.actor_ids, (6, 7, 8, 9))
        self.assertEqual(out.canonical_text, '\u0b85</search>')
    def test_real_interior_pad_and_special_have_probability_mass(self):
        out = self.action([2, 0, 98, 3, 99, 0], length=5)
        self.assertEqual(out.actor_ids, (2, 0, 98, 3))
        self.assertEqual(out.canonical_text, '<search>q</search>')
    def test_terminal_eos_is_excluded_but_empty_action_is_supported(self):
        self.assertEqual(self.action([1, 99, 0], length=2).actor_ids, (1,))
        self.assertEqual(self.action([99, 0], length=1).actor_ids, ())
        self.assertEqual(self.action([0], length=0).actor_ids, ())
    def test_nonterminal_eos_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Nonterminal EOS'):
            self.action([1, 99, 1])
    def test_lengths_and_mask_are_not_guessed_from_ids(self):
        self.assertEqual(sampled_ids_with_evidence([1, 0, 1, 0], 3, [1, 1, 1, 0]), [1, 0, 1])
        for length, mask in [(3, [1, 0, 1, 0]), (2, [1, 1, 1, 0]), (5, [1] * 4), (2.5, [1, 1, 0, 0])]:
            with self.subTest(length=length, mask=mask), self.assertRaises(ValueError):
                sampled_ids_with_evidence([1, 0, 1, 0], length, mask)
    def test_canonical_text_or_decoder_inconsistency_rejects(self):
        with self.assertRaisesRegex(ValueError, 'Canonical environment text'):
            self.action([2, 3], canonical='<answer>wrong</answer>')
        tok = copy.deepcopy(self.tok)
        tok.convert_ids_to_tokens = lambda token: self.tok._encoder[ord('B')]
        with self.assertRaisesRegex(ValueError, 'ByteLevel evidence'):
            retain_sampled_action(tok, [1], 1, [1], 'A')
    def test_unsupported_decoder_alphabet_rejects(self):
        tok = copy.deepcopy(self.tok)
        tok.convert_ids_to_tokens = lambda token: '\u4e00'
        with self.assertRaisesRegex(ValueError, 'Unsupported non-ByteLevel'):
            retain_sampled_action(tok, [1], 1, [1], 'A')


class ActualDataProtoLayout(unittest.TestCase):
    def run_pair(self, actions):
        raw_manager, raw_worker = manager(actions)
        canonical_manager, canonical_worker = manager(actions, 'legacy_canonical')
        raw_engine = SearchRollout(raw_manager, method='gigpo')
        canonical_engine = SearchRollout(canonical_manager)
        raw_batch, sources = raw_engine.run(initial_batch(raw_manager.tokenizer))
        old_batch, old_sources = canonical_engine.run(initial_batch(canonical_manager.tokenizer))
        return raw_batch, old_batch, sources, old_sources, raw_worker, canonical_worker
    def test_canonical_environment_and_next_sampling_context_unchanged(self):
        raw, old, sources, old_sources, worker, old_worker = self.run_pair([[2, 3, 99], [4, 5, 99]])
        np.testing.assert_array_equal(sources, old_sources)
        for got, expected in zip(worker.calls, old_worker.calls):
            for key in ('input_ids', 'attention_mask', 'position_ids'):
                torch.testing.assert_close(got.batch[key], expected.batch[key])
        for key, canonical_key in [('canonical_reward_responses', 'responses'),
                                    ('canonical_reward_attention_mask', 'attention_mask'),
                                    ('canonical_reward_info_mask', 'info_mask'),
                                    ('canonical_reward_step_ids', 'step_ids')]:
            torch.testing.assert_close(raw.batch[key], old.batch[canonical_key])
        self.assertEqual(raw.non_tensor_batch['anchors'].tolist(), old.non_tensor_batch['anchors'].tolist())
        self.assertEqual(raw.meta_info['actor_token_identity'], IDENTITY_MODE)
        self.assertEqual(raw.batch['actor_action_count'].tolist(), [2, 2])
        self.assertEqual(raw.batch['actor_action_lengths'].tolist(), [[2, 2, 0, 0]] * 2)
        for i in range(2):
            for step in range(2):
                record = raw.non_tensor_batch['action_prefix_records'][i][step]
                prefix = worker.calls[step].batch['input_ids'][i]
                prefix = prefix[worker.calls[step].batch['attention_mask'][i].bool()].tolist()
                self.assertEqual(record['generation_prefix'], prefix)
                self.assertEqual(record['prefix_position_ids'], list(range(len(prefix))))
                offset = int(raw.batch['actor_action_output_offsets'][i, step])
                length = int(raw.batch['actor_action_lengths'][i, step])
                self.assertEqual(raw.batch['responses'][i, offset:offset + length].tolist(),
                                 raw.batch['actor_action_tokens'][i, step, :length].tolist())
                self.assertEqual(raw.batch['step_ids'][i, offset:offset + length].tolist(), [step] * length)
    def test_true_sampled_pad_has_attention_and_loss_and_survives_row_selection(self):
        raw, _, _, _, _, _ = self.run_pair([[2, 0, 98, 3, 99], [4, 5, 99]])
        prompt_width = raw.batch['prompts'].shape[1]
        self.assertEqual(int(raw.batch['responses'][0, 1]), 0)
        self.assertEqual(int(raw.batch['attention_mask'][0, prompt_width + 1]), 1)
        self.assertEqual(int(raw.batch['info_mask'][0, prompt_width + 1]), 1)
        self.assertEqual(int(raw.batch['step_ids'][0, 1]), 0)
        selected = select_rows(raw, [1, 0, 1])
        self.assertEqual(selected.batch['actor_action_lengths'].shape, (3, 4))
        for i, original in enumerate([1, 0, 1]):
            self.assertEqual(selected.non_tensor_batch['action_prefix_records'][i],
                             raw.non_tensor_batch['action_prefix_records'][original])
    def test_shapes_and_empty_actions_preserve_horizon(self):
        obj, _ = manager([[99], [4, 5, 99]])
        out, _ = SearchRollout(obj, method='gigpo').run(initial_batch(obj.tokenizer, 1))
        self.assertEqual(out.batch['actor_action_prefix_ids'].shape, (1, 4, 4096))
        self.assertEqual(out.batch['actor_action_tokens'].shape, (1, 4, 500))
        self.assertEqual(out.batch['actor_action_lengths'][0].tolist(), [0, 2, 0, 0])
        self.assertEqual(out.batch['actor_action_output_offsets'][0, 0], 0)
        self.assertEqual(len(out.non_tensor_batch['anchors'][0]), 2)
    def test_missing_or_invalid_actual_sampler_evidence_rejects(self):
        for missing in ('response_sampled_lengths', 'response_sample_mask'):
            obj, worker = manager([[4, 5, 99]])
            original = worker.generate_sequences
            def generate(batch, field=missing):
                out = original(batch)
                del out.batch[field]
                return out
            worker.generate_sequences = generate
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'original engine'):
                SearchRollout(obj, method='gigpo').run(initial_batch(obj.tokenizer))
    def test_raw_enabled_configuration_scope_rejects(self):
        for key, value in [('max_turns', 5), ('max_prompt_length', 4000),
                           ('max_response_length', 128), ('force_final_answer', True),
                           ('max_start_length', 1024), ('num_gpus', 4), ('no_think_rl', True)]:
            obj, _ = manager([[4, 5, 99]])
            setattr(obj.config, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                SearchRollout(obj, method='gigpo')
        obj, _ = manager([[4, 5, 99]])
        with self.assertRaises(ValueError):
            SearchRollout(obj, method='tree_grpo')
        with self.assertRaises(ValueError):
            SearchRollout(obj, method='gigpo', anchor_mode='context')
        engine = SearchRollout(obj, method='gigpo')
        with self.assertRaisesRegex(ValueError, 'without tree expansion'):
            engine.run(initial_batch(obj.tokenizer), tree=SimpleNamespace())
    def test_incorrect_output_mapping_rejects_at_compose(self):
        obj, _ = manager([[4, 5, 99]])
        engine = SearchRollout(obj, method='gigpo')
        states = engine.initial_states(initial_batch(obj.tokenizer, 1))
        engine.advance(states, {})
        states[0].raw_steps[0] = 3
        with self.assertRaisesRegex(ValueError, 'output mapping'):
            engine.compose(states)
    def test_legacy_flag_keeps_original_batch_schema(self):
        obj, _ = manager([[4, 5, 99]], 'legacy_canonical')
        out, _ = SearchRollout(obj).run(initial_batch(obj.tokenizer, 1))
        self.assertNotIn('actor_action_tokens', out.batch)
        self.assertNotIn('canonical_reward_responses', out.batch)
        self.assertNotIn('actor_token_identity', out.meta_info)
    def test_raw_joint_sum_sequence_mean_and_grouping_formula(self):
        raw, _, _, _, _, _ = self.run_pair([[2, 3, 99], [4, 5, 99]])
        raw.non_tensor_batch['uid'] = np.asarray(['q', 'q'], dtype=object)
        width = raw.batch['responses'].shape[1]
        mask = raw.batch['attention_mask'][:, -width:] * raw.batch['info_mask'][:, -width:]
        rewards = torch.zeros_like(mask, dtype=torch.float32)
        active = mask[0].bool().nonzero().flatten()
        rewards[0, active[-1]] = 1
        adv = gigpo(rewards, mask, raw.batch['step_ids'], raw.non_tensor_batch['anchors'], ['q', 'q'])
        self.assertEqual(adv[0, active[0]], 0.975)
        self.assertEqual(adv[0, active[-1]], 1.)
        for step in range(2):
            selected = raw.batch['step_ids'][0] == step
            self.assertTrue(bool((adv[0, selected] == adv[0, selected][0]).all()))
        attach_policy_weights(raw, 'gigpo', 'sequence_mean')
        torch.testing.assert_close(raw.batch['policy_weights'].sum(-1), torch.ones(2))
        current = torch.tensor(0.01, requires_grad=True)
        old = torch.zeros_like(adv)
        row_sign = torch.tensor([[1.], [-1.]])
        logp = current * row_sign.expand_as(adv)
        loss, _, _ = weighted_policy_loss(old, logp, adv, mask, raw.batch['policy_weights'], .2, 2,
                                          raw.batch['step_ids'], dual_clip=3.)
        # Each action has exactly two raw tokens; the joint logratio is 2*x,
        # never a token mean, and the full trajectory has four equal-mass tokens.
        expected = -(adv.double() * mask.double() * torch.exp(current.double() * row_sign.double() * 2) / 4).sum() / 2
        torch.testing.assert_close(loss, expected)
        grad = torch.autograd.grad(loss, current, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, current)[0]
        torch.testing.assert_close(grad, expected_grad)
        split = sum(weighted_policy_loss(old[i:i + 1], logp[i:i + 1], adv[i:i + 1], mask[i:i + 1],
                                         raw.batch['policy_weights'][i:i + 1], .2, 2,
                                         raw.batch['step_ids'][i:i + 1], dual_clip=3.)[0]
                    for i in range(2))
        torch.testing.assert_close(loss, split)
        torch.testing.assert_close(torch.autograd.grad(split, current)[0], grad)


if __name__ == '__main__':
    unittest.main()
