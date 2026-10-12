"""CPU layout/gradient oracles, not actual HF/FSDP/GPU probability evidence."""
import copy
import unittest

import torch

from search_r1.baselines.action_prefix_forward import (
    ACTION_PREFIX_KEYS, MAX_ACTIONS, MAX_PREFIX, MAX_ACTION,
    prepare_action_prefix_micro, forward_prepared_action_prefix,
    scatter_action_prefix_outputs, forward_action_prefix_micro,
)


PAD, EOS, VOCAB = 0, 16, 17


def micro_fixture(rows=None, width=14):
    # Each tuple is actual sampler prefix, complete raw action, timeline offset.
    if rows is None:
        rows = [[([1, 0, 3], [4, 0, 5], 0), ([7, 2], [6, 8], 8)],
                [([9], [10], 2), ([3, 2, 0, 1], [], 6), ([7, 4], [12, 13, 14], 10)]]
    n = len(rows)
    out = dict(responses=torch.full((n, width), 15, dtype=torch.long),
               actor_action_prefix_ids=torch.full((n, MAX_ACTIONS, MAX_PREFIX), PAD, dtype=torch.long),
               actor_action_tokens=torch.full((n, MAX_ACTIONS, MAX_ACTION), PAD, dtype=torch.long),
               actor_action_lengths=torch.zeros((n, MAX_ACTIONS), dtype=torch.long),
               actor_action_prefix_lengths=torch.zeros((n, MAX_ACTIONS), dtype=torch.long),
               actor_action_output_offsets=torch.full((n, MAX_ACTIONS), -1, dtype=torch.long),
               actor_action_count=torch.tensor([len(row) for row in rows], dtype=torch.long),
               loss_mask=torch.zeros((n, width)), step_ids=torch.full((n, width), -1, dtype=torch.long))
    for trajectory, actions in enumerate(rows):
        for slot, (prefix, tokens, offset) in enumerate(actions):
            out['actor_action_prefix_ids'][trajectory, slot, :len(prefix)] = torch.tensor(prefix)
            out['actor_action_tokens'][trajectory, slot, :len(tokens)] = torch.tensor(tokens)
            out['actor_action_lengths'][trajectory, slot] = len(tokens)
            out['actor_action_prefix_lengths'][trajectory, slot] = len(prefix)
            out['actor_action_output_offsets'][trajectory, slot] = offset
            out['responses'][trajectory, offset:offset + len(tokens)] = torch.tensor(tokens)
            out['loss_mask'][trajectory, offset:offset + len(tokens)] = 1
            out['step_ids'][trajectory, offset:offset + len(tokens)] = slot
    return out


def logits(theta, hidden):
    column = torch.linspace(-.4, .5, VOCAB, dtype=theta.dtype, device=theta.device)
    return hidden[..., None] * column + theta[None, :]


class CausalCPUForward:
    """Explicit CPU causal interface with optional pack/roll/unpad semantics."""
    def __init__(self, theta, packed=False):
        self.theta, self.packed, self.calls = theta, packed, 0
    def __call__(self, flat, temperature):
        self.calls += 1
        ids, mask, positions = (flat[key] for key in ('input_ids', 'attention_mask', 'position_ids'))
        action_width = flat['responses'].shape[1]
        if self.packed:
            # Match the original remove-padding position reset and rolled label
            # boundary. Final valid-token cross-row labels must never be scored.
            row_tokens, row_hidden = [], []
            for row in range(len(ids)):
                tokens, pos = ids[row][mask[row].bool()], positions[row][mask[row].bool()]
                row_tokens.append(tokens)
                row_hidden.append(self.theta[tokens].cumsum(0) + pos.to(self.theta.dtype) * .007)
            packed_ids = torch.cat(row_tokens)
            packed_hidden = torch.cat(row_hidden)
            scores = logits(self.theta, packed_hidden) / temperature
            log_softmax = scores.log_softmax(-1)
            logp = log_softmax.gather(-1, packed_ids.roll(-1)[:, None]).squeeze(-1)
            entropy = -(log_softmax.exp() * log_softmax).sum(-1)
            indices = mask.reshape(-1).nonzero().flatten()
            logp = logp.new_zeros(ids.numel()).index_copy(0, indices, logp).reshape_as(ids)
            entropy = entropy.new_zeros(ids.numel()).index_copy(0, indices, entropy).reshape_as(ids)
            return entropy[:, -action_width - 1:-1], logp[:, -action_width - 1:-1]
        hidden = (self.theta[ids] * mask).cumsum(-1) + positions.to(self.theta.dtype) * .007
        scores = logits(self.theta, hidden[:, -action_width - 1:-1]) / temperature
        log_softmax = scores.log_softmax(-1)
        return (-(log_softmax.exp() * log_softmax).sum(-1),
                log_softmax.gather(-1, flat['responses'][..., None]).squeeze(-1))


def manual_per_action(micro, theta, temperature=1.0):
    n, width = micro['responses'].shape
    entropy, logp = theta[:0].sum().expand(n * width), theta[:0].sum().expand(n * width)
    dests, entropies, logps = [], [], []
    forward = CausalCPUForward(theta)
    for trajectory in range(n):
        for slot in range(int(micro['actor_action_count'][trajectory])):
            p = int(micro['actor_action_prefix_lengths'][trajectory, slot])
            length = int(micro['actor_action_lengths'][trajectory, slot])
            offset = int(micro['actor_action_output_offsets'][trajectory, slot])
            if not length:
                continue
            ids = torch.cat((micro['actor_action_prefix_ids'][trajectory, slot, :p],
                             micro['actor_action_tokens'][trajectory, slot, :length]))[None, :]
            flat = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                        position_ids=torch.arange(p + length)[None, :], responses=ids[:, p:])
            ent, lp = forward(flat, temperature)
            entropies.append(ent.flatten()); logps.append(lp.flatten())
            dests.extend(trajectory * width + offset + token for token in range(length))
    if dests:
        indices = torch.tensor(dests)
        entropy = entropy.index_add(0, indices, torch.cat(entropies))
        logp = logp.index_add(0, indices, torch.cat(logps))
    return entropy.reshape(n, width), logp.reshape(n, width)


class ActionPrefixForwardTests(unittest.TestCase):
    def setUp(self):
        self.micro = micro_fixture()
    def theta(self):
        return torch.linspace(-.12, .17, VOCAB, dtype=torch.float64).requires_grad_()
    def assert_oracle(self, micro, packed=False):
        theta, manual_theta = self.theta(), self.theta()
        forward = CausalCPUForward(theta, packed=packed)
        got = forward_action_prefix_micro(micro, .8, forward, PAD, EOS)
        expected = manual_per_action(micro, manual_theta, .8)
        for a, b in zip(got, expected):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        # Original trajectory sequence mean and global mini denominator survive
        # scatter; actions never become independent weighted training rows.
        mask = micro['loss_mask'].double()
        weights = mask / mask.sum(-1, keepdim=True).clamp_min(1)
        row_scale = torch.arange(1, len(mask) + 1, dtype=torch.float64)[:, None]
        loss = ((got[0] * .01 + got[1]) * weights * row_scale).sum() / 32
        expected_loss = ((expected[0] * .01 + expected[1]) * weights * row_scale).sum() / 32
        grad, = torch.autograd.grad(loss, theta)
        expected_grad, = torch.autograd.grad(expected_loss, manual_theta)
        torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)
        self.assertEqual(forward.calls, 1)
    def test_values_and_gradients_match_manual_actual_prefixes(self):
        self.assert_oracle(self.micro)
    def test_remove_padding_rolled_sequence_boundary_values_and_gradients(self):
        self.assert_oracle(self.micro, packed=True)
    def test_row_permutation_keeps_trajectory_layout(self):
        permuted = {key: value[torch.tensor([1, 0])] for key, value in self.micro.items()}
        self.assert_oracle(permuted, packed=True)
        first = forward_action_prefix_micro(self.micro, 1., CausalCPUForward(self.theta()), PAD, EOS)
        second = forward_action_prefix_micro(permuted, 1., CausalCPUForward(self.theta()), PAD, EOS)
        for got, expected in zip(second, first):
            torch.testing.assert_close(got, expected[[1, 0]])
    def test_left_prefix_right_action_padding_and_natural_positions(self):
        prepared = prepare_action_prefix_micro(self.micro, PAD, EOS)
        flat = prepared.flat_micro
        self.assertEqual(tuple(flat['input_ids'].shape), (8, 7))
        self.assertEqual(flat['prompts'][0].tolist(), [PAD, 1, PAD, 3])
        self.assertEqual(flat['responses'][0].tolist(), [4, PAD, 5])
        self.assertEqual(flat['attention_mask'][0].tolist(), [0, 1, 1, 1, 1, 1, 1])
        self.assertEqual(flat['position_ids'][0].tolist(), [0, 0, 1, 2, 3, 4, 5])
        self.assertEqual(flat['attention_mask'][1].tolist(), [0, 0, 1, 1, 1, 1, 0])
        self.assertEqual(flat['position_ids'][1].tolist(), [0, 0, 0, 1, 2, 3, 0])
    def test_real_pad_in_prefix_and_action_is_not_deleted(self):
        prepared = prepare_action_prefix_micro(self.micro, PAD, EOS)
        self.assertTrue(prepared.action_mask[0, 1])
        self.assertEqual(len(prepared.source_indices), int(self.micro['loss_mask'].sum()))
        self.assert_oracle(self.micro, packed=True)
    def test_four_actions_in_same_trajectory_still_one_flat_call(self):
        self.assert_oracle(micro_fixture([[([1], [2], 0), ([3], [4], 3), ([5], [6], 6), ([7], [8], 9)]]), packed=True)
    def test_dummy_eos_nonempty_without_probability_mass(self):
        prepared = prepare_action_prefix_micro(self.micro, PAD, EOS)
        flat = prepared.flat_micro
        for row in (2, 3, 7):
            self.assertEqual(flat['prompts'][row].tolist(), [PAD, PAD, PAD, EOS])
            self.assertEqual(int(flat['attention_mask'][row].sum()), 1)
            self.assertFalse(bool(prepared.action_mask[row].any()))
        self.assertFalse(bool(prepared.action_mask[5].any()))  # actual empty action
        self.assertEqual(int(flat['attention_mask'][5].sum()), 4)
    def test_all_empty_actions_have_graph_connected_zero(self):
        micro = micro_fixture([[([1, 0, 2], [], 0)], []], width=5)
        theta = self.theta(); forward = CausalCPUForward(theta, packed=True)
        entropy, logp = forward_action_prefix_micro(micro, 1., forward, PAD, EOS)
        self.assertTrue(entropy.requires_grad and logp.requires_grad)
        self.assertEqual(float((entropy.sum() + logp.sum()).detach()), 0)
        grad, = torch.autograd.grad(entropy.sum() + logp.sum(), theta)
        torch.testing.assert_close(grad, torch.zeros_like(theta))
        self.assertEqual(forward.calls, 1)
    def test_empty_response_timeline_supported_without_minus_zero_slice(self):
        micro = micro_fixture([[([1], [], 0)], []], width=0)
        theta = self.theta(); forward = CausalCPUForward(theta, packed=True)
        result = forward_action_prefix_micro(micro, 1., forward, PAD, EOS)
        self.assertEqual(tuple(result[0].shape), (2, 0))
        self.assertEqual(forward.calls, 1)
        torch.testing.assert_close(torch.autograd.grad(result[0].sum() + result[1].sum(), theta)[0], torch.zeros_like(theta))
    def test_unselected_nonfinite_does_not_poison_empty_graph_zero(self):
        prepared = prepare_action_prefix_micro(micro_fixture([[]], width=3), PAD, EOS)
        values = torch.full((4, 1), float('nan'), requires_grad=True)
        entropy, logp = scatter_action_prefix_outputs(prepared, values, values)
        self.assertTrue(bool(torch.isfinite(entropy).all() & torch.isfinite(logp).all()))
        (entropy.sum() + logp.sum()).backward()
        torch.testing.assert_close(values.grad, torch.zeros_like(values))
    def test_nonfinite_real_action_stays_visible(self):
        prepared = prepare_action_prefix_micro(self.micro, PAD, EOS)
        values = torch.zeros((8, 3), requires_grad=True)
        bad = values.clone(); bad[0, 0] = float('inf')
        entropy, logp = scatter_action_prefix_outputs(prepared, bad, values)
        self.assertTrue(torch.isinf(entropy[0, 0]))
        self.assertTrue(bool(torch.isfinite(logp).all()))
    def test_inputs_not_mutated_and_cpu_device_not_forced(self):
        before = copy.deepcopy(self.micro)
        prepared = prepare_action_prefix_micro(self.micro, PAD, EOS)
        self.assertEqual(prepared.flat_micro['input_ids'].device.type, 'cpu')
        for key in self.micro:
            torch.testing.assert_close(before[key], self.micro[key])
    def test_info_attention_support_exact_match(self):
        micro = copy.deepcopy(self.micro)
        micro['info_mask'] = torch.cat((torch.ones(2, 3), micro['loss_mask']), 1)
        micro['attention_mask'] = torch.ones_like(micro['info_mask'])
        prepare_action_prefix_micro(micro, PAD, EOS)
        micro['attention_mask'][0, 3] = 0
        with self.assertRaisesRegex(ValueError, 'Info/attention'):
            prepare_action_prefix_micro(micro, PAD, EOS)
    def test_no_grad_reference_outputs_supported(self):
        with torch.no_grad():
            result = forward_action_prefix_micro(self.micro, 1., CausalCPUForward(self.theta()), PAD, EOS)
        self.assertFalse(result[0].requires_grad or result[1].requires_grad)
    def test_malformed_metadata_rejected_before_any_callback(self):
        mutations = [
            lambda m: m.pop('actor_action_tokens'),
            lambda m: m.update(actor_action_tokens=m['actor_action_tokens'].float()),
            lambda m: m.update(actor_action_prefix_ids=m['actor_action_prefix_ids'][:, :, :4095]),
            lambda m: m['actor_action_count'].__setitem__(0, 5),
            lambda m: m['actor_action_count'].__setitem__(0, -1),
            lambda m: m['actor_action_lengths'].__setitem__((0, 0), 501),
            lambda m: m['actor_action_lengths'].__setitem__((0, 0), -1),
            lambda m: m['actor_action_prefix_lengths'].__setitem__((0, 0), 0),
            lambda m: m['actor_action_prefix_lengths'].__setitem__((0, 0), 4097),
            lambda m: m['actor_action_output_offsets'].__setitem__((0, 0), -1),
            lambda m: m['actor_action_output_offsets'].__setitem__((0, 0), 13),
            lambda m: m['actor_action_output_offsets'].__setitem__((0, 1), 2),
            lambda m: m['actor_action_lengths'].__setitem__((0, 2), 1),
            lambda m: m['actor_action_output_offsets'].__setitem__((0, 2), 0),
            lambda m: m['actor_action_prefix_ids'].__setitem__((0, 0, 1), -1),
            lambda m: m['actor_action_tokens'].__setitem__((0, 0, 1), 11),
            lambda m: m['loss_mask'].__setitem__((0, 0), 0),
            lambda m: m['loss_mask'].__setitem__((0, 5), 1),
            lambda m: m['loss_mask'].__setitem__((0, 5), float('nan')),
            lambda m: m['step_ids'].__setitem__((0, 0), 1),
            lambda m: m['step_ids'].__setitem__((0, 5), 0),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                micro = copy.deepcopy(self.micro); mutation(micro)
                forward = CausalCPUForward(self.theta())
                with self.assertRaises(ValueError):
                    forward_action_prefix_micro(micro, 1., forward, PAD, EOS)
                self.assertEqual(forward.calls, 0)
    def test_bad_token_id_and_temperature_rejected(self):
        for pad, eos in [(-1, EOS), (PAD, -1), (True, EOS), (PAD, 1.5)]:
            with self.subTest(pad=pad, eos=eos), self.assertRaises(ValueError):
                prepare_action_prefix_micro(self.micro, pad, eos)
        prepared = prepare_action_prefix_micro(self.micro, PAD, EOS)
        for temperature in [0, -1, float('inf'), float('nan')]:
            forward = CausalCPUForward(self.theta())
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                forward_prepared_action_prefix(prepared, temperature, forward)
            self.assertEqual(forward.calls, 0)
    def test_bad_flat_output_shape_or_dtype_rejected(self):
        prepared = prepare_action_prefix_micro(self.micro, PAD, EOS)
        for values in [torch.zeros(2, 3), torch.zeros(8, 3, dtype=torch.long)]:
            with self.assertRaises(ValueError):
                scatter_action_prefix_outputs(prepared, values, torch.zeros(8, 3))
    def test_full_response_cap_boundaries_preserve_all_500_tokens(self):
        tokens = [1] * 499 + [2]
        micro = micro_fixture([[([3] * 4096, tokens, 0)]], width=500)
        prepared = prepare_action_prefix_micro(micro, PAD, EOS)
        self.assertEqual(tuple(prepared.flat_micro['input_ids'].shape), (4, 4596))
        self.assertEqual(prepared.output_indices.tolist(), list(range(500)))
        self.assertEqual(int(prepared.action_mask.sum()), 500)
    def test_observations_are_zero_and_do_not_enter_action_prefix(self):
        changed = copy.deepcopy(self.micro)
        changed['responses'][~changed['loss_mask'].bool()] = 2
        got = forward_action_prefix_micro(self.micro, 1., CausalCPUForward(self.theta()), PAD, EOS)
        alternate = forward_action_prefix_micro(changed, 1., CausalCPUForward(self.theta()), PAD, EOS)
        for a, b in zip(got, alternate):
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a[~self.micro['loss_mask'].bool()], torch.zeros_like(a[~self.micro['loss_mask'].bool()]))
    def test_no_collective_model_optimizer_or_rng_operations_in_prepare(self):
        state = torch.get_rng_state().clone()
        prepare_action_prefix_micro(self.micro, PAD, EOS)
        torch.testing.assert_close(torch.get_rng_state(), state)
        self.assertEqual(set(ACTION_PREFIX_KEYS), set(self.micro) - {'responses', 'loss_mask', 'step_ids'})


if __name__ == '__main__':
    unittest.main()
