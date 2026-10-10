"""Small synthetic fixtures only; no historical datasets, weights or rollouts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('raw_c_preflight_v2', HERE / 'reconstruct_train_raw_c_preflight_v2.py')
code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code)


class CharacterTokenizer:
    is_fast = True

    def decode(self, ids):
        return ''.join(chr(value) for value in ids)


def row_for(parts):
    prompt = 'Question: synthetic\n'
    ids = [0] + list(map(ord, prompt))
    mask = [0] + [1] * len(prompt)
    width = len(ids)
    info = list(mask)
    for text, authored in parts:
        ids.extend(map(ord, text)); mask.extend([1] * len(text))
        info.extend([int(authored)] * len(text))
    ids.extend([0, 0]); mask.extend([0, 0]); info.extend([0, 0])
    positions = [0] + list(range(sum(mask))) + [0, 0]
    return dict(trajectory_id='synthetic', input_ids=ids, attention_mask=mask,
                info_mask=info, position_ids=positions, prompt_width=width)


def gate_fixture():
    """Synthetic complete 512x5 identity chain, not a real generation proof."""
    run = dict(run_id='1' * 32, source_sha256={'synthetic.py': 'a' * 64})
    step, aid = 7, '2' * 32
    rows, ids, upstream = [], [], {}
    for index in range(2560):
        group = code.document_sha(index // 5)
        identity = (run['run_id'] + '-7-attempt-' + aid + '-' + str(index // 5) +
                    '-' + group + '-sample-' + str(index % 5))
        ids.append(identity); upstream[identity] = group
        rows.append(dict(trajectory_id=identity, input_ids=[65, 66], attention_mask=[1, 1],
                         info_mask=[1, 1], position_ids=[0, 1], prompt_width=1))
    trace = dict(actor_rows=rows, logical_update=step, source_sha256=run['source_sha256'],
                 run_id=run['run_id'], synthetic_unparsed_journal='small synthetic journal')
    tsha = code.document_sha(trace)
    update = dict(step=step, attempt_id=aid, attempt_index=8, behavior_trace_sha256=tsha)
    gate = dict(run_id=run['run_id'], logical_update=step, source_sha256=run['source_sha256'],
                policy_snapshot_id=run['run_id'] + ':7:policy-' + aid,
                accepted=True, trajectory_count=2560, attempt_id=aid, attempt_index=8,
                private_trace_sha256=tsha, trace_sha256=tsha, trajectory_ids=ids,
                actor_rows_sha256=code.document_sha(rows), actor_token_counts=[1] * 2560,
                upstream_group_sha256=upstream, row_tensor_sha256={x: 'b' * 64 for x in ids})
    return run, update, gate, trace


class RawCFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
        if repo is None or not (repo / 'search_g1/signals.py').is_file():
            raise RuntimeError('Explicit pure-source repository path is required for these fixtures')
        sys.path.insert(0, str(repo))
        cls.signals = code.load_pure_module('fixture_signals', repo / 'search_g1/signals.py')
        cls.qa = code.load_pure_module('fixture_qa', repo / 'verl/utils/reward_score/qa_em.py')
        cls.tokenizer = CharacterTokenizer()

    def reconstruct(self, parts, target='gold'):
        return code.reconstruct_row(row_for(parts), {'target': target},
                                    self.tokenizer, self.signals, self.qa)

    def test_exact_answer(self):
        self.assertEqual(self.reconstruct([('<answer>gold</answer>', True)])['C_raw'], 1)

    def test_closed_think(self):
        self.assertEqual(self.reconstruct([('<think>reason</think><answer>gold</answer>', True)])['C_raw'], 1)

    def test_alias_and_original_em_normalization(self):
        self.assertEqual(self.reconstruct([('<answer>The U.S.</answer>', True)], ['US', 'USA'])['C_raw'], 1)

    def test_valid_wrong_is_not_invalid(self):
        value = self.reconstruct([('<answer>wrong</answer>', True)])
        self.assertEqual(value['C_raw'], 0); self.assertTrue(value['final_action_valid'])

    def test_unclosed_think_invalid(self):
        value = self.reconstruct([('<think>reason<answer>gold</answer>', True)])
        self.assertEqual(value['C_raw'], 0); self.assertFalse(value['final_action_valid'])

    def test_nested_tag_invalid(self):
        self.assertEqual(self.reconstruct([('<answer><think>gold</think></answer>', True)])['C_raw'], 0)

    def test_empty_answer_invalid(self):
        self.assertEqual(self.reconstruct([('<answer> </answer>', True)])['C_raw'], 0)

    def test_observation_answer_never_scores(self):
        value = self.reconstruct([('<search>query</search>', True),
                                  ('<information><answer>gold</answer></information>', False)])
        self.assertEqual(value['C_raw'], 0); self.assertEqual(value['model_authored_search_count'], 1)

    def test_observation_search_does_not_count(self):
        value = self.reconstruct([('<information><search>fake</search></information>', False),
                                  ('<answer>gold</answer>', True)])
        self.assertEqual(value['C_raw'], 1); self.assertEqual(value['model_authored_search_count'], 0)

    def test_earlier_answer_cannot_replace_final_search(self):
        self.assertEqual(self.reconstruct([('<answer>gold</answer>', True), ('observation', False),
                                          ('<search>query</search>', True)])['C_raw'], 0)

    def test_answer_cannot_straddle_observation(self):
        self.assertEqual(self.reconstruct([('<answer>', True), ('gold', False), ('</answer>', True)])['C_raw'], 0)

    def test_trailing_text_invalid(self):
        self.assertEqual(self.reconstruct([('<answer>gold</answer> trailing', True)])['C_raw'], 0)

    def test_mask_outside_attention_holds(self):
        row = row_for([('<answer>gold</answer>', True)]); row['info_mask'][-1] = 1
        with self.assertRaisesRegex(code.Hold, 'info_outside_attention'):
            code.actor_parts(row)

    def test_response_attention_hole_holds(self):
        row = row_for([('<answer>gold</answer>', True)]); row['attention_mask'][row['prompt_width'] + 2] = 0
        with self.assertRaisesRegex(code.Hold, 'actor_attention_hole'):
            code.actor_parts(row)

    def test_gold_hash_matches_unicode_canonical(self):
        rows = [dict(extra_info={'index': '题目'}, reward_model={'ground_truth': {'target': ['gold']}})]
        mapping, counts = code.gold_mapping(rows)
        self.assertEqual(code.unique_gold(mapping, code.document_sha('题目')), {'target': ['gold']})
        self.assertEqual(counts[code.document_sha('题目')], 1)

    def test_repeated_index_same_gold_is_allowed(self):
        row = dict(extra_info={'index': 7}, reward_model={'ground_truth': {'target': ['gold']}})
        mapping, counts = code.gold_mapping([row, copy.deepcopy(row)])
        self.assertEqual(code.unique_gold(mapping, code.document_sha(7)), {'target': ['gold']})
        self.assertEqual(counts[code.document_sha(7)], 2)

    def test_repeated_index_conflicting_gold_holds(self):
        rows = [dict(extra_info={'index': 7}, reward_model={'ground_truth': {'target': [gold]}})
                for gold in ('one', 'two')]
        mapping, _ = code.gold_mapping(rows)
        with self.assertRaisesRegex(code.Hold, 'ambiguous_index_gold_join'):
            code.unique_gold(mapping, code.document_sha(7))

    def test_missing_index_holds(self):
        with self.assertRaisesRegex(code.Hold, 'missing_index_gold_join'):
            code.unique_gold({}, code.document_sha(7))

    def test_native_rounded_EM_accepts_only_half_ulp(self):
        self.assertTrue(code.native_em_check(411, 2560, 0.161)['native_EM_check_pass'])
        with self.assertRaisesRegex(code.Hold, 'strict_C_native_EM_mismatch'):
            code.native_em_check(411, 2560, 0.164)

    def test_zero_or_one_C_still_valid_preflight_not_correlation(self):
        self.assertEqual(code.native_em_check(0, 2560, 0.0)['C_hits_exact'], 0)
        self.assertEqual(code.native_em_check(2560, 2560, 1.0)['C_hits_exact'], 2560)

    def test_no_torch_imported(self):
        self.assertNotIn('torch', sys.modules)

    def test_complete_2560_native_gate_and_stream(self):
        run, update, gate, trace = gate_fixture()
        self.assertEqual(len(code.verify_native_gate(gate, update, run)), 2560)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / 'synthetic.json'; path.write_bytes(code.canonical(trace))
            self.assertEqual(code.stream_trace_sha(path, update['behavior_trace_sha256'])['fresh_full_sha256'],
                             update['behavior_trace_sha256'])
            digest = hashlib.sha256(b'['); count = 0
            for index, row in enumerate(code.stream_actor_rows(path, chunk_chars=127)):
                code.verify_streamed_actor(row, index, gate)
                if index:
                    digest.update(b',')
                digest.update(code.canonical(row)); count += 1
            digest.update(b']')
            code.verify_actor_array_digest(digest.hexdigest(), count, gate)

    def test_gate_rejects_identity_and_shape_mutations(self):
        run, update, gate, trace = gate_fixture()
        mutations = [
            ('accepted', False), ('trajectory_count', 2559), ('run_id', 'foreign'),
            ('logical_update', 8), ('source_sha256', {'changed.py': 'c' * 64}),
            ('policy_snapshot_id', 'foreign'), ('attempt_id', '3' * 32), ('attempt_index', 9),
            ('private_trace_sha256', 'c' * 64), ('trace_sha256', 'c' * 64),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                changed = dict(gate); changed[key] = value
                with self.assertRaises(code.Hold):
                    code.verify_native_gate(changed, update, run)

    def test_gate_rejects_duplicate_and_wrong_order_IDs(self):
        run, update, gate, trace = gate_fixture()
        for mode in ('duplicate', 'wrong_order'):
            changed = dict(gate); changed['trajectory_ids'] = list(gate['trajectory_ids'])
            if mode == 'duplicate':
                changed['trajectory_ids'][1] = changed['trajectory_ids'][0]
            else:
                changed['trajectory_ids'][0:2] = changed['trajectory_ids'][1::-1]
            with self.subTest(mode=mode), self.assertRaises(code.Hold):
                code.verify_native_gate(changed, update, run)

    def test_gate_rejects_upstream_hash_and_missing_bindings(self):
        run, update, gate, trace = gate_fixture()
        changed = dict(gate); changed['upstream_group_sha256'] = dict(gate['upstream_group_sha256'])
        changed['upstream_group_sha256'][gate['trajectory_ids'][1]] = code.document_sha('another')
        with self.assertRaisesRegex(code.Hold, 'five_sibling_upstream_hash_mismatch'):
            code.verify_native_gate(changed, update, run)
        changed = dict(gate); changed['row_tensor_sha256'] = dict(gate['row_tensor_sha256'])
        changed['row_tensor_sha256'].pop(gate['trajectory_ids'][0])
        with self.assertRaisesRegex(code.Hold, 'incomplete_gate_row_bindings'):
            code.verify_native_gate(changed, update, run)

    def test_stream_rejects_extra_or_missing_actor_rows(self):
        run, update, gate, trace = gate_fixture()
        with self.assertRaisesRegex(code.Hold, 'too_many_streamed_actor_rows'):
            code.verify_streamed_actor(trace['actor_rows'][0], 2560, gate)
        with self.assertRaisesRegex(code.Hold, 'incomplete_streamed_actor_rows'):
            code.verify_actor_array_digest(gate['actor_rows_sha256'], 2559, gate)

    def test_stream_rejects_rowID_mask_and_token_count_changes(self):
        run, update, gate, trace = gate_fixture()
        changed = dict(trace['actor_rows'][0]); changed['trajectory_id'] = gate['trajectory_ids'][1]
        with self.assertRaisesRegex(code.Hold, 'streamed_actor_ID_order_mismatch'):
            code.verify_streamed_actor(changed, 0, gate)
        changed = dict(trace['actor_rows'][0]); changed['info_mask'] = [1, 0]
        with self.assertRaisesRegex(code.Hold, 'native_actor_token_count_mismatch'):
            code.verify_streamed_actor(changed, 0, gate)
        with self.assertRaisesRegex(code.Hold, 'streamed_actor_array_sha_mismatch'):
            code.verify_actor_array_digest('c' * 64, 2560, gate)

    def test_trace_full_SHA_mismatch_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / 'synthetic.json'; path.write_bytes(b'{"actor_rows":[],"metadata":1}')
            with self.assertRaisesRegex(code.Hold, 'fresh_complete_trace_sha_mismatch'):
                code.stream_trace_sha(path, 'f' * 64)

    def test_stream_is_bounded_and_does_not_parse_trailing_journal(self):
        # Deliberately malformed trailing metadata is not decoded.  Real use
        # authenticates the COMPLETE canonical trace SHA against its native gate.
        row = row_for([('<answer>gold</answer>', True)])
        raw = b'{"actor_rows":[' + code.canonical(row) + b'],"unparsed":' + b'X' * 32768
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / 'synthetic.json'; path.write_bytes(raw)
            self.assertEqual(list(code.stream_actor_rows(path, chunk_chars=31, max_row_json_chars=2048)), [row])
            with self.assertRaisesRegex(code.Hold, 'actor_row_exceeds_stream_buffer_limit'):
                list(code.stream_actor_rows(path, chunk_chars=16, max_row_json_chars=32))

    def test_stream_rejects_noncanonical_prefix_and_partial_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / 'synthetic.json'
            path.write_bytes(b'{"other":1,"actor_rows":[]}')
            with self.assertRaisesRegex(code.Hold, 'noncanonical_trace_actor_rows_prefix'):
                list(code.stream_actor_rows(path))
            path.write_bytes(b'{"actor_rows":[{"input_ids":[65')
            with self.assertRaisesRegex(code.Hold, 'truncated_actor_rows_array'):
                list(code.stream_actor_rows(path, chunk_chars=8))


if __name__ == '__main__':
    # Preserve the explicit repository argument for setUpClass; unittest receives
    # a clean argv so it cannot interpret the path as a test name.
    unittest.main(argv=[sys.argv[0]], verbosity=2)
