"""Real CPU protocol fixtures; they do not authenticate GPU execution/health."""
import ast
import copy
import re
from types import SimpleNamespace
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8 import behavior_trace as bt


def original_upstream_manager():
    """Execute the original parser/executor bodies with explicit CPU boundaries.

    AST extraction avoids importing model/retriever code. Only annotations
    are removed; tokenizer/search are fixtures, not native execution evidence.
    """
    source = Path(__file__).resolve().parents[2] / 'search_r1/llm_agent/generation.py'
    tree = ast.parse(source.read_text())
    names = {'_postprocess_responses', 'postprocess_predictions', 'execute_predictions'}
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    for node in nodes:
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    namespace = {'re': re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), namespace)
    manager_type = type('OriginalCpuParser', (), {name: namespace[name] for name in names})
    manager = manager_type()
    manager.config = SimpleNamespace(no_think_rl=False)
    manager.tokenizer = SimpleNamespace(batch_decode=lambda rows, **kwargs: rows)
    manager._batch_tokenize = lambda rows: rows
    manager.search_calls = []
    def search(queries):
        manager.search_calls.append(list(queries))
        return ['CPU fixture evidence'] * len(queries)
    manager.batch_search = search
    return manager


def tiny_outer_fixture():
    """A complete512/G5 recorded two-turn outer for CPU integration tests.

    Every row includes model token20, observed token30, then model token21.
    Its native selected LP includes trailing EOS; final actor excludes that
    EOS, preserves both model token IDs, masks the observation, and pads both
    sides. This is a constructed CPU fixture, never physical/model evidence.
    """
    sampling = dict(n=1, best_of=1, do_sample=True, temperature=1.0, top_p=1.0,
                    top_k=-1, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                    repetition_penalty=1.0, ignore_eos=False, logprobs=1, min_tokens=0,
                    logits_processors=[], logprob_semantics=bt.LOGPROB_SEMANTICS,
                    stop=None, stop_token_ids=None, seed=None, max_tokens=500,
                    logit_bias=None, allowed_token_ids=None, guided_decoding=None,
                    truncate_prompt_tokens=None)
    source = {'reproduction/common8/behavior_trace.py': 'a' * 64}
    parents = ['question-occurrence-' + str(index) for index in range(512)]
    ids = ['cpu-trajectory-' + str(index) for index in range(2560)]
    trajectories, rows = [], []
    for index, identity in enumerate(ids):
        turns = []
        for turn_index, (request, token) in enumerate((([10, 11], 20), ([10, 11, 20, 30], 21))):
            selected = -(0.2 + (index % 5) * 0.01 + turn_index * 0.1)
            record = dict(schema_version=1, request_id='native-' + str(index) + '-' + str(turn_index),
                          row_id=identity, rollout_ordinal=turn_index,
                          rollout_call_id='call-' + str(turn_index), actual_request_token_ids=request,
                          raw_generated_token_ids=[token, 99], raw_generated_token_count=2,
                          selected_logprobs=[selected, -1.0], finish_reason='stop', stop_reason=None,
                          eos_token_id=99, pad_token_id=0, sampling=copy.deepcopy(sampling),
                          policy_snapshot_id='cpu-policy0')
            turns.append(bt.make_turn(record, trajectory_id=identity, turn_index=turn_index,
                                      kept_token_count=1, special_token_ids=[0, 99],
                                      observation_token_ids=[30] if turn_index == 0 else [],
                                      final_turn=turn_index == 1))
        trajectories.append(dict(trajectory_id=identity, parent_occurrence_index=index // 5,
                                 sample_index=index % 5, turns=turns))
        rows.append(bt.make_actor_row(trajectory_id=identity, input_ids=[0, 10, 11, 20, 30, 21, 0],
                                      attention_mask=[0, 1, 1, 1, 1, 1, 0],
                                      info_mask=[0, 1, 1, 1, 0, 1, 0],
                                      position_ids=[0, 0, 1, 2, 3, 4, 0], prompt_width=3))
    trace = bt.make_trace(run_id='cpu-fixture-run', source_sha256=source, slurm_job_id='12345',
                          logical_update=1, policy_snapshot_id='cpu-policy0', sampling=sampling,
                          parent_group_ids=parents, trajectories=trajectories, actor_rows=rows)
    bindings = dict(expected_run_id=trace['run_id'], expected_source_sha256=source,
                    expected_slurm_job_id='12345', expected_logical_update=1,
                    expected_parent_group_ids=parents, expected_trajectory_ids=ids,
                    expected_policy_snapshot_id='cpu-policy0', expected_sampling=sampling)
    return trace, bindings


class RawPrefixTests(unittest.TestCase):
    def cut(self, fragments, raw):
        return bt.raw_prefix_to_first_closing_tag(raw, lambda ids: ''.join(fragments[i] for i in ids),
                                                 eos_token_id=99, pad_token_id=0,
                                                 special_token_ids=[0, 99, 98])

    def test_exact_original_tokens_and_cross_token_tag(self):
        fragments = {1: '<answer>A', 2: '</ans', 3: 'wer>', 4: 'discarded text'}
        proof = self.cut(fragments, [1, 2, 3, 4, 99])
        self.assertEqual(proof['kept_token_ids'], [1, 2, 3])
        self.assertEqual(proof['execution_text'], '<answer>A</answer>')
        self.assertTrue(proof['trailing_eos_excluded'])
        self.assertEqual(proof['raw_generated_token_count'], 5)

    def test_inside_token_nonwhitespace_preserves_whole_token(self):
        proof = self.cut({1: '<answer>A</answer>secret'}, [1])
        self.assertEqual(proof['kept_token_ids'], [1])
        self.assertEqual(proof['execution_text'], '<answer>A</answer>')
        self.assertEqual(proof['decoded_retained_prefix'], '<answer>A</answer>secret')
        self.assertEqual(proof['retained_token_boundary_suffix'], 'secret')
        self.assertEqual(proof['token_boundary_suffix_classification'], 'nonwhitespace')
        self.assertEqual(proof['retained_trailing_whitespace'], '')
        self.assertFalse(proof['token_boundary_whitespace_retained'])

    def test_inside_token_whitespace_preserves_whole_token(self):
        proof = self.cut({1: '<answer>A</answer> \n', 2: 'later'}, [1, 2])
        self.assertEqual(proof['kept_token_ids'], [1])
        self.assertEqual(proof['execution_text'], '<answer>A</answer>')
        self.assertEqual(proof['retained_trailing_whitespace'], ' \n')
        self.assertTrue(proof['token_boundary_whitespace_retained'])

    def test_original_search_boundary_and_first_complete_action_ast(self):
        cases = (
            ({1: '<answer>A</answer>', 2: '<search>B</search>'}, [1, 2], [1, 2], 'answer', 1, 0),
            ({1: '<search>B</search>', 2: '<answer>A</answer>'}, [1, 2], [1], 'search', 0, 1),
            ({1: '</answer>', 2: '<search>B</search>'}, [1, 2], [1, 2], 'search', 0, 1),
            ({1: '<answer>A</answer>', 2: '</search>'}, [1, 2], [1, 2], 'answer', 1, 0),
            ({1: '<search>B</search>', 2: '<search>C</search>'}, [1, 2], [1], 'search', 0, 1),
        )
        for fragments, raw, kept, action, done, searches in cases:
            with self.subTest(raw=fragments):
                proof = self.cut(fragments, raw)
                original = original_upstream_manager()
                _, texts = original._postprocess_responses([''.join(fragments[token] for token in raw)])
                self.assertEqual(proof['execution_text'], texts[0])
                self.assertEqual(proof['kept_token_ids'], kept)
                self.assertEqual(original.postprocess_predictions(texts)[0], [action])
                observation, dones, valid, is_search = original.execute_predictions(texts, None, [True])
                self.assertEqual((dones, valid, is_search), ([done], [1], [searches]))
                self.assertEqual(len(original.search_calls[-1]), searches)

    def test_selected_search_boundary_preserves_nonwhitespace_token_suffix(self):
        proof = self.cut({1: '<answer>A</answer><search>B</search>secret'}, [1])
        self.assertEqual(proof['execution_text'], '<answer>A</answer><search>B</search>')
        self.assertEqual(proof['decoded_retained_prefix'], '<answer>A</answer><search>B</search>secret')
        self.assertEqual(proof['kept_token_ids'], [1])

    def test_no_tag_only_tail_eos_excluded(self):
        proof = self.cut({1: 'text'}, [1, 99])
        self.assertEqual(proof['kept_token_ids'], [1])
        self.assertIsNone(proof['closing_tag'])
        for raw in ([1, 99, 1], [1, 0], [1, 98]):
            with self.subTest(raw=raw), self.assertRaises(bt.BehaviorTraceError):
                self.cut({1: 'text', 99: '<eos>', 0: '<pad>', 98: '<special>'}, raw)

    def test_lossy_no_tag_projection_is_explicit(self):
        proof = self.cut({1: 'text\ufffd'}, [1, 99])
        self.assertEqual(proof['kept_token_ids'], [1])
        self.assertEqual(proof['decoded_full_generation'], 'text\ufffd')
        self.assertEqual(proof['full_decode_replacement_count'], 1)
        self.assertEqual(proof['retained_decode_replacement_count'], 1)
        self.assertEqual(proof['decode_projection_role'], bt.DECODE_PROJECTION_ROLE)
        self.assertIsNone(proof['closing_tag'])

    def test_original_token_whitespace_proof_is_checked(self):
        proof = self.cut({20: '<answer>A</answer> \n'}, [20, 99])
        trace, _ = tiny_outer_fixture()
        turn = trace['trajectories'][0]['turns'][0]
        record = {key: turn[key] for key in bt._NATIVE_FIELDS}
        kept = bt.make_turn(record, trajectory_id=turn['trajectory_id'], turn_index=0,
                            kept_token_count=1, special_token_ids=[0, 99], prefix_cut=proof)
        self.assertEqual(kept['raw_generated_token_ids'], [20, 99])
        proof['retained_trailing_whitespace'] = 'changed'
        with self.assertRaises(bt.BehaviorTraceError):
            bt.make_turn(record, trajectory_id=turn['trajectory_id'], turn_index=0,
                         kept_token_count=1, special_token_ids=[0, 99], prefix_cut=proof)

    def test_prefix_proof_token_and_length_types_reject_boolean_aliases(self):
        proof = self.cut({1: '<answer>A</answer>'}, [1])
        trace, _ = tiny_outer_fixture()
        turn = trace['trajectories'][0]['turns'][0]
        record = {key: turn[key] for key in bt._NATIVE_FIELDS}
        record.update(raw_generated_token_ids=[1], raw_generated_token_count=1,
                      selected_logprobs=[-0.2], finish_reason='length')
        for field, value in (('kept_token_ids', [True]), ('raw_generated_token_count', True)):
            changed = dict(proof, **{field: value})
            with self.subTest(field=field), self.assertRaises(bt.BehaviorTraceError):
                bt.make_turn(record, trajectory_id=turn['trajectory_id'], turn_index=0,
                             kept_token_count=1, special_token_ids=[0, 99], prefix_cut=changed)


    def test_atomic_suffix_executor_stays_original_first_action(self):
        # Synthetic fragment-to-ID fixtures, not actual Qwen/GPU tokens.
        cases = ('<search>q</search>tail', '<answer>A</answer>tail',
                 '<answer>A</answer><search>q</search>tail',
                 '<search>q</search><answer>A</answer>tail')
        for text in cases:
            with self.subTest(text=text):
                proof = self.cut({20: text}, [20, 99])
                original = original_upstream_manager()
                _, execution = original._postprocess_responses([text])
                self.assertEqual(proof['execution_text'], execution[0])
                self.assertEqual(proof['decoded_retained_prefix'], text)
                self.assertEqual(proof['kept_token_ids'], [20])
                self.assertEqual(original.postprocess_predictions([proof['execution_text']]),
                                 original.postprocess_predictions(execution))

    def test_new_suffix_proof_tampering_and_legacy_whitespace(self):
        trace, _ = tiny_outer_fixture()
        turn = trace['trajectories'][0]['turns'][0]
        record = {key: turn[key] for key in bt._NATIVE_FIELDS}
        make = lambda proof: bt.make_turn(record, trajectory_id=turn['trajectory_id'], turn_index=0,
            kept_token_count=1, special_token_ids=[0, 99], prefix_cut=proof)
        proof = self.cut({20: '<search>q</search>tail'}, [20, 99])
        make(proof)
        for field, value in (('retained_token_boundary_suffix', 'changed'),
                             ('token_boundary_suffix_classification', 'whitespace'),
                             ('retained_trailing_whitespace', 'tail'),
                             ('token_boundary_whitespace_retained', True),
                             ('execution_text', '<search>fake</search>'),
                             ('decoded_retained_prefix', '<search>q</search>')):
            with self.subTest(field=field), self.assertRaises(bt.BehaviorTraceError):
                make(dict(proof, **{field: value}))
        legacy = self.cut({20: '<answer>A</answer> \n'}, [20, 99])
        for name in ('retained_token_boundary_suffix', 'token_boundary_suffix_classification',
                     'decode_projection_role', 'decoded_full_generation', 'full_decode_replacement_count',
                     'retained_decode_replacement_count'):
            del legacy[name]
        make(legacy)
        legacy['retained_trailing_whitespace'] = 'tail'
        legacy['decoded_retained_prefix'] = legacy['execution_text'] + 'tail'
        with self.assertRaises(bt.BehaviorTraceError): make(legacy)

    def test_atomic_suffix_future_context_lp_and_mask_preserved(self):
        trace, bindings = tiny_outer_fixture()
        # Token20 includes closing+suffix; it remains one complete action token.
        first = trace['trajectories'][0]['turns'][0]
        first['prefix_cut'] = self.cut({20: '<search>q</search>tail'}, [20, 99])
        report = bt.project_sampling_logprobs(trace, **bindings)
        self.assertTrue(report['audit']['context_verified'])
        self.assertEqual(report['sampling_log_probs'][0], [-0.2, 0.0, -0.30000000000000004, 0.0])
        self.assertEqual(report['sampling_mask'][0], [True, False, True, False])
        self.assertEqual(trace['trajectories'][0]['turns'][1]['actual_request_token_ids'], [10, 11, 20, 30])
        trace['trajectories'][0]['turns'][1]['actual_request_token_ids'] = [10, 11, 777, 30]
        with self.assertRaises(bt.BehaviorTraceError): bt.audit_outer(trace, **bindings)

    def test_atomic_answer_tail_is_not_cleaned_for_strict_reward(self):
        from search_r1.qa_protocol import extract_final_answer
        from search_g1.signals import contexts
        proof = self.cut({20: '<answer>A</answer>tail'}, [20, 99])
        full = proof['decoded_retained_prefix']
        self.assertIsNone(extract_final_answer([full]))
        self.assertIsNone(contexts('CPU fixture prompt', full, [[0, len(full)]]))
        self.assertEqual(extract_final_answer([proof['execution_text']]), 'A')
        # The last value is only the executor projection, never the reward span.

    def test_suffix_does_not_relax_eos_special_or_unicode_boundary(self):
        proof = self.cut({20: '<search>q</search>tail'}, [20, 99])
        self.assertTrue(proof['trailing_eos_excluded'])
        for raw in ([99, 20], [98, 20], [0, 20]):
            with self.subTest(raw=raw), self.assertRaises(bt.BehaviorTraceError):
                self.cut({20: '<search>q</search>tail', 99: '<eos>', 98: '<special>', 0: '<pad>'}, raw)
        def unstable(ids):
            return '<answer>A</answer>tail' if len(ids) == 1 else '<answer>B</answer>tailmore'
        with self.assertRaisesRegex(bt.BehaviorTraceError, 'ambiguous'):
            bt.raw_prefix_to_first_closing_tag([20, 21], unstable, eos_token_id=99, pad_token_id=0,
                                              special_token_ids=[0, 99])
        proof = self.cut({20: '<answer>A</answer>\ufffd'}, [20])
        self.assertEqual(proof['retained_decode_replacement_count'], 1)
        self.assertEqual(proof['retained_token_boundary_suffix'], '\ufffd')


    def test_stable_lossy_projection_keeps_prefix_does_not_extend_to_repair(self):
        proof = self.cut({20: '<search>q\ufffd</search> ', 21: 'later\ufffd'}, [20, 21, 99])
        self.assertEqual(proof['kept_token_ids'], [20])
        self.assertEqual(proof['execution_text'], '<search>q\ufffd</search>')
        self.assertEqual((proof['full_decode_replacement_count'], proof['retained_decode_replacement_count']), (2, 1))
        original = original_upstream_manager()
        _, execution = original._postprocess_responses([proof['decoded_full_generation']])
        self.assertEqual(execution, [proof['execution_text']])

    def test_synthetic_byte_projection_keeps_invalid_bytes_but_never_extends_to_heal(self):
        # Synthetic byte fragments are not Qwen vocabulary/GPU observations.
        stable = {20: b'<search>q\xff</search> ', 21: b'later'}
        decode = lambda ids: b''.join(stable[token] for token in ids).decode('utf-8', errors='replace')
        proof = bt.raw_prefix_to_first_closing_tag([20, 21], decode, eos_token_id=99,
                                                pad_token_id=0, special_token_ids=[0, 99])
        self.assertEqual(proof['kept_token_ids'], [20])
        self.assertEqual(proof['execution_text'], '<search>q\ufffd</search>')
        self.assertEqual(proof['retained_decode_replacement_count'], 1)
        healing = {20: b'<search>q</search>\xc3', 21: b'\xa9'}
        decode = lambda ids: b''.join(healing[token] for token in ids).decode('utf-8', errors='replace')
        self.assertIn('\ufffd', decode([20]))
        self.assertNotIn('\ufffd', decode([20, 21]))
        with self.assertRaisesRegex(bt.BehaviorTraceError, 'ambiguous_decoding_boundary'):
            bt.raw_prefix_to_first_closing_tag([20, 21], decode, eos_token_id=99,
                                             pad_token_id=0, special_token_ids=[0, 99])

    def test_lossy_projection_census_role_fullprefix_and_no_tag_are_verified(self):
        trace, _ = tiny_outer_fixture(); turn = trace['trajectories'][0]['turns'][0]
        record = {key: turn[key] for key in bt._NATIVE_FIELDS}
        make = lambda proof: bt.make_turn(record, trajectory_id=turn['trajectory_id'], turn_index=0,
            kept_token_count=1, special_token_ids=[0, 99], prefix_cut=proof)
        proof = self.cut({20: '<search>q\ufffd</search>tail'}, [20, 99]); make(proof)
        for field, value in (('decode_projection_role', 'lossless'),
                             ('full_decode_replacement_count', 0),
                             ('retained_decode_replacement_count', True),
                             ('retained_decode_replacement_count', -1),
                             ('decoded_full_generation', '<search>changed\ufffd</search>tail')):
            with self.subTest(field=field), self.assertRaises(bt.BehaviorTraceError):
                make(dict(proof, **{field: value}))
        for remove in (('decode_projection_role',),
                       ('decode_projection_role', 'decoded_full_generation', 'full_decode_replacement_count',
                        'retained_decode_replacement_count')):
            changed = copy.deepcopy(proof)
            for field in remove: del changed[field]
            with self.subTest(remove=remove), self.assertRaises(bt.BehaviorTraceError): make(changed)
        # Claiming no tag cannot swallow the original action boundary.
        changed = dict(proof, closing_tag=None, execution_text=proof['decoded_retained_prefix'],
                       retained_token_boundary_suffix='', token_boundary_suffix_classification='none')
        with self.assertRaises(bt.BehaviorTraceError): make(changed)
        no_tag = self.cut({20: 'text\ufffd'}, [20, 99]); make(no_tag)
        changed = dict(no_tag, decoded_full_generation='text\ufffdextra')
        with self.assertRaises(bt.BehaviorTraceError): make(changed)

    def test_lossy_retained_token_still_preserves_lp_mask_and_next_raw_context(self):
        trace, bindings = tiny_outer_fixture()
        trace['trajectories'][0]['turns'][0]['prefix_cut'] = self.cut({20: '<search>q\ufffd</search>tail'}, [20, 99])
        projected = bt.project_sampling_logprobs(trace, **bindings)
        self.assertEqual(projected['sampling_log_probs'][0], [-.2, 0., -.30000000000000004, 0.])
        self.assertEqual(projected['sampling_mask'][0], [True, False, True, False])
        self.assertEqual(trace['trajectories'][0]['turns'][1]['actual_request_token_ids'], [10, 11, 20, 30])


class OuterProtocolTests(unittest.TestCase):
    def setUp(self):
        self.trace, self.bindings = tiny_outer_fixture()

    def reject(self, expression=None):
        with self.assertRaises(bt.BehaviorTraceError):
            (expression or (lambda: bt.audit_outer(self.trace, **self.bindings)))()

    def test_complete_outer_and_private_projection(self):
        projected = bt.project_sampling_logprobs(self.trace, **self.bindings)
        report = projected['audit']
        self.assertEqual(report, bt.audit_outer(self.trace, **self.bindings))
        self.assertEqual(report['included_token_count'], 5120)
        self.assertEqual(report['parent_count'], 512)
        self.assertEqual(report['response_count'], 2560)
        self.assertTrue(report['context_verified'])
        for key in ('production_admission', 'healthy_checkpoint_accepted', 'physical_execution_authenticated'):
            self.assertIs(report[key], False)
        self.assertEqual(projected['sampling_mask'][0], [True, False, True, False])
        self.assertEqual(projected['sampling_log_probs'][0], [-0.2, 0.0, -(0.2 + 0.1), 0.0])
        self.assertNotIn('raw_generated_token_ids', json.dumps(report))
        self.assertNotIn('selected_logprobs', json.dumps(report))
        self.assertEqual(report['trace_sha256'], hashlib.sha256(bt.canonical_bytes(self.trace)).hexdigest())

    def test_last_trajectory_context_error_rejects_whole_original_outer(self):
        self.trace['trajectories'][-1]['turns'][1]['actual_request_token_ids'][-1] = 31
        self.reject()

    def test_raw_count_cannot_be_inferred_from_pad_or_eos(self):
        turn = self.trace['trajectories'][-1]['turns'][0]
        turn['raw_generated_token_count'] = 1
        self.reject()
        turn['raw_generated_token_count'] = 2
        turn['raw_generated_token_ids'] = [20, 0]
        turn['kept_token_count'] = 2
        self.reject()

    def test_actual_sampler_token_budget_is_bound_not_only_kept_prefix(self):
        turn = self.trace['trajectories'][0]['turns'][0]
        turn['raw_generated_token_ids'] = [20] * 501
        turn['raw_generated_token_count'] = 501
        turn['selected_logprobs'] = [-0.2] * 501
        turn['kept_token_count'] = 501
        self.reject()
        for value in (501, True, 500.0):
            turn['sampling']['max_tokens'] = value
            self.reject()

    def test_native_finish_reason_must_prove_excluded_eos(self):
        self.trace['trajectories'][0]['turns'][0]['finish_reason'] = 'length'
        self.reject()

    def test_tokenizer_ids_cannot_drift_between_rows_or_turns(self):
        self.trace['trajectories'][-1]['turns'][1]['pad_token_id'] = 98
        self.reject()
        self.trace['trajectories'][-1]['turns'][1]['pad_token_id'] = 0
        self.trace['trajectories'][-1]['turns'][1]['special_token_ids'].append(97)
        self.reject()

    def test_same_eos_and_pad_use_real_raw_count_not_id_filtering(self):
        for trajectory in self.trace['trajectories']:
            for turn in trajectory['turns']:
                turn['pad_token_id'] = 99
        self.assertEqual(bt.audit_outer(self.trace, **self.bindings)['included_token_count'], 5120)
        self.trace['trajectories'][-1]['turns'][1]['raw_generated_token_count'] = 1
        self.reject()

    def test_retokenization_window_drop_and_actor_position_changes_reject(self):
        self.trace['actor_rows'][-1]['input_ids'][3] = 22
        self.reject()
        self.trace['actor_rows'][-1]['input_ids'][3] = 20
        self.trace['trajectories'][-1]['turns'][1]['actual_request_token_ids'] = [11, 20, 30]
        self.reject()
        self.trace['trajectories'][-1]['turns'][1]['actual_request_token_ids'] = [10, 11, 20, 30]
        self.trace['actor_rows'][-1]['position_ids'][5] = 5
        self.reject()

    def test_mask_cannot_optimize_observation_or_hide_model(self):
        self.trace['actor_rows'][-1]['info_mask'][4] = 1
        self.reject()
        self.trace['actor_rows'][-1]['info_mask'][4] = 0
        self.trace['actor_rows'][-1]['info_mask'][3] = 0
        self.reject()

    def test_correction_same_context_and_environment_order_are_enforced(self):
        turn = self.trace['trajectories'][0]['turns'][0]
        turn['correction_token_ids'] = turn['observation_token_ids']
        turn['observation_token_ids'] = []
        self.assertTrue(bt.audit_outer(self.trace, **self.bindings)['context_verified'])
        turn['correction_token_ids'] = [31]
        self.reject()
        turn['observation_token_ids'] = [30]
        self.reject()

    def test_legitimate_actor_reordering_preserves_projection_identity(self):
        rows = list(reversed(self.trace['actor_rows']))
        projected = bt.project_sampling_logprobs(self.trace, actor_rows=rows, **self.bindings)
        self.assertEqual(projected['trajectory_ids'][0], self.bindings['expected_trajectory_ids'][-1])
        raw = self.trace['trajectories'][-1]['turns']
        self.assertEqual(projected['sampling_log_probs'][0],
                         [raw[0]['selected_logprobs'][0], 0.0, raw[1]['selected_logprobs'][0], 0.0])
        self.assertEqual(projected['audit']['actor_rows_sha256'], bt.document_sha256(rows))

    def test_missing_duplicate_group_and_filtering_reject(self):
        self.trace['actor_rows'][-1] = copy.deepcopy(self.trace['actor_rows'][0])
        self.reject()
        self.trace, self.bindings = tiny_outer_fixture()
        self.trace['trajectories'][5]['parent_occurrence_index'] = 0
        self.reject()
        self.trace['trajectories'].pop()
        self.reject()

    def test_duplicate_parent_labels_still_use_occurrence_groups(self):
        parents = ['same-question'] * 512
        self.trace['parent_group_ids'] = parents
        self.bindings['expected_parent_group_ids'] = parents
        self.assertEqual(bt.audit_outer(self.trace, **self.bindings)['response_count'], 2560)

    def test_bound_run_source_job_sampling_and_policy_are_not_labels_only(self):
        for field in ('run_id', 'slurm_job_id', 'policy_snapshot_id'):
            original = self.trace[field]
            self.trace[field] = 'wrong'
            self.reject()
            self.trace[field] = original
        self.trace['trajectories'][-1]['turns'][1]['sampling']['top_p'] = 0.95
        self.reject()
        self.trace['trajectories'][-1]['turns'][1]['sampling']['top_p'] = 1.0
        self.trace['source_sha256'][next(iter(self.trace['source_sha256']))] = 'b' * 64
        # The constructor does not share the bindings source object.
        self.reject()

    def test_nonfinite_or_missing_selected_lp_rejects_in_last_row(self):
        selected = self.trace['trajectories'][-1]['turns'][1]['selected_logprobs']
        selected[0] = float('nan')
        self.reject()
        selected[0] = -0.3
        selected.pop()
        self.reject()

    def test_sampling_semantics_and_native_transport_ids_are_strict(self):
        turn = self.trace['trajectories'][0]['turns'][0]
        record = {key: copy.deepcopy(turn[key]) for key in bt._NATIVE_FIELDS}
        record['row_id'] = 'wrong'
        self.reject(lambda: bt.make_turn(record, trajectory_id=turn['trajectory_id'], turn_index=0,
                                         kept_token_count=1, special_token_ids=[0, 99]))
        record['row_id'] = turn['trajectory_id']
        record['sampling']['logprob_semantics'] = 'HF_recomputed'
        self.reject(lambda: bt.make_turn(record, trajectory_id=turn['trajectory_id'], turn_index=0,
                                         kept_token_count=1, special_token_ids=[0, 99]))
        record['sampling']['logprob_semantics'] = bt.LOGPROB_SEMANTICS
        record['raw_generated_token_count'] = True
        self.reject(lambda: bt.make_turn(record, trajectory_id=turn['trajectory_id'], turn_index=0,
                                         kept_token_count=1, special_token_ids=[0, 99]))

    def test_turns_after_final_and_untraced_actor_tail_reject(self):
        self.trace['trajectories'][0]['turns'][0]['observation_token_ids'] = []
        self.trace['trajectories'][0]['turns'][0]['final_turn'] = True
        self.reject()
        self.trace, self.bindings = tiny_outer_fixture()
        self.trace['actor_rows'][0]['attention_mask'][-1] = 1
        self.trace['actor_rows'][0]['info_mask'][-1] = 1
        self.trace['actor_rows'][0]['position_ids'][-1] = 5
        self.trace['actor_rows'][0]['input_ids'][-1] = 22
        self.reject()


class PrivateEvidenceTests(unittest.TestCase):
    def test_file_private_exclusive_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve() / 'private'
            parent.mkdir(mode=0o700)
            path = parent / 'raw.json'
            bt._exclusive_private_file(path, b'{"cpu_fixture":true}')
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(path.read_bytes(), b'{"cpu_fixture":true}')
            with self.assertRaises(FileExistsError):
                bt._exclusive_private_file(path, b'replace')
            link = Path(temporary).resolve() / 'link'
            link.symlink_to(parent, target_is_directory=True)
            with self.assertRaises(bt.BehaviorTraceError):
                bt._exclusive_private_file(link / 'new.json', b'bad')

    def test_evidence_nonfinite_marker_retains_raw_failure_tokens(self):
        raw = dict(raw_generated_token_ids=[20, 99], selected_logprobs=[float('nan'), -float('inf')])
        converted, transformed = bt._evidence_value(raw)
        self.assertTrue(transformed)
        self.assertEqual(converted['raw_generated_token_ids'], [20, 99])
        self.assertEqual(converted['selected_logprobs'][0], {'__behavior_trace_nonfinite_float__': 'nan'})
        self.assertEqual(converted['selected_logprobs'][1], {'__behavior_trace_nonfinite_float__': '-inf'})
        self.assertIn('raw_generated_token_ids', json.loads(bt.canonical_bytes(converted)))

    def test_non_native_namespace_rejects_before_artifact_mutation(self):
        trace, bindings = tiny_outer_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(bt.BehaviorTraceError):
                bt.write_private_evidence(temporary, trace)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            with self.assertRaises(bt.BehaviorTraceError):
                bt.write_private_journal(temporary, trace, **bindings)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_writer_preserves_evidence_before_audit_rejection(self):
        trace, bindings = tiny_outer_fixture()
        trace['trajectories'][-1]['turns'][1]['actual_request_token_ids'][-1] = 31
        evidence = {'private_trace_sha256': bt.document_sha256(trace), 'serialization_transformed': False}
        # Only the native filesystem boundary is isolated. The complete real
        # CPU protocol audit runs unchanged and must reject after preservation.
        with patch.object(bt, 'write_private_evidence', return_value=evidence) as writer:
            with self.assertRaises(bt.BehaviorTraceError):
                bt.write_private_journal('/not-an-executed-cluster-fixture', trace, **bindings)
            writer.assert_called_once_with('/not-an-executed-cluster-fixture', trace)


if __name__ == '__main__':
    unittest.main()
