"""Actual DataProto/controller CPU tests with explicit simulated native records.

The tiny tokenizer and worker perform no model/GPU kernels. These tests verify
the real controller's tensor/identity/context operations, not native execution,
HF probabilities, physical eight-rank resources, or healthy-policy admission.
"""
import ast
import copy
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from verl import DataProto
from reproduction.common8 import behavior_trace as bt
from reproduction.common8 import generation_behavior as controller
from reproduction.common8.test_behavior_trace import original_upstream_manager
from search_g1.signals import contexts
from search_r1.qa_protocol import extract_final_answer
from verl.utils.reward_score.qa_em import compute_score_generated_em
from reproduction.common8.calibration_evaluation import _f1, rows_from_batch
from pathlib import Path


def sampler_fixture():
    return dict(n=1, best_of=1, do_sample=True, temperature=1.0, top_p=1.0,
                top_k=-1, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                repetition_penalty=1.0, ignore_eos=False, logprobs=1, min_tokens=0,
                logits_processors=[], logprob_semantics=bt.LOGPROB_SEMANTICS,
                max_tokens=500, stop=None, stop_token_ids=None, seed=None,
                logit_bias=None, allowed_token_ids=None, guided_decoding=None,
                truncate_prompt_tokens=None)


class TokenizerFixture:
    pad_token_id = 0
    eos_token_id = 99
    pad_token = '<pad>'
    all_special_ids = [0, 98, 99]
    fragments = {0: '<pad>', 98: '<special>', 99: '<eos>',
                 20: '<search>query', 21: '</search> \n', 22: 'discarded tail',
                 23: '<answer>A', 24: '</answer>', 25: 'invalid action',
                 26: '<answer>A</answer>secret'}

    def __init__(self):
        self.decode_calls = []
        self.encode_calls = 0

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        self.decode_calls.append(list(ids))
        return ''.join(self.fragments.get(value, 'P') for value in ids)

    def encode(self, *args, **kwargs):
        self.encode_calls += 1
        raise AssertionError('Controller must not retokenize model output')

    __call__ = encode


class WorkerFixture:
    """Return simulated sampling records through real TensorDict/DataProto."""
    def __init__(self, script=None, failure=None):
        self.script = script or (lambda ordinal, identity: [20, 21, 22, 99] if ordinal == 0 else [23, 24, 99])
        self.failure = failure
        self.calls = []

    def generate_sequences(self, call):
        assert isinstance(call, DataProto)
        self.calls.append(DataProto.from_dict(
            {name: value.clone() for name, value in call.batch.items()},
            non_tensors={name: value.copy() for name, value in call.non_tensor_batch.items()},
            meta_info=copy.deepcopy(call.meta_info)))
        ordinal = call.meta_info['behavior_rollout_ordinal']
        identities = call.non_tensor_batch['behavior_row_id'].tolist()
        records = []
        for index, identity in enumerate(identities):
            prompt = [int(value) for value, active in zip(call.batch['input_ids'][index].tolist(),
                       call.batch['attention_mask'][index].tolist()) if active]
            raw = self.script(ordinal, identity)
            sampling = sampler_fixture()
            if call.meta_info.get('do_sample') is False:
                sampling.update(do_sample=False, temperature=0.0)
            records.append(dict(schema_version=1, request_id='cpu-native-' + identity + '-' + str(ordinal),
                row_id=identity, rollout_ordinal=ordinal, rollout_call_id=call.meta_info['behavior_call_id'],
                policy_snapshot_id=call.meta_info['policy_snapshot_id'],
                actual_request_token_ids=prompt, raw_generated_token_ids=list(raw),
                raw_generated_token_count=len(raw), selected_logprobs=[-0.25 - 0.125 * j for j in range(len(raw))],
                finish_reason='stop' if raw and raw[-1] == 99 else 'length', stop_reason=None,
                eos_token_id=99, pad_token_id=0, sampling=sampling))
        width = max(len(record['raw_generated_token_ids']) for record in records)
        response = torch.zeros((len(records), width), dtype=torch.long)
        selected = torch.zeros((len(records), width), dtype=torch.float32)
        sampled = torch.zeros((len(records), width), dtype=torch.bool)
        for index, record in enumerate(records):
            count = record['raw_generated_token_count']
            response[index, :count] = torch.tensor(record['raw_generated_token_ids'], dtype=torch.long)
            selected[index, :count] = torch.tensor(record['selected_logprobs'], dtype=torch.float32)
            sampled[index, :count] = True
        if self.failure == 'wrong-row':
            records[0]['row_id'] = 'offending-return-row'
        elif self.failure == 'wrong-token':
            response[0, 0] = 777
        elif self.failure == 'wrong-lp':
            selected[0, 0] = -2.0
        elif self.failure == 'wrong-count':
            records[0]['raw_generated_token_count'] += 1
        values = dict(responses=response, sampling_log_probs=selected, sampling_mask=sampled,
                      prompts=call.batch['input_ids'].clone(),
                      input_ids=torch.cat((call.batch['input_ids'], response), dim=1))
        values['attention_mask'] = torch.cat((call.batch['attention_mask'], sampled.long()), dim=1)
        values['position_ids'] = (values['attention_mask'].cumsum(-1) - 1) * values['attention_mask']
        if self.failure == 'wrong-size':
            values = {name: value[:-1] for name, value in values.items()}
            records = records[:-1]
        array = np.empty(len(records), dtype=object)
        array[:] = records
        return DataProto.from_dict(values, non_tensors={'sampling_behavior': array}, meta_info={})


class ManagerFixture:
    def __init__(self, script=None, failure=None, *, max_prompt_length=4096, env_length=1):
        self.tokenizer = TokenizerFixture()
        self.actor_rollout_wg = WorkerFixture(script, failure)
        self.config = SimpleNamespace(num_gpus=8, max_turns=4, no_think_rl=False,
            max_start_length=2048, max_prompt_length=max_prompt_length,
            max_response_length=500, max_obs_length=500)
        self.env_length = env_length
        self.environment_calls = []

    def execute_predictions(self, texts, pad_token, active_mask, do_search):
        self.environment_calls.append(dict(texts=list(texts), active=active_mask.tolist(), do_search=do_search))
        observations, done, valid, searches = [], [], [], []
        for text, active in zip(texts, active_mask.tolist()):
            match = re.search(r'<(search|answer)>(.*?)</\1>', text, re.DOTALL)
            action = match.group(1) if match else None
            if not active:
                observations.append(''); done.append(1); valid.append(0); searches.append(0)
            elif action == 'answer':
                observations.append(''); done.append(1); valid.append(1); searches.append(0)
            elif action == 'search':
                observations.append('evidence' if do_search else ''); done.append(0); valid.append(1); searches.append(1)
            else:
                observations.append('correction'); done.append(0); valid.append(0); searches.append(0)
        return observations, done, valid, searches

    def _process_next_obs(self, texts):
        rows = [[30 if text == 'evidence' else 31] * self.env_length if text else [] for text in texts]
        width = max(map(len, rows))
        result = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            if row:
                result[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        return result


def generation_fixture(prompts=None, *, validation=False, do_sample=True):
    prompts = prompts or [[10, 11], [10, 11]]
    padded = controller._left_pad(prompts, 0)
    identities = ['cpu-traj-' + str(index) for index in range(len(prompts))]
    identity = dict(run_id='cpu-controller-fixture', source_sha256={'cpu_fixture.py': 'a' * 64},
                    slurm_job_id='not-native-cpu-fixture', logical_update=0 if validation else 1,
                    policy_snapshot_id='cpu-policy0', parent_group_ids=['cpu-parent'],
                    trajectory_ids=identities, validation=validation)
    batch = DataProto.from_dict(padded, non_tensors={'behavior_row_id': identities},
        meta_info=dict(behavior_identity=identity, recompute_log_prob=False, do_sample=do_sample,
                       validate=validation,
                       eos_token_id=99, pad_token_id=0))
    return batch, batch.batch['input_ids'].clone()


class ControllerCpuTests(unittest.TestCase):
    def run_success(self, manager=None, prompts=None):
        manager = manager or ManagerFixture()
        batch, initial = generation_fixture(prompts)
        with patch('reproduction.common8.behavior_runtime.failed_generation_evidence') as failure:
            result = controller.run(manager, batch, initial)
            failure.assert_not_called()
        self.assertIsInstance(result, DataProto)
        self.assertEqual(len(result), len(batch))
        self.assertEqual(result.non_tensor_batch['behavior_trajectory_id'].tolist(),
                         batch.non_tensor_batch['behavior_row_id'].tolist())
        return manager, batch, result

    def run_failure(self, manager, expression, prompts=None):
        batch, initial = generation_fixture(prompts)
        with patch('reproduction.common8.behavior_runtime.failed_generation_evidence') as preserved:
            with self.assertRaisesRegex(bt.BehaviorTraceError, expression):
                controller.run(manager, batch, initial)
            preserved.assert_called_once()
            identity, records, reason = preserved.call_args.args
            self.assertEqual(identity, batch.meta_info['behavior_identity'])
            self.assertEqual(reason, 'BehaviorTraceError')
            self.assertTrue(records)
            return records

    def test_active_one_pads_to_eight_with_unique_ids_and_trims_every_field(self):
        def script(ordinal, identity):
            if ordinal or identity == 'cpu-traj-0':
                return [23, 24, 99]
            return [20, 21, 99]
        manager, batch, result = self.run_success(ManagerFixture(script))
        self.assertEqual(len(manager.actor_rollout_wg.calls), 2)
        calls = manager.actor_rollout_wg.calls
        for call in calls:
            self.assertEqual(len(call), 8)
            ids = call.non_tensor_batch['behavior_row_id'].tolist()
            self.assertEqual(len(set(ids)), 8)
            self.assertIs(call.meta_info['recompute_log_prob'], False)
        self.assertEqual(calls[1].non_tensor_batch['behavior_row_id'][0], 'cpu-traj-1')
        self.assertTrue(all(identity.startswith('padding-') for identity in
                            calls[1].non_tensor_batch['behavior_row_id'].tolist()[1:]))
        turns = result.non_tensor_batch['behavior_turns'].tolist()
        self.assertEqual([len(row) for row in turns], [1, 2])
        self.assertEqual(turns[1][1]['trajectory_id'], 'cpu-traj-1')
        self.assertEqual(result.batch['sampling_log_probs'].shape[0], 2)
        self.assertEqual(result.meta_info['behavior_identity'], batch.meta_info['behavior_identity'])

    def test_direct_generate_padded_active_one_preserves_actual_records_and_float_lp(self):
        batch, _ = generation_fixture([[10, 11]])
        batch.meta_info.update(behavior_call_id='cpu-call', behavior_rollout_ordinal=0,
                               policy_snapshot_id='cpu-policy0')
        manager = ManagerFixture()
        captured = []
        result = controller.generate_padded(manager, batch, captured)
        self.assertEqual(len(manager.actor_rollout_wg.calls[0]), 8)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result.non_tensor_batch['sampling_behavior']), 1)
        self.assertEqual(len(captured), 8)
        self.assertEqual(result.non_tensor_batch['sampling_behavior'][0]['row_id'], 'cpu-traj-0')
        self.assertEqual(result.batch['sampling_log_probs'].dtype, torch.float32)
        self.assertEqual(result.batch['sampling_mask'].dtype, torch.bool)

    def test_raw_prefix_not_encoded_observation_and_context_are_exact(self):
        manager, _, result = self.run_success()
        self.assertEqual(manager.tokenizer.encode_calls, 0)
        self.assertEqual(result.batch['responses'][0].tolist(), [20, 21, 30, 23, 24])
        self.assertEqual(result.batch['info_mask'][0, -5:].tolist(), [1, 1, 0, 1, 1])
        self.assertEqual(result.batch['sampling_mask'][0].tolist(), [True, True, False, True, True])
        self.assertEqual(result.batch['sampling_log_probs'].dtype, torch.float32)
        self.assertEqual(result.batch['sampling_log_probs'][0].tolist(), [-0.25, -0.375, 0.0, -0.25, -0.375])
        turns = result.non_tensor_batch['behavior_turns'][0]
        self.assertEqual(turns[0]['raw_generated_token_ids'], [20, 21, 22, 99])
        self.assertEqual(turns[0]['kept_token_count'], 2)
        self.assertEqual(turns[0]['prefix_cut']['retained_trailing_whitespace'], ' \n')
        self.assertEqual(turns[0]['observation_token_ids'], [30])
        self.assertEqual(turns[0]['correction_token_ids'], [])
        self.assertEqual(turns[1]['actual_request_token_ids'], [10, 11, 20, 21, 30])
        for index, identity in enumerate(result.non_tensor_batch['behavior_trajectory_id']):
            row = bt.make_actor_row(trajectory_id=identity, prompt_width=result.batch['prompts'].shape[1],
                **{key: result.batch[key][index].tolist() for key in
                   ('input_ids', 'attention_mask', 'info_mask', 'position_ids')})
            count, projected, mask = bt._audit_trajectory(result.non_tensor_batch['behavior_turns'][index],
                row, result.meta_info['behavior_sampling'], 'cpu-policy0')
            self.assertEqual(count, 4)
            self.assertEqual(projected, result.batch['sampling_log_probs'][index].tolist())
            self.assertEqual(mask, result.batch['sampling_mask'][index].tolist())

    def test_invalid_action_correction_is_context_and_never_actor_token(self):
        script = lambda ordinal, identity: [25, 99] if ordinal == 0 else [23, 24, 99]
        _, _, result = self.run_success(ManagerFixture(script))
        first, second = result.non_tensor_batch['behavior_turns'][0]
        self.assertEqual(first['observation_token_ids'], [])
        self.assertEqual(first['correction_token_ids'], [31])
        self.assertEqual(second['actual_request_token_ids'], [10, 11, 25, 31])
        self.assertEqual(result.batch['responses'][0].tolist(), [25, 31, 23, 24])
        self.assertEqual(result.batch['sampling_log_probs'][0].tolist(), [-0.25, 0.0, -0.25, -0.375])
        self.assertEqual(result.batch['sampling_mask'][0].tolist(), [True, False, True, True])

    def test_first_request_and_final_prompt_share_the_exact_2048_crop(self):
        prompt = list(range(100, 2150))
        script = lambda ordinal, identity: [23, 24, 99]
        manager, _, result = self.run_success(ManagerFixture(script), [prompt])
        self.assertEqual(result.batch['prompts'].shape, torch.Size([1, 2048]))
        self.assertEqual(result.batch['prompts'][0].tolist(), prompt[-2048:])
        first_request = result.non_tensor_batch['behavior_turns'][0][0]['actual_request_token_ids']
        self.assertEqual(first_request, prompt[-2048:])
        self.assertEqual(manager.actor_rollout_wg.calls[0].batch['input_ids'][0].tolist(), prompt[-2048:])
        self.assertEqual(result.batch['input_ids'][0, :2048].tolist(), first_request)
        self.assertEqual(result.batch['position_ids'][0].tolist(), list(range(2050)))

    def test_initial_greedy_validation_preserves_zero_step_and_actual_sampler_metadata(self):
        batch, initial = generation_fixture(validation=True, do_sample=False)
        manager = ManagerFixture(lambda ordinal, identity: [23, 24, 99])
        with patch('reproduction.common8.behavior_runtime.failed_generation_evidence') as failure:
            result = controller.run(manager, batch, initial)
            failure.assert_not_called()
        self.assertIsInstance(result, DataProto)
        self.assertEqual(result.meta_info['behavior_identity'], batch.meta_info['behavior_identity'])
        self.assertEqual(result.meta_info['behavior_identity']['logical_update'], 0)
        self.assertIs(result.meta_info['behavior_identity']['validation'], True)
        actual = result.meta_info['behavior_sampling']
        self.assertIs(actual['do_sample'], False)
        self.assertEqual(actual['temperature'], 0.0)
        self.assertEqual(actual['max_tokens'], 500)
        self.assertEqual(len(manager.actor_rollout_wg.calls), 1)
        call = manager.actor_rollout_wg.calls[0]
        self.assertIs(call.meta_info['do_sample'], False)
        self.assertIs(call.meta_info['validate'], True)
        self.assertIs(call.meta_info['recompute_log_prob'], False)
        self.assertEqual(result.batch['sampling_log_probs'].dtype, torch.float32)
        for turns in result.non_tensor_batch['behavior_turns']:
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]['turn_index'], 0)
            self.assertIs(turns[0]['sampling']['do_sample'], False)
            self.assertEqual(turns[0]['sampling']['temperature'], 0.0)
            self.assertIs(turns[0]['final_turn'], True)
            self.assertEqual(turns[0]['actual_request_token_ids'], [10, 11])

    def test_four_search_rounds_then_fifth_final_answer_keeps_all_context(self):
        script = lambda ordinal, identity: [20, 21, 99] if ordinal < 4 else [23, 24, 99]
        manager, _, result = self.run_success(ManagerFixture(script))
        self.assertEqual(len(manager.actor_rollout_wg.calls), 5)
        self.assertEqual([call.meta_info['behavior_rollout_ordinal'] for call in
                          manager.actor_rollout_wg.calls], [0, 1, 2, 3, 4])
        self.assertEqual([event['do_search'] for event in manager.environment_calls],
                         [True, True, True, True, False])
        self.assertEqual(result.meta_info['actual_retrieval_stats'], [4, 4])
        self.assertEqual(result.meta_info['valid_search_stats'], [4, 4])
        self.assertEqual(result.meta_info['active_mask'], [False, False])
        expected_response = [20, 21, 30] * 4 + [23, 24]
        self.assertEqual(result.batch['responses'][0].tolist(), expected_response)
        self.assertEqual(result.batch['sampling_mask'][0].tolist(), [True, True, False] * 4 + [True, True])
        turns = result.non_tensor_batch['behavior_turns'][0]
        self.assertEqual(len(turns), 5)
        for ordinal, turn in enumerate(turns):
            self.assertEqual(turn['turn_index'], ordinal)
            self.assertEqual(turn['actual_request_token_ids'], [10, 11] + [20, 21, 30] * ordinal)
            self.assertIs(turn['final_turn'], ordinal == 4)
            self.assertEqual(turn['observation_token_ids'], [30] if ordinal < 4 else [])
            self.assertEqual(turn['correction_token_ids'], [])
        row = bt.make_actor_row(trajectory_id='cpu-traj-0', prompt_width=2,
            **{key: result.batch[key][0].tolist() for key in
               ('input_ids', 'attention_mask', 'info_mask', 'position_ids')})
        count, projected, mask = bt._audit_trajectory(turns, row, result.meta_info['behavior_sampling'], 'cpu-policy0')
        self.assertEqual(count, 10)
        self.assertEqual(projected, result.batch['sampling_log_probs'][0].tolist())
        self.assertEqual(mask, result.batch['sampling_mask'][0].tolist())

    def test_rolling_overflow_rejects_entire_call_and_preserves_environment_cause(self):
        manager = ManagerFixture(max_prompt_length=5, env_length=2)
        records = self.run_failure(manager, 'rolling_window')
        self.assertEqual(len(manager.actor_rollout_wg.calls), 1)
        event = next(record for record in records if record.get('role') == 'actual_environment_event')
        self.assertEqual(event['observation_or_correction_token_ids'], [[30, 30], [30, 30]])
        self.assertEqual(event['is_search'], [1, 1])
        self.assertEqual(event['row_ids'], ['cpu-traj-0', 'cpu-traj-1'])
        self.assertEqual(sum('raw_generated_token_ids' in record for record in records), 8)

    def test_response_overflow_preserves_actual_model_and_environment_events(self):
        records = self.run_failure(ManagerFixture(max_prompt_length=3, env_length=4), 'response_prefix_budget')
        event = next(record for record in records if record.get('role') == 'actual_environment_event')
        self.assertEqual(event['observation_or_correction_token_ids'][0], [30] * 4)
        self.assertEqual(records[0]['raw_generated_token_ids'], [20, 21, 22, 99])

    def test_internal_eos_pad_and_special_reject_whole_outer_with_raw_evidence(self):
        for special in (0, 98, 99):
            with self.subTest(special=special):
                raw = [23, special, 24, 99]
                manager = ManagerFixture(lambda ordinal, identity: raw)
                records = self.run_failure(manager, 'included_special_or_pad_token')
                self.assertEqual(records[0]['raw_generated_token_ids'], raw)
                self.assertEqual(len(records), 8)
                self.assertEqual(len(manager.actor_rollout_wg.calls), 1)

    def test_answer_before_search_retains_all_tokens_lp_masks_and_original_action(self):
        raw = [23, 24, 20, 21, 99]
        manager = ManagerFixture(lambda ordinal, identity: raw)
        original = original_upstream_manager()
        manager.execute_predictions = original.execute_predictions
        _, _, result = self.run_success(manager)
        self.assertEqual(manager.tokenizer.encode_calls, 0)
        self.assertEqual(len(manager.actor_rollout_wg.calls), 1)
        kept = raw[:-1]
        self.assertEqual(result.batch['responses'][0].tolist(), kept)
        self.assertEqual(result.batch['sampling_log_probs'][0].tolist(), [-.25, -.375, -.5, -.625])
        self.assertEqual(result.batch['sampling_mask'][0].tolist(), [True] * 4)
        self.assertEqual(result.batch['info_mask'][0, -4:].tolist(), [1] * 4)
        self.assertEqual(result.batch['input_ids'][0].tolist(), [10, 11] + kept)
        self.assertEqual(result.meta_info['actual_retrieval_stats'], [0, 0])
        self.assertEqual(result.meta_info['valid_action_stats'], [1, 1])
        self.assertEqual(result.meta_info['active_mask'], [False, False])
        self.assertEqual(original.search_calls, [[]])
        turn = result.non_tensor_batch['behavior_turns'][0][0]
        self.assertEqual(turn['actual_request_token_ids'], [10, 11])
        self.assertEqual(turn['raw_generated_token_ids'], raw)
        self.assertEqual(turn['kept_token_count'], 4)
        self.assertIs(turn['final_turn'], True)
        self.assertEqual(turn['observation_token_ids'], [])
        text = turn['prefix_cut']['execution_text']
        self.assertEqual(text, '<answer>A</answer><search>query</search>')
        row = bt.make_actor_row(trajectory_id='cpu-traj-0', prompt_width=2,
            **{key: result.batch[key][0].tolist() for key in
               ('input_ids', 'attention_mask', 'info_mask', 'position_ids')})
        count, projected, mask = bt._audit_trajectory([turn], row, result.meta_info['behavior_sampling'], 'cpu-policy0')
        self.assertEqual(count, 4)
        self.assertEqual(projected, result.batch['sampling_log_probs'][0].tolist())
        self.assertEqual(mask, [True] * 4)
        # Reward uses the complete authored span, never a cleaned first answer.
        retained = turn['prefix_cut']['decoded_retained_prefix']
        self.assertIsNone(extract_final_answer([retained]))
        self.assertIsNone(contexts('', retained, [(0, len(retained))]))
        self.assertEqual(compute_score_generated_em([retained], ['A']), 0.0)
        self.assertEqual(_f1(None, ['A']), 0.0)
        result.non_tensor_batch.update(
            __g1_question_id=np.array(['cpu-question0', 'cpu-question1'], dtype=object),
            data_source=np.array(['nq', 'hotpotqa'], dtype=object),
            reward_model=np.array([{'ground_truth': {'target': ['A']}}] * 2, dtype=object))
        decoder = SimpleNamespace(decode=lambda ids: ''.join(manager.tokenizer.fragments.get(int(token), 'P') for token in ids))
        measured = rows_from_batch(result, decoder)
        for measurement in measured:
            self.assertEqual((measurement['em'], measurement['f1'], measurement['answer_present']), (0, 0.0, False))
            self.assertEqual((measurement['actual_calls'], measurement['valid_action'], measurement['turns']), (0, 1, 1))
        source = Path(__file__).resolve().parents[2] / 'verl/trainer/main_ppo_eco.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EcoRewardManager')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_score_probe')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        score, trust, kind = namespace['_score_probe'](SimpleNamespace(PENALTY=.1),
            dict(n_search=0, correct=False, answer=None), None)
        self.assertEqual((score, trust, kind), (-.1, 0.0, None))

    def test_inside_tag_token_nonwhitespace_retains_failure_evidence(self):
        raw = [26, 99]
        records = self.run_failure(ManagerFixture(lambda ordinal, identity: raw), 'nonwhitespace')
        self.assertEqual(records[0]['raw_generated_token_ids'], raw)

    def test_return_row_and_size_failures_preserve_offending_native_return(self):
        for failure, expression, count in (('wrong-row', 'row_identity', 8),
                                          ('wrong-size', 'row_count', 7)):
            with self.subTest(failure=failure):
                records = self.run_failure(ManagerFixture(failure=failure), expression)
                self.assertEqual(len(records), count)
                if failure == 'wrong-row':
                    self.assertEqual(records[0]['row_id'], 'offending-return-row')
                self.assertEqual(records[0]['raw_generated_token_ids'], [20, 21, 22, 99])

    def test_raw_token_lp_and_count_corruption_preserve_all_returned_rows(self):
        for failure, expression in (('wrong-token', 'raw_tensor'), ('wrong-lp', 'probability_transport'),
                                    ('wrong-count', 'raw_count')):
            with self.subTest(failure=failure):
                records = self.run_failure(ManagerFixture(failure=failure), expression)
                self.assertEqual(len(records), 8)
                self.assertEqual(records[0]['raw_generated_token_ids'], [20, 21, 22, 99])


if __name__ == '__main__':
    unittest.main()
