"""New CPU interfaces only: real Torch hooks and source prefix packing, no GPU/HF.

The fake forward deliberately supplies known probabilities. It verifies the
observer/orchestrator mapping and rejection boundaries, not numerical parity.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from search_r1.baselines.action_prefix_forward import prepare_action_prefix_micro, forward_prepared_action_prefix
from raw_hf_observation_v1 import capture_existing_flat_calls, unpack_complete_action_calls, validate_actual_model_inputs
from raw_alignment_v1 import collect_raw_actions, validate_real_sample
from raw_runner_observation_v1 import observe_actual_runner, validate_actual_runner_observation

PAD, EOS = 14, 15


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False):
        return input_ids


class Actor:
    use_remove_padding = True
    ulysses_sequence_parallel_size = 1

    def __init__(self):
        self.actor_module = Model()

    def _forward_flat_micro_batch(self, micro, temperature):
        valid = micro['attention_mask'].bool()
        self.actor_module(input_ids=micro['input_ids'][valid].reshape(1, -1), attention_mask=None,
            position_ids=micro['position_ids'][valid].reshape(1, -1), use_cache=False)
        return torch.zeros_like(micro['responses'], dtype=torch.float32), -micro['responses'].float() / 100


def inputs(n):
    responses = torch.tensor([[12, PAD, 99, 13]] * n)
    prefixes = torch.full((n, 4, 4096), PAD, dtype=torch.long)
    actions = torch.full((n, 4, 500), PAD, dtype=torch.long)
    prefixes[:, 0, :2] = torch.tensor([10, 11])
    prefixes[:, 1, :4] = torch.tensor([10, 11, 80, 99])  # Canonical context, not raw 12/PAD.
    actions[:, 0, :2] = torch.tensor([12, PAD])
    actions[:, 1, 0] = 13
    return dict(responses=responses, actor_action_prefix_ids=prefixes, actor_action_tokens=actions,
        actor_action_lengths=torch.tensor([[2, 1, 0, 0]] * n),
        actor_action_prefix_lengths=torch.tensor([[2, 4, 0, 0]] * n),
        actor_action_output_offsets=torch.tensor([[0, 3, -1, -1]] * n), actor_action_count=torch.full((n,), 2),
        loss_mask=torch.tensor([[1, 1, 0, 1]] * n), step_ids=torch.tensor([[0, 0, -1, 1]] * n))


def provenance():
    return [dict(step_id=0, generation_prefix=[10, 11], actor_action_tokens=[12, PAD], actor_output_offset=0, sampled_length=2),
            dict(step_id=1, generation_prefix=[10, 11, 80, 99], actor_action_tokens=[13], actor_output_offset=3, sampled_length=1)]


def complete_raw(rank, role):
    data = inputs(4)
    actor, prepared = Actor(), prepare_action_prefix_micro(data, PAD, EOS)
    with capture_existing_flat_calls(actor) as calls:
        _, logp = forward_prepared_action_prefix(prepared, 1., actor._forward_flat_micro_batch)
    return dict(schema='RAW_ACTION_HF_FORWARD_OBSERVATION_V1', role=role, rank=rank, pid=100+rank,
        code_commit='c'*40, binding_sha256='d'*64, actor_token_identity='sampled_action_prefix_v1',
        actual_pad_token_id=PAD, actual_eos_token_id=EOS, temperature=1.,
        input_tensors=data, output_log_probs=logp, flat_calls=calls, micro_batch_size=4,
        trajectory_sources=list(range(rank*4, rank*4+4)), trajectory_question_ids=[i//5 for i in range(rank*4,rank*4+4)],
        action_prefix_records=[provenance() for _ in range(4)],
        observer_errors=[])


def sample_raw(rank, serial):
    prefix = [10, 11] if serial == 0 else [10, 11, 80, 99]
    chosen = [12, PAD] if serial == 0 else [13]
    prompt = torch.tensor([prefix] * 4)
    response = torch.tensor([chosen] * 4)
    ids = torch.cat((prompt, response), 1)
    attention = torch.ones_like(ids)
    positions = torch.arange(ids.shape[1])[None, :].repeat(4, 1)
    flat = dict(responses=response, input_ids=ids, attention_mask=attention, position_ids=positions)
    actor = Actor()
    with capture_existing_flat_calls(actor) as old_calls:
        _, old = actor._forward_flat_micro_batch(flat, 1.)
    with capture_existing_flat_calls(actor) as repeat_calls:
        _, repeated = actor._forward_flat_micro_batch(flat, 1.)
    lengths, mask = torch.full((4,), len(chosen), dtype=torch.long), torch.ones_like(response)
    tensors = dict(flat, prompts=prompt, old_log_probs=old, response_sampled_lengths=lengths, response_sample_mask=mask)
    def trace_call(ids, positions, groups, selected, prompt_phase):
        return dict(input_tokens=torch.tensor(ids), input_positions=torch.tensor(positions),
            query_lens=[item['query_length'] for item in groups], seq_lens=[item['sequence_length'] for item in groups],
            groups=groups, selected_token_indices=torch.tensor(selected), observer_errors=[], decode_uses_cuda_graph=False,
            slot_mapping=torch.arange(len(ids)),
            actual_model_calls=[dict(args=(), input_ids=torch.tensor(ids), positions=torch.tensor(positions),
                                    slot_mapping=torch.arange(len(ids)))])
    groups = [dict(request_id=str(i), seq_id=i, is_prompt=True, sequence_length=len(prefix), prompt_length=len(prefix),
        num_computed_tokens=0, query_length=len(prefix), mrope_position_delta=None) for i in range(4)]
    calls = [trace_call(prefix*4, list(range(len(prefix)))*4, groups, [(i+1)*len(prefix)-1 for i in range(4)], True)]
    for output_index in range(len(chosen)-1):
        groups = [dict(request_id=str(i), seq_id=i, is_prompt=False, sequence_length=len(prefix)+output_index+1,
            prompt_length=len(prefix), num_computed_tokens=len(prefix)+output_index, query_length=1,
            mrope_position_delta=None) for i in range(4)]
        calls.append(trace_call([chosen[output_index]]*4, [len(prefix)+output_index]*4, groups, list(range(4)), False))
    observation = dict(schema='ACTUAL_VLLM_RUNNER_INPUT_OBSERVATION_V1', calls=calls, observer_errors=[],
        actual_forward_calls_added=0, rng_restores=0, request_outputs=[dict(request_id=str(i), prompt_token_ids=prefix,
            completions=[dict(index=0,token_ids=chosen,chosen_token_log_probs=old[i].tolist())]) for i in range(4)])
    return dict(serial=serial, rank=rank, pid=100+rank, phase='train', source_commit='c'*40, binding_sha256='d'*64,
        sampled_ids=response, sampled_lengths=lengths, sampled_mask=mask,
        tensors=tensors, sampled_log_probs=old, actor_repeated_log_probs=repeated,
        perturn_hf_flat_calls=old_calls, repeated_perturn_hf_flat_calls=repeat_calls,
        actual_pad_token_id=PAD, actual_eos_token_id=EOS, actual_engine_prompt_token_ids=[prefix]*4,
        actual_runner_observation=observation,
        actual_sampling_parameters=dict(temperature=1., top_p=1., top_k=-1, n=1),
        repeated_probe_state_before={'identity': {'pid': 100+rank}}, repeated_probe_state_after={'identity': {'pid': 100+rank}},
        probe_rng_unchanged=True, probe_complete_unchanged=True)


def fixture():
    roles = {role: {} for role in ('old', 'current', 'reference')}
    complete = {}
    for role in roles:
        for rank in range(8):
            raw = complete_raw(rank, role)
            complete[(role, rank)] = raw
            roles[role].update(unpack_complete_action_calls(raw, prepare=prepare_action_prefix_micro, pad_id=PAD, eos_id=EOS))
    turns = [[dict(source=i, step_id=step, generation_prefix=provenance()[step]['generation_prefix'],
                   retained_action_tokens=provenance()[step]['actor_action_tokens'],
                   raw_actor_record=provenance()[step]) for i in range(32)] for step in range(2)]
    raw_turns = {(serial, rank): sample_raw(rank, serial) for serial in range(2) for rank in range(8)}
    data = inputs(32)
    prepared = prepare_action_prefix_micro(data, PAD, EOS)
    actor = Actor()
    _, old = forward_prepared_action_prefix(prepared, 1., actor._forward_flat_micro_batch)
    data.update(old_log_probs=old, ref_log_prob=old.clone())
    return turns, raw_turns, roles, data, complete


class Tests(unittest.TestCase):
    def collect(self, fixture_value=None):
        turns, raw, roles, batch, _ = fixture_value or fixture()
        return collect_raw_actions(turns, raw, roles, batch, list(range(32)), pad=PAD, eos=EOS,
                                   question_ids=[i//5 for i in range(32)], code_commit='c'*40, binding_sha256='d'*64)

    def test_real_hooks_do_not_change_outputs_model_or_rng(self):
        actor = Actor()
        prepared = prepare_action_prefix_micro(inputs(4), PAD, EOS)
        original = actor._forward_flat_micro_batch
        before_rng, before_parameter = torch.get_rng_state().clone(), actor.actor_module.weight.detach().clone()
        expected = original(prepared.flat_micro, 1.)
        with capture_existing_flat_calls(actor) as records:
            actual = actor._forward_flat_micro_batch(prepared.flat_micro, 1.)
        self.assertEqual(actor._forward_flat_micro_batch, original)
        self.assertNotIn('_forward_flat_micro_batch',vars(actor))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual, expected)))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(before_parameter, actor.actor_module.weight))
        self.assertTrue(validate_actual_model_inputs(records[0])['actual_hf_model_positions_observed'])

    def test_complete_orchestration_preserves_actual_canonical_prefix(self):
        alignment, records, checks, actions = self.collect()
        self.assertEqual(alignment['recorded_actions'], 64)
        self.assertEqual(sum(row['valid_tokens'] for row in records), 96)
        self.assertEqual(actions[0]['actions'][1]['scoring_prefix'], [10, 11, 80, 99])
        self.assertEqual(checks[0]['effective_trajectory_token_weight'], 1/3)
        self.assertTrue(alignment['actual_vllm_runner_positions_observed'])  # Synthetic captured-input fixture only.

    def test_real_middle_sampled_pad_is_preserved(self):
        alignment, records, checks, actions = self.collect()
        self.assertEqual(actions[0]['actions'][0]['retained_action_tokens'], [12, PAD])
        self.assertEqual(actions[0]['actions'][0]['actual_hf_mask'], [1, 1])
        self.assertEqual(checks[0]['effective_tokens'], 2)

    def test_actual_hf_position_tamper_rejected(self):
        raw = complete_raw(0, 'old')
        raw['flat_calls'][0]['model_calls'][0]['kwargs']['position_ids'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'model positions'):
            unpack_complete_action_calls(raw, prepare=prepare_action_prefix_micro, pad_id=PAD, eos_id=EOS)

    def test_actual_hf_input_tamper_rejected(self):
        raw = complete_raw(0, 'reference')
        raw['flat_calls'][0]['model_calls'][0]['kwargs']['input_ids'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'model tokens'):
            unpack_complete_action_calls(raw, prepare=prepare_action_prefix_micro, pad_id=PAD, eos_id=EOS)

    def test_flat_scattered_probability_mapping_tamper_rejected(self):
        raw = complete_raw(0, 'old')
        raw['output_log_probs'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'scattered'):
            unpack_complete_action_calls(raw, prepare=prepare_action_prefix_micro, pad_id=PAD, eos_id=EOS)

    def test_source_offset_and_mask_tamper_rejected(self):
        for key in ('actor_action_output_offsets', 'loss_mask', 'actor_action_count'):
            raw = complete_raw(0, 'old')
            raw['input_tensors'][key].view(-1)[0] += 1
            with self.assertRaises(ValueError):
                unpack_complete_action_calls(raw, prepare=prepare_action_prefix_micro, pad_id=PAD, eos_id=EOS)

    def test_worker_and_engine_sampling_lengths_must_match(self):
        raw = sample_raw(0, 0)
        raw['sampled_lengths'] = torch.tensor([1, 2, 2, 2])
        with self.assertRaisesRegex(ValueError, 'contiguous sampling support'):
            validate_real_sample(raw, PAD, EOS)

    def test_real_pad_mask_cannot_be_silently_dropped(self):
        raw = sample_raw(0, 0)
        raw['tensors']['attention_mask'][0, -1] = 0
        with self.assertRaisesRegex(ValueError, 'PAD token lost'):
            validate_real_sample(raw, PAD, EOS)

    def test_missing_ref_forward_rejected(self):
        value = fixture()
        del value[2]['reference'][(0,0)]
        with self.assertRaisesRegex(ValueError, 'forwarded complete action'):
            self.collect(value)

    def test_complete_current_probability_error_is_reported_not_absorbed(self):
        value = fixture()
        value[2]['current'][(0,0)]['log_probs'] = value[2]['current'][(0,0)]['log_probs'] + 2.
        checks = self.collect(value)[2]
        self.assertGreater(checks[0]['complete_current_vs_old_joint_sum_delta'], 3.9)
        self.assertFalse(checks[0]['probability_accepted'])

    def test_sampler_residual_remains_visible(self):
        value = fixture()
        value[1][(0,0)]['sampled_log_probs'] = value[1][(0,0)]['sampled_log_probs'] - 2.836
        for row, request in enumerate(value[1][(0,0)]['actual_runner_observation']['request_outputs']):
            request['completions'][0]['chosen_token_log_probs'] = value[1][(0,0)]['sampled_log_probs'][row].tolist()
        alignment, records, checks, _ = self.collect(value)
        self.assertGreater(records[0]['actor_vllm_max_abs_delta'], 2.8)
        self.assertGreater(abs(checks[0]['full_vs_vllm_joint_sum_delta']), 5.6)
        self.assertFalse(alignment['probability_accepted'])

    def test_nonfinite_effective_sampler_rejected(self):
        value = fixture()
        value[1][(0,0)]['sampled_log_probs'][0,0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite actual sampled'):
            self.collect(value)

    def test_incorrect_raw_concat_as_prefix_rejected(self):
        value = fixture()
        value[2]['old'][(0,1)]['prefix'] = [10, 11, 12, PAD, 99]
        with self.assertRaisesRegex(ValueError, 'input differs from sampler'):
            self.collect(value)

    def test_missing_actual_model_call_is_not_position_evidence(self):
        raw = complete_raw(0, 'old')
        raw['flat_calls'][0]['model_calls'] = []
        with self.assertRaisesRegex(ValueError, 'actual HF module calls'):
            unpack_complete_action_calls(raw, prepare=prepare_action_prefix_micro, pad_id=PAD, eos_id=EOS)

    def test_observer_restores_original_hook_after_original_failure(self):
        actor = Actor()
        def failing(micro, temperature):
            raise RuntimeError('original error')
        actor._forward_flat_micro_batch = failing
        with self.assertRaisesRegex(RuntimeError, 'original error'):
            with capture_existing_flat_calls(actor):
                actor._forward_flat_micro_batch({}, 1.)
        self.assertIs(actor._forward_flat_micro_batch, failing)
        self.assertEqual(len(actor.actor_module._forward_pre_hooks), 0)

    def driver_fixture(self, root, value):
        from search_r1.baselines.recovery import file_sha256
        turns, raw_turns, roles, data, complete = value
        def save(path, raw):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(raw, path)
            return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))
        (root / 'actual-vllm-backend-identity.json').write_text('{}')
        (root / 'actual-rollout-2560.pt').write_text('CPU fixture, not real rollout')
        (root / 'training-prefix-turns.json').write_text(json.dumps(turns))
        for (serial, rank), raw in raw_turns.items():
            receipt = save(root / f'probabilities/train/call-{serial}-rank-{rank}.pt', raw)
            (root / f'probabilities/train/call-{serial}-rank-{rank}.json').write_text(json.dumps(dict(
                rank=rank, pid=100+rank, serial=serial, source_commit='c'*40, raw=receipt,
                repeated_probability_probe_rng_unchanged=True,
                repeated_probability_probe_complete_state_unchanged=True)))
        for (role, rank), raw in complete.items():
            receipt = save(root / f'probabilities/complete-actions/{role}-rank-{rank}.pt', raw)
            (root / f'probabilities/complete-actions/{role}-rank-{rank}.json').write_text(json.dumps(dict(
                role=role, rank=rank, pid=100+rank, code_commit='c'*40, binding_sha256='d'*64, raw=receipt)))
            if role != 'old':
                (root / f'probabilities/complete-actions/{role}-neutral-rank-{rank}.json').write_text(json.dumps(dict(
                    rank=rank, actual_adam_writes=0, complete_state_unchanged=True, fixed_reference_unchanged=True)))
        from verl import DataProto
        return DataProto.from_dict(tensors=data, non_tensors={'audit_trajectory_row':np.arange(32,dtype=object),
            'audit_question_index':np.array([i//5 for i in range(32)],dtype=object)})

    def test_actual_driver_pre_pause_orchestration(self):
        import calibration_driver_v1 as driver
        value = fixture()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, SEARCH_R1_BASELINE_COMMIT='c'*40,
                GIGPO_POLICY_NATIVE_BINDING_SHA256='d'*64):
            root = Path(directory)
            batch = self.driver_fixture(root, value)
            with patch.object(driver, 'wait_probability_acceptance') as pause:
                result = driver.probability_and_alignment(None, root, value[0], dict(code_commit='c'*40,
                    pad_token_id=PAD, eos_token_id=EOS, calibration_phase='probability_probe_then_wait'), batch)
            self.assertEqual(pause.call_count, 1)
            self.assertEqual(result['recorded_actions'], 64)
            manifest = json.loads((root / 'probability-material-manifest.json').read_text())
            self.assertEqual(manifest['schema'], 'RAW_ACTION_PROBABILITY_MATERIAL_V1')
            self.assertEqual(manifest['actual_hf_roles'], ['old','current','reference'])
            self.assertGreater(len(manifest['files']), 90)
            self.assertFalse(torch.cuda.is_initialized())

    def test_actual_driver_reference_neutrality_rejects_before_pause(self):
        import calibration_driver_v1 as driver
        value = fixture()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, SEARCH_R1_BASELINE_COMMIT='c'*40,
                GIGPO_POLICY_NATIVE_BINDING_SHA256='d'*64):
            root = Path(directory)
            batch = self.driver_fixture(root, value)
            path = root / 'probabilities/complete-actions/reference-neutral-rank-5.json'
            row = json.loads(path.read_text()); row['fixed_reference_unchanged'] = False
            path.write_text(json.dumps(row))
            with patch.object(driver, 'wait_probability_acceptance') as pause:
                with self.assertRaisesRegex(ValueError, 'changed real actor/reference'):
                    driver.probability_and_alignment(None, root, value[0], dict(code_commit='c'*40,
                        pad_token_id=PAD, eos_token_id=EOS, calibration_phase='probability_probe_then_wait'), batch)
            self.assertEqual(pause.call_count, 0)

    def test_new_raw_contract_rejects_not_ready_and_unreviewed_modes(self):
        from calibration_contract_v1 import validate_binding, RAW_SOURCE
        with self.assertRaisesRegex(ValueError, 'No isolated independent review'):
            validate_binding(dict(status='DRAFT_NOT_READY'), RAW_SOURCE)

    def test_actual_decode_runner_position_changed_rejected(self):
        raw = sample_raw(0,0)
        raw['actual_runner_observation']['calls'][1]['input_positions'][0] += 1
        with self.assertRaisesRegex(ValueError, 'runner/model tokens, positions'):
            validate_real_sample(raw,PAD,EOS)

    def test_unobserved_cuda_graph_or_missing_forward_is_not_position_proof(self):
        for mutate in ('graph','missing'):
            raw = sample_raw(0,0)
            call = raw['actual_runner_observation']['calls'][0]
            if mutate == 'graph':
                call['decode_uses_cuda_graph'] = True
            else:
                call['actual_model_calls'] = []
            with self.assertRaises(ValueError):
                validate_real_sample(raw,PAD,EOS)

    def test_sampler_selected_prediction_crossing_request_rejected(self):
        raw = sample_raw(0,0)
        raw['actual_runner_observation']['calls'][0]['selected_token_indices'][0] = 0
        with self.assertRaisesRegex(ValueError, 'prediction crosses'):
            validate_real_sample(raw,PAD,EOS)

    def test_incomplete_decode_prediction_coverage_rejected(self):
        raw = sample_raw(0,0)
        raw['actual_runner_observation']['calls'] = raw['actual_runner_observation']['calls'][:1]
        with self.assertRaisesRegex(ValueError, 'prediction coverage'):
            validate_real_sample(raw,PAD,EOS)

    def test_request_order_or_real_completion_token_mismatch_rejected(self):
        raw = sample_raw(0,0)
        raw['actual_runner_observation']['request_outputs'][0]['completions'][0]['token_ids'][0] = 13
        with self.assertRaisesRegex(ValueError, 'CompletionOutput tokens'):
            validate_real_sample(raw,PAD,EOS)

    def test_runner_hooks_preserve_calls_inputs_rng_and_restore(self):
        class VModel(torch.nn.Module):
            def forward(self, input_ids, positions, attn_metadata):
                return input_ids
        class Seq:
            mrope_position_delta = None
            def get_len(self): return 2
            def get_prompt_len(self): return 2
            def get_num_computed_tokens(self): return 0
        class Runner:
            def __init__(self): self.model = VModel(); self.forward_count = 0
            def prepare_model_input(self, seq_group_metadata_list, virtual_engine=0, finished_requests_ids=None):
                return SimpleNamespace(input_tokens=torch.tensor([10,11]), input_positions=torch.tensor([0,1]),
                    query_lens=[2],seq_lens=[2],sampling_metadata=SimpleNamespace(selected_token_indices=torch.tensor([1])),
                    attn_metadata=SimpleNamespace(num_prefills=1,num_prefill_tokens=2,num_decode_tokens=0,
                        slot_mapping=torch.tensor([0,1]),decode_metadata=None))
            def execute_model(self, model_input, kv_caches, intermediate_tensors=None, num_steps=1):
                self.forward_count += 1
                return self.model(input_ids=model_input.input_tokens,positions=model_input.input_positions,
                                  attn_metadata=model_input.attn_metadata)
        class Engine:
            def __init__(self, runner):
                self.llm_engine = SimpleNamespace(model_executor=SimpleNamespace(worker=SimpleNamespace(model_runner=runner)))
            def _post_process_outputs(self, request_outputs): return request_outputs
        runner, engine = Runner(), None
        engine = Engine(runner)
        methods = (runner.prepare_model_input,runner.execute_model,engine._post_process_outputs)
        group = SimpleNamespace(request_id='0',is_prompt=True,seq_data={0:Seq()})
        request = SimpleNamespace(request_id='0',prompt_token_ids=[10,11],outputs=[SimpleNamespace(index=0,token_ids=[12],
            logprobs=[{12:SimpleNamespace(logprob=-.1)}])])
        before_rng = torch.get_rng_state().clone()
        with observe_actual_runner(engine) as observation:
            model_input = runner.prepare_model_input([group])
            result = runner.execute_model(model_input,[])
            returned = engine._post_process_outputs([request])
        self.assertEqual((runner.prepare_model_input,runner.execute_model,engine._post_process_outputs),methods)
        self.assertNotIn('prepare_model_input',vars(runner))
        self.assertNotIn('execute_model',vars(runner))
        self.assertNotIn('_post_process_outputs',vars(engine))
        self.assertTrue(torch.equal(result,model_input.input_tokens))
        self.assertIs(returned[0],request)
        self.assertEqual(runner.forward_count,1)
        self.assertTrue(torch.equal(before_rng,torch.get_rng_state()))
        self.assertEqual(len(runner.model._forward_pre_hooks),0)
        rows = validate_actual_runner_observation(observation,[[10,11]],torch.tensor([[12]]),torch.tensor([1]),torch.tensor([[-.1]]))
        self.assertEqual(rows[0]['sampled_prediction_positions'],[1])


if __name__ == '__main__':
    unittest.main()
