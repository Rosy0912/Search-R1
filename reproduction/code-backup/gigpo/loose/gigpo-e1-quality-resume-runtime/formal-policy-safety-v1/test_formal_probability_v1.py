"""CPU counterexamples for metadata controls. Never claims GPU/FSDP acceptance."""
import ast
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import types
import unittest

from formal_probability_v1 import (collective_gate, collective_rejection_gate, digest, require,
                                 validate_outer, validate_live_arms, processed_tokens_without_right_padding)
from formal_probability_proofs_v1 import file_record, verify_record, read_json_record, manifest_records, validate_manifest_closure

SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'
TOLERANCES = dict(actor_recompute_abs_tolerance=.01, actor_vllm_abs_tolerance=.01,
                  actor_recompute_joint_abs_tolerance=.01, actor_vllm_joint_abs_tolerance=.01)


def action():
    return dict(step_id=0, generation_prefix=[10, 11], scoring_prefix=[10, 11],
        final_scoring_prefix=[10, 11], engine_prompt_token_ids=[10, 11], worker_prompt_token_ids=[10, 11],
        complete_processed_action_tokens=[20, 21], retained_action_tokens=[20, 21],
        effective_scored_tokens=[20, 21], raw_sampled_tokens=[20, 21, 22, 0], raw_sampled_response_mask=[1, 1, 1, 0],
        worker_response_tokens=[20, 21, 22, 0], worker_prompt_position_ids=[0, 1],
        sampler_runner_positions_observed=False, engine_position_evidence_scope='DERIVED_BACKEND_CONTRACT_EXPECTATION',
        worker_action_position_ids=[2, 3], scoring_action_position_ids=[2, 3], worker_action_mask=[1, 1],
        scoring_action_mask=[1, 1], sampling_parameters=dict(temperature=1, top_p=1, top_k=-1, n=1,
            min_p=0,best_of=1,logits_processors_count=0,repetition_penalty=1, presence_penalty=0, frequency_penalty=0),
        full_scoring_temperature=1, per_turn_scoring_temperature=1,
        full_old_log_probs=[-2., -3.], per_turn_old_log_probs=[-2., -3.], raw_sampled_log_probs=[-2., -3.])


def trajectories():
    return [dict(source=index, trajectory_uid=f'actual-trajectory-{index}', actions=[action()]) for index in range(2560)]


def ranks():
    return [dict(rank=rank, pid=1000 + rank, start_ticks=str(100 + rank), outer_step=26,
        source_commit=SOURCE, job_id='9999', run_id='cpu-counterexample-no-job', local_ok=True, error=None,
        global_manifest_sha256='a' * 64, global_batch_digest='b' * 64, production_artifact_sha256='c' * 64,
        probability_acceptance_sha256='d' * 64, global_arm_sha256='f' * 64,
        local_batch_digest=digest(dict(rank=rank))) for rank in range(8)]


def extracted_methods(file, cls, names, namespace):
    """Exercise actual candidate control bytes without importing CUDA or Ray."""
    tree = ast.parse((Path(__file__).parent / file).read_text())
    owner = next(row for row in tree.body if isinstance(row, ast.ClassDef) and row.name == cls)
    functions = [copy.deepcopy(row) for row in owner.body if isinstance(row, ast.FunctionDef) and row.name in names]
    for row in functions: row.decorator_list = []
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, file, 'exec'), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


class StructuralCounterexamples(unittest.TestCase):
    def test_processed_interior_pad_not_silently_deleted(self):
        with self.assertRaises(ValueError): processed_tokens_without_right_padding([20, 0, 21, 0], 0)

    def test_processed_right_padding_only_is_removed(self):
        raw = [20, 21, 0, 0]
        self.assertEqual(processed_tokens_without_right_padding(raw, 0), [20, 21])
        self.assertEqual(raw, [20, 21, 0, 0])

    def test_full2560_legal_closed_tag_tail_preserves_inputs(self):
        rows = trajectories(); before = copy.deepcopy(rows)
        summary = validate_outer(rows, TOLERANCES, 0)
        self.assertEqual(summary['trajectories'], 2560)
        self.assertEqual(summary['effective_tokens'], 5120)
        self.assertEqual(rows, before)

    def reject(self, change):
        rows = trajectories(); change(rows)
        with self.assertRaises((ValueError, TypeError, KeyError)):
            validate_outer(rows, TOLERANCES, 0)

    def test_later_row_prefix_does_not_hide_behind_first_row(self):
        self.reject(lambda rows: rows[2559]['actions'][0]['generation_prefix'].append(12))

    def test_full_context_vs_truncated_scoring(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(scoring_prefix=[11]))

    def test_full_processed_action_vs_cropped_scored_action(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(complete_processed_action_tokens=[20, 21, 22]))

    def test_retokenized_decode_equivalent_token_ids_still_reject(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(raw_sampled_tokens=[99, 22, 22, 0]))

    def test_interior_sample_pad_cannot_be_deleted(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(raw_sampled_tokens=[20, 0, 21, 0]))

    def test_mask_hole_is_not_a_valid_sampling_length(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(raw_sampled_response_mask=[1, 0, 1, 0]))

    def test_no_EOS_rightpad_is_not_fabricated_into_action_probability_mass(self):
        rows = trajectories(); rows[0]['actions'][0].update(raw_sampled_tokens=[20, 21, 0, 0],
            raw_sampled_response_mask=[1, 1, 1, 1])
        self.assertEqual(validate_outer(rows, TOLERANCES, 0)['effective_tokens'], 5120)

    def test_worker_position_shift(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(worker_action_position_ids=[1, 2]))

    def test_scoring_mask_cannot_absorb_observation(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(scoring_action_mask=[1, 0]))

    def test_final_batch_prefix_must_equal_observed_context(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(final_scoring_prefix=[10, 12]))

    def test_temperature_change_cannot_use_ppo_kl_threshold(self):
        self.reject(lambda rows: rows[0]['actions'][0]['sampling_parameters'].update(temperature=.7))

    def test_minp_distribution_change_refuses_even_when_selected_logprobs_match(self):
        self.reject(lambda rows: rows[0]['actions'][0]['sampling_parameters'].update(min_p=.1))

    def test_duplicate_after_balance_trajectory(self):
        self.reject(lambda rows: rows[2559].update(source=2558))

    def test_duplicate_trajectory_uid(self):
        self.reject(lambda rows: rows[2559].update(trajectory_uid=rows[0]['trajectory_uid']))

    def test_missing_action_timeline(self):
        self.reject(lambda rows: rows[2000].update(actions=[]))

    def test_missing_middle_step(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(step_id=1))

    def test_nan_effective_probability(self):
        self.reject(lambda rows: rows[0]['actions'][0].update(full_old_log_probs=[float('nan'), -3.]))

    def test_joint_sum_not_only_max_token(self):
        # Each token is within.01; complete textual-action error is.018>.01.
        self.reject(lambda rows: rows[0]['actions'][0].update(full_old_log_probs=[-1.991, -2.991]))

    def test_null_calibrated_tolerance_refuses(self):
        with self.assertRaises(ValueError): validate_outer(trajectories(), None, 0)

    def test_empty_local_action_allowed_when_global_has_effective_mass(self):
        rows = trajectories(); empty = rows[0]['actions'][0]
        for key in ('complete_processed_action_tokens', 'retained_action_tokens', 'effective_scored_tokens',
                    'worker_action_position_ids', 'scoring_action_position_ids', 'worker_action_mask',
                    'scoring_action_mask', 'full_old_log_probs', 'per_turn_old_log_probs', 'raw_sampled_log_probs'):
            empty[key] = []
        self.assertEqual(validate_outer(rows, TOLERANCES, 0)['effective_tokens'], 5118)


class CollectiveCounterexamples(unittest.TestCase):
    def test_eight_matching_actual_ranks(self):
        self.assertEqual(collective_gate(ranks())['actual_adam_writes_at_gate'], 0)

    def test_single_late_rank_error_rejects_all(self):
        rows = ranks(); rows[7].update(local_ok=False, error='different real mask')
        with self.assertRaises(ValueError): collective_gate(rows)

    def test_single_rank_wrong_outer(self):
        rows = ranks(); rows[7]['outer_step'] = 27
        with self.assertRaises(ValueError): collective_gate(rows)

    def test_old_manifest_cannot_be_replayed_on_one_rank(self):
        rows = ranks(); rows[7]['global_manifest_sha256'] = 'e' * 64
        with self.assertRaises(ValueError): collective_gate(rows)

    def test_all_null_shas_do_not_look_like_matching_acceptance(self):
        rows = ranks()
        for row in rows: row['probability_acceptance_sha256'] = None
        with self.assertRaises(ValueError): collective_gate(rows)


class ActualWorkerControlCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.old_env = dict(os.environ); self.addCleanup(self.restore_env)
        os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=self.temp.name,
            SEARCH_R1_BASELINE_COMMIT=SOURCE, SLURM_JOB_ID='9999',
            SEARCH_R1_BASELINE_RUN_ID='cpu-counterexample-no-job')
        self.collective = ranks()
        self.adam_calls = []
        def gather(output, local):
            output[:] = copy.deepcopy(self.collective); output[0] = copy.deepcopy(local)
        def save(value, path): Path(path).write_text(json.dumps(value, allow_nan=False))
        class Output(str):
            meta_info = dict(metrics={'actor/optimizer_steps': 10})
        original = types.SimpleNamespace(update_actor=lambda worker, data: self.adam_calls.append('original') or Output('same-output'))
        ns = dict(torch=types.SimpleNamespace(distributed=types.SimpleNamespace(all_gather_object=gather),
            load=lambda path, **kwargs: json.loads(Path(path).read_text())),
            os=os, Path=Path, state_digest=digest, scored_batch_state=lambda data: data.value,
            collective_gate=collective_gate, require=require, atomic_json_save=save, ActorRolloutRefWorker=original,
            file_sha256=lambda path: file_record(path)['sha256'], validate_live_arms=validate_live_arms,
            read_json_record=read_json_record, verify_record=verify_record, json=json)
        self.method = extracted_methods('formal_probability_worker_v1.py', 'ProductionProbabilityWorker',
                                        ['update_actor'], ns).update_actor
        self.pause = dict(identity=dict(pid=os.getpid(), start_ticks='100'),
                          model='model-before', Adam='Adam-before', RNG='RNG-before', buffers='buffer-before')
        manifest = dict(global_batch_digest='b' * 64, production_artifact_sha256='c' * 64,
            probability_acceptance_sha256='d' * 64, driver_post_sampling_state_digest='e' * 64,
            chunks=[dict(rank=r, batch_digest=digest(dict(rank=r))) for r in range(8)])
        pause_path = Path(self.temp.name) / 'pause.json'; pause_path.write_text(json.dumps(self.pause))
        arm_path = Path(self.temp.name) / 'arm.json'
        arm_path.write_text(json.dumps(dict(complete_pause_state_sha256=digest(self.pause), pause_state=file_record(pause_path))))
        own = dict(rank=0, pid=os.getpid(), start_ticks='100', outer_step=26, manifest_sha256='a'*64,
            complete_pause_state_sha256=digest(self.pause), arm_receipt=file_record(arm_path), pause_state=file_record(pause_path))
        arms = [dict(own, rank=r, pid=own['pid']+r) for r in range(8)]
        all_path = Path(self.temp.name) / 'armed-workers.json'
        all_path.write_text(json.dumps(dict(outer_step=26, manifest_sha256='a'*64,
            driver_post_sampling_state_digest='e'*64, workers=arms)))
        arms_sha = file_record(all_path)['sha256']
        for row in self.collective: row['global_arm_sha256'] = arms_sha
        self.worker = types.SimpleNamespace(rank=0, world_size=8, _production_capture=False,
            _production_outer_step=26, _production_arm=dict(manifest=manifest, manifest_sha256='a' * 64,
                complete_pause_state=copy.deepcopy(self.pause), arm_receipt=file_record(arm_path)),
                _complete_pause_state=lambda: copy.deepcopy(self.pause))
        class Data:
            value = dict(rank=0)
            meta_info = dict(formal_probability_gate=dict(manifest_sha256='a' * 64,
                armed_workers_path=str(all_path), armed_workers_sha256=arms_sha), baseline_outer_step=26)
            def __len__(self): return 320
        self.data = Data()

    def restore_env(self):
        os.environ.clear(); os.environ.update(self.old_env)

    def deny(self, mutate):
        mutate()
        with self.assertRaises(ValueError): self.method(self.worker, self.data)
        self.assertEqual(self.adam_calls, [])

    def test_original_update_called_exactly_once_after_matching_gate(self):
        output = self.method(self.worker, self.data)
        self.assertEqual(output, 'same-output'); self.assertEqual(self.adam_calls, ['original'])

    def test_unarmed_call_zero_original_optimizer(self):
        self.deny(lambda: setattr(self.worker, '_production_arm', None))

    def test_model_changed_after_admission_zero_original_optimizer(self):
        self.deny(lambda: self.pause.update(model='different'))

    def test_Adam_changed_after_admission_zero_original_optimizer(self):
        self.deny(lambda: self.pause.update(Adam='different'))

    def test_RNG_changed_after_admission_zero_original_optimizer(self):
        self.deny(lambda: self.pause.update(RNG='different'))

    def test_buffers_changed_after_admission_zero_original_optimizer(self):
        self.deny(lambda: self.pause.update(buffers='different'))

    def test_PID_start_changed_after_admission_zero_original_optimizer(self):
        self.deny(lambda: self.pause['identity'].update(start_ticks='999'))

    def test_actual_dispatched_provenance_changed_zero_original_optimizer(self):
        self.deny(lambda: setattr(self.data, 'value', dict(rank=0, trajectory_uid='different')))

    def test_one_remote_bad_flag_zero_original_optimizer(self):
        self.deny(lambda: self.collective[7].update(local_ok=False, error='actual local mismatch'))

    def test_previous_outer_arm_reuse_zero_original_optimizer(self):
        self.deny(lambda: setattr(self.worker, '_production_last_admitted', 26))

    def test_local_raw_journal_failure_matches_before_optimizer(self):
        def fail(*args): raise OSError('CPU injected journal read-only')
        self.method.__globals__['atomic_json_save'] = fail
        self.deny(lambda: None)

    def test_remote_journal_failure_in_second_collective_zero_optimizer(self):
        calls = []
        def gather(output, local):
            calls.append(1); output[:] = copy.deepcopy(self.collective); output[0] = copy.deepcopy(local)
            if len(calls) == 2: output[7].update(local_ok=False, error='remote actual journal failed')
        self.method.__globals__['torch'].distributed.all_gather_object = gather
        self.deny(lambda: None)
        self.assertEqual(len(calls), 2)


class ActualRejectionControlCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.old_env = dict(os.environ); self.addCleanup(self.restore_env)
        os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=self.temp.name,
            SEARCH_R1_BASELINE_COMMIT=SOURCE, SLURM_JOB_ID='9999',
            SEARCH_R1_BASELINE_RUN_ID='cpu-counterexample-no-job')
        self.path = Path(self.temp.name) / 'before-adam-rejection.json'
        self.path.write_text('{"status":"CPU_INJECTED_REAL_REFUSAL"}')
        self.sha = file_record(self.path)['sha256']
        self.remote = [dict(rank=rank, pid=1000+rank, start_ticks=str(100+rank), outer_step=26,
            source_commit=SOURCE, job_id='9999', run_id='cpu-counterexample-no-job',
            failure_path=str(self.path), failure_sha256=self.sha, local_ok=True, error=None) for rank in range(8)]
        self.gathers = []
        self.inject_second = lambda rows: None
        def gather(output, local):
            self.gathers.append(copy.deepcopy(local)); output[:] = copy.deepcopy(self.remote)
            output[0] = copy.deepcopy(local)
            if len(self.gathers) == 2: self.inject_second(output)
        def save(value, path): Path(path).write_text(json.dumps(value, allow_nan=False))
        # macOS has no procfs. This controls-only fixture supplies a declared
        # Linux start-tick string; actual process identity remains a GPU gate.
        def control_path(path):
            if str(path) == '/proc/self/stat':
                return types.SimpleNamespace(read_text=lambda: '1 (CPU fixture) ' + ' '.join(map(str, range(30))))
            return Path(path)
        ns = dict(torch=types.SimpleNamespace(distributed=types.SimpleNamespace(all_gather_object=gather)),
            os=os, Path=control_path, require=require, file_sha256=lambda path: file_record(path)['sha256'],
            atomic_json_save=save, collective_rejection_gate=collective_rejection_gate)
        self.method = extracted_methods('formal_probability_worker_v1.py', 'ProductionProbabilityWorker',
                                        ['production_reject_outer'], ns).production_reject_outer
        self.worker = types.SimpleNamespace(rank=0, world_size=8, _production_outer_step=26,
                                            _production_capture=False, _production_arm='not-consumed')

    def restore_env(self):
        os.environ.clear(); os.environ.update(self.old_env)

    def deny(self, change):
        change()
        with self.assertRaises(ValueError): self.method(self.worker, 26, str(self.path), self.sha)
        self.assertEqual(len(self.gathers), 2)
        self.assertEqual(self.worker._production_arm, 'not-consumed')
        self.assertFalse(hasattr(self.worker, '_production_rejected'))

    def test_actual_eight_matching_refusals_use_two_collectives(self):
        receipt = self.method(self.worker, 26, str(self.path), self.sha)
        self.assertEqual(receipt['status'], 'ALL_RANKS_REJECTED_BEFORE_ORIGINAL_UPDATE_RPC')
        self.assertEqual(receipt['actual_adam_writes_at_gate'], 0)
        self.assertEqual(len(self.gathers), 2)
        self.assertIsNone(self.worker._production_arm)

    def test_bad_local_boundary_still_reaches_both_collectives(self):
        self.deny(lambda: setattr(self.worker, '_production_outer_step', 27))

    def test_changed_failure_original_still_matches_refusal(self):
        self.deny(lambda: self.path.write_text('{"changed":true}'))

    def test_local_rejection_journal_failure_matches_without_update(self):
        def fail(*args): raise OSError('CPU injected journal IO failure')
        self.method.__globals__['atomic_json_save'] = fail
        self.deny(lambda: None)

    def test_remote_rejection_journal_failure_matches_without_update(self):
        self.inject_second = lambda rows: rows[7].update(local_ok=False, error='remote journal failed')
        self.deny(lambda: None)

    def test_remote_different_failure_original_refused(self):
        self.deny(lambda: self.remote[7].update(failure_sha256='a'*64))

    def test_remote_source_relabel_refused(self):
        self.deny(lambda: self.remote[7].update(source_commit='a'*40))


class ActualDriverControlCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.old_env = dict(os.environ); self.addCleanup(self.restore_env)
        os.environ['SEARCH_R1_BASELINE_COMMIT'] = SOURCE
        self.state = dict(pid=1234, start_ticks='123', rng='after-sampling-RNG', cursor=13312,
                          gigpo='same-actual-GiGPO', fixed_batch='same-scored-batch')
        self.adam_calls, self.rejections = [], []
        self.change_during_arm = lambda: None
        def collect(trainer, batch, turns, directory, binding):
            path = Path(directory) / 'manifest.json'; path.write_text('{}')
            return path, dict(driver_post_sampling_state_digest=digest(self.state))
        def arm(*args):
            self.change_during_arm()
            rows = []
            for rank in range(8):
                pause = Path(self.temp.name)/f'pause-{rank}.json'; pause.write_text('{}')
                row = dict(rank=rank, pid=1000+rank, start_ticks=str(100+rank), outer_step=26,
                    manifest_sha256='a'*64, complete_pause_state_sha256='b'*64, pause_state=file_record(pause))
                path = Path(self.temp.name)/f'arm-{rank}.json'; path.write_text(json.dumps(row))
                rows.append(dict(row, arm_receipt=file_record(path)))
            return rows
        def reject(*args):
            self.rejections.append(args)
            return [dict(status='ALL_RANKS_REJECTED_BEFORE_ORIGINAL_UPDATE_RPC') for _ in range(8)]
        def save(value, path): Path(path).write_text(json.dumps(value, allow_nan=False))
        namespace = dict(collect_outer=collect, actual_driver_boundary=lambda *args: copy.deepcopy(self.state),
                         state_digest=digest, file_sha256=lambda path: 'a' * 64,
                         require=require, atomic_json_save=save, os=os,
                         validate_live_arms=validate_live_arms, read_json_record=read_json_record,
                         verify_record=verify_record)
        self.update = extracted_methods('formal_probability_driver_v1.py', 'OuterProbabilityHooks', ['update'], namespace).update
        group = types.SimpleNamespace(production_arm_outer=arm, production_reject_outer=reject)
        trainer = types.SimpleNamespace(global_steps=26, actor_rollout_wg=group)
        self.hook = types.SimpleNamespace(trainer=trainer, outer=26, last_completed=25, turns=[['actual']],
            directory=Path(self.temp.name), binding={}, original_update=lambda data: self.adam_calls.append('original') or data)
        self.batch = types.SimpleNamespace(meta_info={})

    def restore_env(self):
        os.environ.clear(); os.environ.update(self.old_env)

    def test_unchanged_post_sampling_driver_calls_existing_rpc_once(self):
        output = self.update(self.hook, self.batch)
        self.assertIs(output, self.batch); self.assertEqual(self.adam_calls, ['original'])
        self.assertEqual(self.hook.last_completed, 26)

    def deny_change(self, change):
        self.change_during_arm = change
        with self.assertRaises(ValueError): self.update(self.hook, self.batch)
        self.assertEqual(self.adam_calls, []); self.assertEqual(len(self.rejections), 1)
        self.assertEqual(self.hook.last_completed, 25)

    def test_data_cursor_change_after_raw_sampling_zero_original_rpc(self):
        self.deny_change(lambda: self.state.update(cursor=13824))

    def test_driver_RNG_reset_to_presampling_state_zero_original_rpc(self):
        self.deny_change(lambda: self.state.update(rng='wrong-presampling-RNG'))

    def test_driver_PID_start_changed_zero_original_rpc(self):
        self.deny_change(lambda: self.state.update(start_ticks='999'))

    def test_GiGPO_state_changed_zero_original_rpc(self):
        self.deny_change(lambda: self.state.update(gigpo='different-state'))


class ObservationalWorkerNeutralityCPU(unittest.TestCase):
    def test_actual_observer_control_returns_same_object_and_uses_one_existing_forward(self):
        # This is a CPU callable/control spy, not a GPU probability comparison
        # or FSDP A/B replay. It detects extra invocation/RNG restoration here.
        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def clone(self): return Tensor(copy.deepcopy(self.value))
        class Output:
            batch = {key: Tensor([[1, 2], [3, 4]]) for key in
                     ('prompts','responses','input_ids','attention_mask','position_ids','old_log_probs')}
            meta_info = dict(temperature=1)
            def __len__(self): return 2
        output = Output()
        params = types.SimpleNamespace(temperature=1, top_p=1, top_k=-1, n=1, logprobs=1,
            max_tokens=500,min_p=0,best_of=1,logits_processors=None,repetition_penalty=1,presence_penalty=0,frequency_penalty=0)
        calls, raw_saved = [], []
        def engine_generate(*args, **kwargs):
            calls.append(kwargs); random.random()
            return Tensor([[20, 21], [20, 22]]), Tensor([[-2., -3.], [-2., -4.]])
        engine = types.SimpleNamespace(generate=engine_generate)
        def original_generate(worker, prompts):
            worker.rollout.inference_engine.generate(prompt_token_ids=[[10, 11], [10, 12]], sampling_params=params)
            return output
        random.seed(123456)
        worker = types.SimpleNamespace(rollout=types.SimpleNamespace(inference_engine=engine))
        baseline_output = original_generate(worker, {'same-input': [1, 2]})
        expected_rng = random.getstate(); calls.clear(); random.seed(123456)
        with tempfile.TemporaryDirectory() as directory:
            before_environment = dict(os.environ)
            try:
                os.environ.update(SEARCH_R1_BASELINE_OUTPUT=directory, SEARCH_R1_BASELINE_COMMIT=SOURCE)
                worker._production_capture = True; worker._production_outer_step = 26
                worker._production_call = 0; worker.rank = 0
                worker.tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=22)
                def raw_save(value, path):
                    raw_saved.append(value)
                    return dict(path=str(path), bytes=1, sha256='a' * 64)
                method = extracted_methods('formal_probability_worker_v1.py', 'ProductionProbabilityWorker',
                    ['generate_sequences'], dict(copy=copy, require=require,
                        ActorRolloutRefWorker=types.SimpleNamespace(generate_sequences=original_generate),
                        Path=Path, os=os, atomic_torch_save=raw_save, atomic_json_save=lambda *args: None)).generate_sequences
                input_value = {'same-input': [1, 2]}; input_before = copy.deepcopy(input_value)
                observed_output = method(worker, input_value)
                self.assertIs(observed_output, baseline_output)
                self.assertEqual(input_value, input_before)
                self.assertEqual(random.getstate(), expected_rng)
                self.assertEqual(len(calls), 1)
                self.assertIs(engine.generate, engine_generate)
                self.assertEqual(len(raw_saved), 1)
                self.assertEqual(raw_saved[0]['actual_engine_prompt_token_ids'], [[10, 11], [10, 12]])
            finally:
                os.environ.clear(); os.environ.update(before_environment)


if __name__ == '__main__':
    unittest.main(verbosity=2)
