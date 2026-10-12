"""UNEXECUTED CPU fixtures: real typed Ray cause, local IO, unknown RPC failure.

These tests initialize no Ray cluster and make no model/GPU/scheduler call.
They are not actual distributed preservation or probability acceptance evidence.
"""
import json
from types import SimpleNamespace
import pytest
import torch
from ray.exceptions import RayTaskError
from search_r1.baselines.policy_safety import PolicySafetyRejected, matched_policy_rejection
from calibration_boundary_v2 import is_matched_refusal
from calibration_failure_v2 import preserve_first_refusal
from generation_rpc_boundary_v3 import track_generation_rpc


def make_trainer(error=None, output=None):
    from test_worker_rpc_boundary_v4 import make_actual_rpc_fixture
    trainer, calls, output, schema, events = make_actual_rpc_fixture(output=output, failure=error)
    trainer._fixture_rpc_schema = schema
    return trainer, calls, output


def save_tensor(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return dict(path=str(path))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def preserve(trainer, path, progress, error):
    return preserve_first_refusal(trainer, path, progress, error,
        driver_snapshot=lambda *args: dict(actual_cursor=512), atomic_torch_save=save_tensor,
        json_save=save_json, cpu_snapshot=lambda value: value,
        rng_state=lambda: dict(actual_rng_progress=123))


def test_actual_typed_ray_policy_cause_is_recognized():
    cause = PolicySafetyRejected('matched source pre-Adam MAX refusal', dict(phase='log_prob_action_prefix_schema', actual_optimizer_writes=0))
    wrapped = RayTaskError('generate_sequences', 'Synthetic CPU fixture traceback', cause,
                          pid=123, ip='127.0.0.1')
    assert matched_policy_rejection(wrapped) is cause
    assert is_matched_refusal(wrapped) and is_matched_refusal(wrapped.as_instanceof_cause())
    assert not is_matched_refusal(RuntimeError('POLICY_SAFETY_REJECTED: unknown FSDP text'))


@pytest.mark.parametrize('local_failure', ['structure', 'retrieval', 'trace_io'])
def test_driver_local_failure_after_real_rpc_return_saves_all_rank_states(tmp_path, local_failure):
    trainer, calls, output = make_trainer()
    progress = dict(outer=1, first_adam_possible=False, workers_safe_to_snapshot=True, stage='driver-local-before-first-rpc')
    original = trainer.actor_rollout_wg.generate_sequences
    failure = RuntimeError('injected driver local ' + local_failure)
    with pytest.raises(RuntimeError) as raised:
        with track_generation_rpc(trainer.actor_rollout_wg, progress, _schema=trainer._fixture_rpc_schema):
            returned = trainer.actor_rollout_wg.generate_sequences(output)
            assert torch.equal(returned.batch['responses'], output.batch['responses']) and progress['workers_safe_to_snapshot']
            progress['active_rollout_states'] = [SimpleNamespace(source=0, rolling=[1, 2], response=[7])]
            raise failure
    assert raised.value is failure
    assert trainer.actor_rollout_wg.generate_sequences is original
    report = preserve(trainer, tmp_path, progress, failure)
    assert calls == ['generate', 'save-ranks'] and not report['missing_ranks']
    assert report['all_rank_zero_adam_scheduler0_verified'] and report['actual_driver_cursor'] == 512
    saved = torch.load(tmp_path / 'probability-rejection-state/driver.pt', weights_only=False)
    assert saved['last_generation_rpc_output']['tensors']['responses'].tolist() == [[7]] * 8
    assert saved['active_rollout_states'][0]['rolling'] == [1, 2]
    assert saved['rng']['actual_rng_progress'] == 123
    assert not report['automatic_resume_allowed']


def test_unknown_actual_rpc_exception_remains_unsafe_without_rank_recovery(tmp_path):
    failure = RuntimeError('unknown actual FSDP collective failure')
    trainer, calls, output = make_trainer(failure)
    progress = dict(outer=1, first_adam_possible=False, workers_safe_to_snapshot=True, stage='driver-local-before-first-rpc')
    with pytest.raises(RuntimeError) as raised:
        with track_generation_rpc(trainer.actor_rollout_wg, progress, _schema=trainer._fixture_rpc_schema):
            trainer.actor_rollout_wg.generate_sequences(output)
    assert raised.value is failure and not progress['workers_safe_to_snapshot']
    assert trainer.actor_rollout_wg.generate_sequences is trainer.actor_rollout_wg.actor_rollout_generate_sequences
    report = preserve(trainer, tmp_path, progress, failure)
    assert calls == ['generate'] and report['missing_ranks'] == list(range(8))
    assert report['status'] == 'UNKNOWN_FAILURE_DRIVER_RETAINED_WORKER_COVERAGE_INCOMPLETE'


def test_actual_typed_ray_policy_rpc_refusal_permits_zero_adam_snapshot(tmp_path):
    cause = PolicySafetyRejected('matched source support rejection', dict(phase='log_prob_action_prefix_schema', actual_optimizer_writes=0))
    failure = RayTaskError('generate_sequences', 'Synthetic CPU fixture traceback', cause,
                          pid=123, ip='127.0.0.1')
    trainer, calls, output = make_trainer(failure)
    progress = dict(outer=1, first_adam_possible=False, workers_safe_to_snapshot=True)
    with pytest.raises(RayTaskError):
        with track_generation_rpc(trainer.actor_rollout_wg, progress, _schema=trainer._fixture_rpc_schema):
            trainer.actor_rollout_wg.generate_sequences(output)
    assert not progress['workers_safe_to_snapshot']
    report = preserve(trainer, tmp_path, progress, failure)
    assert calls == ['generate', 'save-ranks'] and not report['missing_ranks']
    assert report['all_rank_zero_adam_scheduler0_verified']


def test_driver_local_controller_setup_failure_before_any_rpc_keeps_workers_safe(tmp_path):
    trainer, calls, _ = make_trainer()
    progress = dict(outer=1, first_adam_possible=False, workers_safe_to_snapshot=True, stage='rollout_driver_local_setup')
    failure = RuntimeError('injected driver controller setup before first RPC')
    with pytest.raises(RuntimeError) as raised:
        with track_generation_rpc(trainer.actor_rollout_wg, progress, _schema=trainer._fixture_rpc_schema):
            raise failure
    assert raised.value is failure and progress['workers_safe_to_snapshot']
    report = preserve(trainer, tmp_path, progress, failure)
    assert calls == ['save-ranks'] and not report['missing_ranks']
    assert progress['worker_rpc_boundary_v4']['calls'] == []
    assert not report['automatic_resume_allowed']


@pytest.mark.parametrize('outer,first_adam_possible,writes', [(2, False, 0), (1, True, 0), (1, False, 1)])
def test_typed_policy_refusal_outside_verified_first_pre_adam_bound_makes_no_rank_rpc(
        tmp_path, outer, first_adam_possible, writes):
    cause = PolicySafetyRejected('actual matched source type',
        dict(phase='log_prob_action_prefix_schema', actual_optimizer_writes=writes))
    failure = RayTaskError('compute_log_prob', 'Synthetic CPU fixture traceback', cause,
                          pid=123, ip='127.0.0.1')
    trainer, calls, _ = make_trainer()
    progress = dict(outer=outer, first_adam_possible=first_adam_possible,
                    local_rank_preservation_rpc_schema_verified=True, workers_safe_to_snapshot=False, stage='complete_old_hf')
    report = preserve(trainer, tmp_path, progress, failure)
    assert calls == [] and report['missing_ranks'] == list(range(8))
    assert not report['known_matched_or_completed_worker_boundary']
    assert not report['automatic_resume_allowed']


@pytest.mark.parametrize('local_failure', ['support', 'retrieval', 'trace_io'])
def test_actual_search_rollout_local_failure_after_generation_return_retains_partial_turn(
        tmp_path, monkeypatch, local_failure):
    # Actual frozen SearchRollout.advance and actual RolloutTrace path; CPU
    # DataProto and controlled tokenizer/environment adapters only. Sampling and
    # distributed execution are surrogates, not probability or GPU acceptance.
    from verl import DataProto
    from search_r1.baselines.rollout import SearchRollout, State
    from search_r1.baselines.trajectory_records import RolloutTrace
    from search_r1.baselines import action_identity, trajectory_records
    from calibration_driver_v1 import install_training_prefix_observer
    response_fields = dict(responses=torch.full((8, 1), 7, dtype=torch.int64),
        response_sampled_lengths=torch.ones(8, dtype=torch.int64),
        response_sample_mask=torch.ones((8, 1), dtype=torch.bool))
    if local_failure == 'support':
        response_fields.pop('response_sample_mask')
    output = DataProto.from_dict(response_fields, meta_info={})
    trainer, calls, output = make_trainer(output=output)
    failure = OSError('injected actual driver-local ' + local_failure)
    def execute(predictions, pad_token, active, **kwargs):
        if local_failure == 'retrieval':
            raise failure
        return [''] * len(predictions), [False] * len(predictions), [True] * len(predictions), [False] * len(predictions)
    if local_failure == 'trace_io':
        def failed_rows(*args, **kwargs):
            raise failure
        monkeypatch.setattr(trajectory_records, 'atomic_rows', failed_rows)
    monkeypatch.setattr(action_identity, 'sampled_ids_with_evidence', lambda *args: [7])
    monkeypatch.setattr(action_identity, 'retain_sampled_action',
        lambda *args: SimpleNamespace(actor_ids=[7], metadata=lambda: {}))
    manager = SimpleNamespace(actor_rollout_wg=trainer.actor_rollout_wg,
        tokenizer=SimpleNamespace(pad_token_id=0, pad_token='<pad>'),
        config=SimpleNamespace(actor_token_identity='sampled_action_prefix_v1', max_turns=4,
            force_final_answer=False, max_response_length=500, max_obs_length=500,
            max_prompt_length=4096, max_start_length=2048, num_gpus=8, no_think_rl=False),
        tensor_fn=SimpleNamespace(create_position_ids=lambda mask: mask.cumsum(-1) - 1),
        rollout_trace=RolloutTrace(tmp_path / 'trace', 1, ['q'], 1),
        _postprocess_responses=lambda raw: (raw, ['<answer>x</answer>'] * len(raw)),
        execute_predictions=execute,
        _process_next_obs=lambda observations: torch.zeros((len(observations), 1), dtype=torch.int64))
    engine = SearchRollout(manager, method='gigpo')
    states = [State(0, [1], [1], observation=[1])]
    progress = dict(outer=1, first_adam_possible=False, workers_safe_to_snapshot=True,
                    stage='rollout_driver_local_setup')
    original_advance, turns, prefix_errors = install_training_prefix_observer(tmp_path, 1, progress)
    try:
        with pytest.raises((ValueError, OSError)) as raised:
            with track_generation_rpc(trainer.actor_rollout_wg, progress, _schema=trainer._fixture_rpc_schema):
                engine.advance(states, {})
    finally:
        SearchRollout.advance = original_advance
    if local_failure != 'support':
        assert raised.value is failure
    assert calls == ['generate'] and progress['workers_safe_to_snapshot']
    assert torch.equal(progress['last_generation_rpc_output'].batch['responses'], output.batch['responses'])
    assert progress['active_rollout_states'] is states
    assert progress['active_turn_prefix_before'][0]['generation_prefix'] == [1]
    report = preserve(trainer, tmp_path, progress, raised.value)
    assert calls == ['generate', 'save-ranks'] and not report['missing_ranks']
    saved = torch.load(tmp_path / 'probability-rejection-state/driver.pt', weights_only=False)
    assert saved['active_turn_prefix_before'][0]['generation_prefix'] == [1]
    assert saved['active_rollout_states'][0]['rolling'] == [1]
    assert saved['last_generation_rpc_output']['tensors']['responses'].shape == (8, 1)
    assert saved['active_rollout_engine_state']['rng'] == engine.rng.getstate()
    assert not report['automatic_resume_allowed']
