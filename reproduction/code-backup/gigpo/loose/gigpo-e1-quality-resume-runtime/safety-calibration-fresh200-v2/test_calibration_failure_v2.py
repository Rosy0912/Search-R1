"""UNEXECUTED CPU refusal fixtures; saves synthetic local evidence only."""
import json
from types import SimpleNamespace
from calibration_failure_v2 import preserve_first_refusal
from calibration_boundary_v2 import MatchedCalibrationRefusal, MARKER


def fixture_trainer():
    calls = []
    def preserve():
        calls.append('rank-save')
        return [dict(rank=rank, saved=rank != 3, zero_adam_verified=True,
                     errors=[] if rank != 3 else ['injected rank save failure']) for rank in range(8)]
    return SimpleNamespace(recovery_sampler=SimpleNamespace(cursor=512, state_dict=lambda: dict(cursor=512)),
        actor_rollout_wg=SimpleNamespace(calibration_preserve_zero_adam=preserve)), calls


def local_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), saved=True)


def damaged_snapshot(*args):
    raise RuntimeError('injected damaged complete observer')


def test_matched_refusal_journals_damaged_observer_and_partial_rank_coverage(tmp_path):
    trainer, calls = fixture_trainer()
    refusal = MatchedCalibrationRefusal(MARKER + '/generate/after_original_before_repeat')
    report = preserve_first_refusal(trainer, tmp_path,
        dict(stage='generate_rollout', workers_safe_to_snapshot=False), refusal,
        driver_snapshot=damaged_snapshot, atomic_torch_save=local_save, json_save=lambda path, value: local_save(value, path),
        cpu_snapshot=lambda value: value, rng_state=lambda: dict(actual_rng_progress=123))
    assert calls == ['rank-save'] and report['missing_ranks'] == [3]
    assert report['actual_driver_cursor'] == 512 and not report['exact_resume_accepted']
    assert not report['automatic_resume_allowed'] and report['preservation_errors']
    saved = json.loads((tmp_path / 'probability-rejection-state/driver.pt').read_text())
    assert saved['actual_cursor'] == 512 and saved['rng']['actual_rng_progress'] == 123
    assert MARKER in saved['original_error']


def test_unknown_original_failure_never_dispatches_rank_recovery(tmp_path):
    trainer, calls = fixture_trainer()
    report = preserve_first_refusal(trainer, tmp_path,
        dict(stage='unknown_model_rpc', workers_safe_to_snapshot=False), RuntimeError('unknown FSDP failure with misleading text ' + MARKER),
        driver_snapshot=lambda *args: dict(actual_cursor=512), atomic_torch_save=local_save, json_save=lambda path, value: local_save(value, path),
        cpu_snapshot=lambda value: value, rng_state=lambda: dict(actual_rng_progress=123))
    assert calls == [] and report['missing_ranks'] == list(range(8))
    assert report['status'] == 'UNKNOWN_FAILURE_DRIVER_RETAINED_WORKER_COVERAGE_INCOMPLETE'
    assert not report['validated_resume_checkpoint'] and not report['automatic_resume_allowed']
