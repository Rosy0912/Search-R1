"""Fail-closed acceptance of real2933 evidence; no GPU/model needed."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('semantic_device_gate', ROOT/'reproduction/baselines/opsd-runtime-20261003/semantic_device_gate.py')
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
RAW = (Path(__file__).parent/'fixtures/opsd_semantic_device_2933.json').read_text()
LOG = 'Traceback (most recent call last):\n  File "gpu_teacher_probe-factor-dtype.py", line 1, in main\n'+GATE.EXPECTED_ASSERTION+'\n'


def validate(value, code=1, log=LOG):
    return GATE.validate_diagnostic(value, code, log, json.loads(RAW)['allocation'], json.loads(RAW)['model'])


def test_real_reviewed_evidence_has_explicit_separate_policy():
    result = validate(GATE.strict_json(RAW))
    assert result['passed'] is True
    assert result['original_bf16_parity_passed'] is False
    assert result['forced_upstream_gradient_control'] == 'numeric_exact'
    assert result['all_parameter_elements_checked'] == 3085938688


@pytest.mark.parametrize('path,value', [
    (('passed',), True),
    (('allocation','job_id'), '29330'),
    (('model',), '/other/model'),
    (('teacher_gpu_runtime','parameter_dtypes'), ['torch.bfloat16']),
    (('teacher_gpu_runtime','active_device'), 'cuda:1'),
    (('teacher_gpu_runtime','matmul_tf32'), True),
    (('teacher_after_return_runtime','parameter_devices'), ['cuda:0']),
    (('teacher_audit','teacher/gradient_tensors_per_rank'), 1),
    (('teacher_probability_max_abs',), 1.1e-5),
    (('loss_absolute_difference',), 1.1e-5),
    (('student_gradient_relative_l2',), float('nan')),
    (('tolerances','gradient_relative_l2'), .1),
    (('identical_cpu_teacher_bf16_backward_repeat_controls',0,'relative_l2'), 1e-12),
    (('identical_gpu_teacher_bf16_backward_repeat_controls',0,'loss_absolute_difference'), 1e-12),
    (('identical_gpu_teacher_bf16_backward_repeat_controls',0,'parameter_elements_compared'), 1),
    (('cached_student_logit_fkl_gradient_comparison','gradient_max_absolute_difference'), 2e-6),
    (('cached_student_logit_fkl_gradient_comparison','clipping_boundary_crossings'), 1),
    (('fp32_student_reference','relative_l2'), .002),
    (('fp32_student_reference','parameter_elements_compared'), 1),
    (('fp32_student_reference','repeated_identical_target_control','max_absolute_difference'), 1e-12),
    (('bf16_backward_factorization','production_gradient_or_loss_modified'), True),
    (('bf16_backward_factorization','actual_projection_cast_checks',0,'exact'), False),
    (('bf16_backward_factorization','actual_hf_logits_gradient_vs_cached_analytic',0,'changed_elements'), 1),
    (('bf16_backward_factorization','actual_hf_logits_gradient_vs_cached_analytic',0,'actual_opsd_loss_chain_capture','chunk_lengths'), [9,16]),
    (('bf16_backward_factorization','actual_hf_logits_gradient_vs_cached_analytic',1,'actual_opsd_loss_chain_capture','teacher_projection_devices'), ['cpu','cpu']),
    (('bf16_backward_factorization','actual_hf_logits_gradient_vs_cached_analytic',1,'actual_opsd_loss_chain_capture','teacher_projection_dtypes'), ['torch.bfloat16','torch.bfloat16']),
    (('bf16_backward_factorization','forced_identical_projection_gradient_all_parameter_comparison','difference_l2'), 1e-12),
    (('bf16_backward_factorization','forced_identical_projection_gradient_all_parameter_comparison','reference_l2'), 0),
    (('bf16_backward_factorization','error'), 'OOM'),
])
def test_rejects_unrelated_or_incomplete_scientific_evidence(path, value):
    j = GATE.strict_json(RAW)
    target = j
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((ValueError, KeyError, TypeError)):
        validate(j)


@pytest.mark.parametrize('code', [0,2,137,-9,-15])
def test_no_exit_or_signal_is_reinterpreted_as_expected_assertion(code):
    with pytest.raises(ValueError):
        validate(GATE.strict_json(RAW), code=code)


@pytest.mark.parametrize('log', ['', 'RuntimeError: CUDA out of memory\n',
    'Traceback (most recent call last):\nAssertionError: other failure\n',
    LOG+'RuntimeError: later failure\n', LOG+LOG])
def test_only_unique_terminal_original_assertion_is_expected(log):
    with pytest.raises(ValueError):
        validate(GATE.strict_json(RAW), log=log)


@pytest.mark.parametrize('text', ['{"passed":false,"passed":true}', '{"x":NaN}', '{"x":Infinity}', '{'])
def test_json_parser_rejects_ambiguous_or_nonfinite_or_partial_output(text):
    with pytest.raises(ValueError):
        GATE.strict_json(text)


def test_optional_diagnostic_sections_are_mandatory_here():
    for key in ('fp32_student_reference','bf16_backward_factorization','teacher_audit'):
        j = GATE.strict_json(RAW)
        del j[key]
        with pytest.raises(KeyError):
            validate(j)


def test_pinned_helper_keeps_original_assertion_and_hash():
    import hashlib
    source = ROOT/'reproduction/baselines/opsd-runtime-20261003/gpu_teacher_probe-factor-dtype.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == GATE.HELPER_SHA256
    assert "raise AssertionError('CPU/GPU FP32 teacher numerical equivalence gate failed')" in source.read_text()


def prepare_runner(monkeypatch, tmp_path, stack, mode='expected'):
    import shutil
    from types import SimpleNamespace
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    shutil.copyfile(ROOT/'reproduction/baselines/opsd-runtime-20261003/gpu_teacher_probe-factor-dtype.py', runtime/'gpu_teacher_probe-factor-dtype.py')
    monkeypatch.setattr(GATE, '__file__', str(runtime/'semantic_device_gate.py'))
    allocation = GATE.strict_json(RAW)['allocation']
    monkeypatch.setattr(GATE, 'allocation_scope', lambda:allocation)
    evidence = tmp_path/'evidence'
    monkeypatch.setenv('OPSD_EVIDENCE', str(evidence))
    monkeypatch.setenv('OPSD_TEACHER', GATE.strict_json(RAW)['model'])
    streams = [stack.enter_context((tmp_path/f'lease-{i}').open('w')) for i in range(8)]
    fds = [stream.fileno() for stream in streams]
    monkeypatch.setenv('SEARCH_R1_GPU_LEASES', json.dumps([dict(fd=fd) for fd in fds]))
    calls = []
    def run(command, stdout, stderr, pass_fds):
        calls.append(command)
        assert set(pass_fds) == set(fds)
        assert command[-3:] == ['--fp32-reference','--flash-deterministic','--factor-bf16-backward']
        path = Path(command[command.index('--output')+1])
        if mode != 'missing_json':
            path.write_text(RAW)
        stdout.write(LOG if mode != 'other_exception' else 'Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n')
        stdout.flush()
        return SimpleNamespace(returncode=-9 if mode=='signal' else 1)
    monkeypatch.setattr(GATE.subprocess, 'run', run)
    return evidence, runtime, calls


def test_runner_separately_records_semantic_success_without_rewriting_failed_diagnostic(monkeypatch,tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        evidence, runtime, calls = prepare_runner(monkeypatch,tmp_path,stack)
        GATE.main()
    assert len(calls) == 1
    assert json.loads((evidence/'semantic-device-gate.json').read_text())['passed'] is True
    assert json.loads((evidence/'gpu-teacher-semantic-diagnostic.json').read_text())['passed'] is False


@pytest.mark.parametrize('mode', ['missing_json','signal','other_exception'])
def test_runner_cannot_reinterpret_partial_failure_or_signal(monkeypatch,tmp_path,mode):
    from contextlib import ExitStack
    with ExitStack() as stack:
        evidence, runtime, calls = prepare_runner(monkeypatch,tmp_path,stack,mode)
        with pytest.raises(ValueError):
            GATE.main()
    assert json.loads((evidence/'semantic-device-gate.json').read_text())['passed'] is False


def test_runner_rejects_source_drift_before_launch(monkeypatch,tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        evidence, runtime, calls = prepare_runner(monkeypatch,tmp_path,stack)
        (runtime/'gpu_teacher_probe-factor-dtype.py').write_text('raise SystemExit(1)\n')
        with pytest.raises(ValueError, match='source hash mismatch'):
            GATE.main()
    assert calls == []
    assert json.loads((evidence/'semantic-device-gate.json').read_text())['passed'] is False


def test_runner_refuses_stale_pass_artifact_without_overwriting_it(monkeypatch,tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        evidence, runtime, calls = prepare_runner(monkeypatch,tmp_path,stack)
        evidence.mkdir()
        stale = evidence/'semantic-device-gate.json'
        stale.write_text('{"passed":true,"job_id":"old"}')
        with pytest.raises(ValueError, match='stale'):
            GATE.main()
    assert calls == []
    assert stale.read_text() == '{"passed":true,"job_id":"old"}'
