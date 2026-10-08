"""Explicit precision-referenced device policy; original BF16 parity failure is retained.

Only a pinned diagnostic's specific terminal parity AssertionError is expected.
This gate proves the reviewed short-context objective/device scenario, not BF16
update equivalence, model quality, GPU coexistence, or five-iteration acceptance.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

HELPER_SHA256 = 'e98dbb92850d4e5b89a17209e209f1c7bcbc9dcf3712981d2b1b6e98ddf044dc'
EXPECTED_ASSERTION = 'AssertionError: CPU/GPU FP32 teacher numerical equivalence gate failed'
PARAMETERS = 3085938688
VOCABULARY = 151936


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def strict_json(text):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, 'Duplicate JSON field: ' + key)
            out[key] = value
        return out
    def bad_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)


def numeric(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Missing/nonfinite numeric observation')
    return value


def bounded(value, upper):
    require(0 <= numeric(value) <= upper, 'Numeric criterion exceeded')


def clean_tree(value):
    if isinstance(value, dict):
        require(not set(value).intersection(('error', 'error_type', 'traceback')), 'Diagnostic contains another error')
        for item in value.values():
            clean_tree(item)
    elif isinstance(value, list):
        for item in value:
            clean_tree(item)
    elif type(value) in (int, float):
        numeric(value)


def allocation_scope():
    job = os.environ.get('SLURM_JOB_ID')
    require(job and job.isdigit(), 'Missing numeric owned Slurm job')
    node = socket.gethostname().split('.')[0]
    cgroup = Path('/proc/self/cgroup').read_text()
    require(node == 'h100-10' and os.environ.get('SLURM_JOB_NODELIST') == 'h100-10', 'Requires single h100-10 allocation')
    require(re.search(r'(?:^|/)job_' + re.escape(job) + r'(?:/|$)', cgroup, re.MULTILINE), 'Process outside declared job cgroup')
    return dict(job_id=job, hostname=socket.gethostname(), node_short=node,
                slurm_job_nodelist='h100-10', own_job_cgroup=cgroup)


def exact_all_gradients(observation, include_loss=False):
    require(observation['parameter_elements_compared'] == PARAMETERS, 'Incomplete all-parameter comparison')
    for field in ('relative_l2', 'difference_l2', 'max_absolute_difference'):
        bounded(observation[field], 0)
    require(numeric(observation['reference_l2']) > 0 and numeric(observation['reference_max_abs']) > 0,
            'Zero reference gradients do not establish the control')
    if include_loss:
        bounded(observation['loss_absolute_difference'], 0)


def validate_diagnostic(j, returncode, log, allocation, model):
    """Pure evidence validator; thresholds belong to this policy, not JSON input."""
    require(returncode == 1, 'Only the specific original assertion exit 1 is expected')
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    require(lines and lines[-1] == EXPECTED_ASSERTION and log.count(EXPECTED_ASSERTION) == 1,
            'Missing/ambiguous original parity assertion marker')
    require(log.count('Traceback (most recent call last):') == 1, 'Unexpected additional exception or no traceback')
    clean_tree(j)
    require(j['allocation'] == allocation and j['job_id'] == allocation['job_id'] and
            j['hostname'] == allocation['hostname'], 'Diagnostic belongs to another allocation')
    require(j['model'] == model, 'Unexpected model source')
    require(j['passed'] is False and j['diagnostic_controls_do_not_relax_original_gate'] is True,
            'Original strict parity failure must remain explicit')
    require(j['flash_attention_deterministic_env'] == '1' and j['flash_deterministic_requested_diagnostic_only'] is True,
            'Missing controlled diagnostic deterministic setting')
    require(j['teacher_dtype'] == 'float32' and j['teacher_attention'] == 'eager' and j['tf32'] is False and
            j['student_parameter_dtypes'] == ['torch.bfloat16'], 'Precision/runtime contract changed')
    for field, device in (('teacher_cpu_runtime', 'cpu'), ('teacher_gpu_runtime', 'cuda:0'),
                          ('teacher_after_return_runtime', 'cpu')):
        r = j[field]
        require(r['parameter_dtypes'] == ['torch.float32'] and r['attention'] == 'eager' and
                r['parameter_devices'] == [device] and r['active_device'] == device and
                r['matmul_tf32'] is False and r['cudnn_tf32'] is False, 'Teacher runtime invariant failed')
    audit = j['teacher_audit']
    for field in ('teacher/trainable_parameters_per_rank', 'teacher/gradient_tensors_per_rank',
                  'teacher/student_optimizer_membership_per_rank'):
        require(type(audit[field]) is int and audit[field] == 0, 'Teacher is not isolated/frozen')
    bounded(j['teacher_probability_max_abs'], 1e-5)
    bounded(j['teacher_cpu_gpu_logits_max_abs'], .001)
    require(-1e-6 <= numeric(j['teacher_kl_max']) <= 1e-5, 'Teacher distribution KL criterion exceeded')
    bounded(j['loss_absolute_difference'], 1e-5)
    numeric(j['cpu_teacher_loss']); numeric(j['gpu_teacher_loss'])
    require(abs(j['cpu_teacher_loss']-j['gpu_teacher_loss']) == j['loss_absolute_difference'], 'Loss metadata inconsistent')
    require(j['all_student_gradient_parameters_compared'] == PARAMETERS, 'Incomplete BF16 parameter audit')
    bf = j['bf16_student_migration_gradient_details']
    require(bf['parameter_elements_compared'] == PARAMETERS and
            bf['relative_l2'] == j['student_gradient_relative_l2'] and
            bf['max_absolute_difference'] == j['student_gradient_max_abs'], 'BF16 observations inconsistent')
    require(numeric(bf['relative_l2']) > .01 or numeric(bf['max_absolute_difference']) > .0005,
            'Original failure is not the stated BF16 parity failure')
    require(j['tolerances'] == dict(probability_max_abs=1e-5, loss_absolute=1e-5,
                                   gradient_relative_l2=.01, gradient_max_abs=.0005), 'Original criteria altered')
    for field in ('identical_cpu_teacher_bf16_backward_repeat_controls',
                  'identical_gpu_teacher_bf16_backward_repeat_controls'):
        require(len(j[field]) == 3, 'Missing three repeat controls')
        for r in j[field]:
            exact_all_gradients(r, include_loss=True)
    leaf = j['cached_student_logit_fkl_gradient_comparison']
    require(leaf['passed'] is True and leaf['source_dtype'] == 'torch.float32' and leaf['vocab_size'] == VOCABULARY,
            'Full-vocabulary FP32 leaf test failed')
    bounded(leaf['loss_absolute_difference'], 1e-5)
    bounded(leaf['gradient_relative_l2'], .001)
    bounded(leaf['gradient_max_absolute_difference'], 1e-6)
    require(len(leaf['explicit_derivative_max_errors']) == 2, 'Missing explicit derivative checks')
    for value in leaf['explicit_derivative_max_errors']:
        bounded(value, 1e-7)
    require(leaf['clipping_boundary_crossings'] == 0 and
            type(leaf['clipped_components_cpu']) is int and leaf['clipped_components_cpu'] >= 0 and
            leaf['clipped_components_cpu'] == leaf['clipped_components_gpu'], 'Changed clipping boundary')
    fp = j['fp32_student_reference']
    require(fp['passed'] is True and fp['parameter_elements_compared'] == PARAMETERS and
            fp['parameter_dtypes'] == ['torch.float32'] and fp['attention'] == 'eager', 'Missing full FP32/eager reference')
    bounded(fp['relative_l2'], .001); bounded(fp['max_absolute_difference'], 1e-4)
    bounded(fp['loss_absolute_difference'], 1e-5)
    exact_all_gradients(fp['repeated_identical_target_control'])
    factor = j['bf16_backward_factorization']
    require(factor['passed'] is True and factor['stage'] == 'completed_forced_gradient_control' and
            factor['student_projection_forward_identical'] is True and
            factor['production_gradient_or_loss_modified'] is False, 'Missing complete diagnostic-only factorization')
    require(len(factor['actual_projection_cast_checks']) == 2 and
            len(factor['actual_hf_logits_gradient_vs_cached_analytic']) == 2, 'Incomplete native-chain captures')
    width = None
    for index, (cast, chain) in enumerate(zip(factor['actual_projection_cast_checks'], factor['actual_hf_logits_gradient_vs_cached_analytic'])):
        require(cast['exact'] is True and cast['projection_gradient_dtype'] == 'torch.bfloat16' and
                cast['hf_logits_gradient_dtype'] == 'torch.bfloat16', 'Projection/native cast mismatch')
        for field in ('relative_l2', 'max_absolute_difference', 'changed_elements'):
            bounded(chain[field], 0)
        require(chain['exact_after_matching_actual_output_dtype'] is True and chain['passed_absolute_1e_minus_7'] is True and
                chain['actual_hf_logits_gradient_dtype'] == 'torch.bfloat16' and
                chain['expected_analytic_gradient_dtype_after_cast'] == 'torch.bfloat16' and
                chain['fp32_leaf_gradient_before_cast_dtype'] == 'torch.float32', 'Native analytical replay mismatch')
        capture = chain['actual_opsd_loss_chain_capture']
        actual_width = capture['total_response_positions']
        require(type(actual_width) is int and actual_width > 0, 'Invalid response width')
        width = actual_width if width is None else width
        require(actual_width == width, 'CPU/GPU response coverage differs')
        lengths = [min(16, width-start) for start in range(0, width, 16)]
        count = len(lengths)
        device = 'cpu' if index == 0 else 'cuda:0'
        require(capture['ordered_response_coverage_verified'] is True and
                capture['chunk_lengths'] == lengths and capture['expected_chunk_lengths'] == lengths and
                capture['captured_tensor_count'] == count and capture['chunk_shapes'] == [[1, n, VOCABULARY] for n in lengths] and
                capture['teacher_projection_dtypes'] == ['torch.float32']*count and
                capture['teacher_projection_devices'] == [device]*count and
                capture['actual_hf_output_dtype'] == 'torch.bfloat16' and capture['analytic_replay_dtype'] == 'torch.float32' and
                capture['analytic_replay_device'] == 'cuda:0' and chain['total_elements'] == width*VOCABULARY,
                'Actual teacher chunks/dtype/device/order coverage inconsistent')
    exact_all_gradients(factor['forced_identical_projection_gradient_all_parameter_comparison'])
    projection = factor['actual_projection_gradient_difference']
    require(numeric(projection['relative_l2']) > 0 and numeric(projection['max_absolute_difference']) > 0 and
            0 < projection['changed_elements'] <= projection['total_elements'], 'Missing actual upstream BF16 change')
    require(projection['total_elements'] >= width*VOCABULARY, 'Invalid projection element count')
    rounding = factor['cached_leaf_gradient_bf16_rounding']
    require(rounding['total_elements'] == width*VOCABULARY and
            0 < rounding['changed_elements'] <= rounding['total_elements'] and
            numeric(rounding['relative_l2']) > 0 and numeric(rounding['max_absolute_difference']) > 0,
            'Missing BF16 rounding evidence')
    return dict(passed=True, policy='precision_referenced_device_v1', original_bf16_parity_passed=False,
        bf16_gradient_relative_l2=j['student_gradient_relative_l2'], bf16_gradient_max_abs=j['student_gradient_max_abs'],
        all_parameter_elements_checked=PARAMETERS, forced_upstream_gradient_control='numeric_exact',
        scope='Reviewed short synthetic actual-3B objective/device scenario; BF16 update equivalence, training efficacy, GPU coexistence, five iterations and HF reload remain separate acceptance requirements')


def main():
    allocation = allocation_scope()
    out = Path(os.environ['OPSD_EVIDENCE'])
    model = os.environ['OPSD_TEACHER']
    helper = Path(__file__).with_name('gpu_teacher_probe-factor-dtype.py')
    record = out/'semantic-device-gate.json'
    diagnostic = out/'gpu-teacher-semantic-diagnostic.json'
    log_path = out/'semantic-device-diagnostic.log'
    require(not record.exists() and not diagnostic.exists() and not log_path.exists(), 'Refuse stale/reused gate evidence')
    out.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, allocation=allocation, policy='precision_referenced_device_v1')
    try:
        actual_hash = hashlib.sha256(helper.read_bytes()).hexdigest()
        require(actual_hash == HELPER_SHA256, 'Diagnostic helper source hash mismatch')
        leases = strict_json(os.environ['SEARCH_R1_GPU_LEASES'])
        fds = tuple(row['fd'] for row in leases)
        require(len(fds) == 8 and len(set(fds)) == 8 and all(type(fd) is int and fd >= 3 for fd in fds), 'Missing eight inherited GPU leases')
        for fd in fds:
            os.fstat(fd)
        command = [sys.executable, '-u', str(helper), '--model', model, '--output', str(diagnostic),
                   '--fp32-reference', '--flash-deterministic', '--factor-bf16-backward']
        with log_path.open('x') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, pass_fds=fds)
        require(diagnostic.is_file(), 'Diagnostic exited without its complete JSON')
        raw = diagnostic.read_text()
        report.update(validate_diagnostic(strict_json(raw), child.returncode, log_path.read_text(), allocation, model))
        report.update(diagnostic_sha256=hashlib.sha256(raw.encode()).hexdigest(), helper_sha256=actual_hash,
                      diagnostic_returncode=child.returncode, original_expected_assertion=EXPECTED_ASSERTION,
                      diagnostic_path=str(diagnostic), diagnostic_log=str(log_path))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        record.write_text(json.dumps(report, indent=2))
        raise
    record.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
