"""Immutable calibration deployment gates. No claim that CPU/static evidence is GPU acceptance."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'
HELPERS = ROOT / 'tools/policy-safety-native-52befe9-v2'
CONTROLS = ROOT / 'tools/policy-safety-calibration-deployment-52befe9-v2'
SCOPE = 'GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION'
NATIVE_SCOPE = 'GIGPO_POLICY_SAFETY_NATIVE_ONLY'
MANIFEST = '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CHECKPOINT = str(ROOT / 'runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_bound(item):
    path = Path(item['path'])
    require(path.is_absolute() and path.is_file() and sha(path) == item['sha256'], 'Missing/changed proof: ' + str(path))
    return json.loads(path.read_text())


def write_once(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_candidate(value):
    require(value['status'] == 'CANDIDATE_NOT_READY' and value['source_native_acceptance'] is None,
            'A candidate cannot claim native acceptance/READY')
    validate_controls(value)


def validate_controls(value):
    require(value['code_commit'] == SOURCE and value['helpers'] == str(HELPERS), 'Wrong source/helper identity')
    require(value['node'] == 'h100-8' and value['repo'] == str(ROOT / 'repo-policy-safety-52befe9'), 'Wrong node/source path')
    require(value['validation_scope'] == NATIVE_SCOPE and value['calibration_scope'] == SCOPE
            and value['allowed_safety_modes'] == ['calibration'], 'Wrong isolated admission scope')
    require(value['calibration_phase'] == 'probability_probe_then_wait'
            and value['allowed_calibration_phases'] == ['probability_probe_then_wait'],
            'Only the current live probability identity review gate is permitted')
    require(value['restore_checkpoint'] == CHECKPOINT and value['restore_manifest_sha256'] == MANIFEST
            and value['restore_step'] == 25 and value['common_manifest_sha256'] == DATA, 'Not actual healthy full25/common data')
    require(value['calibration_actual_outer_budget'] == 1 and value['optimizer_steps_per_outer'] == 10
            and value['retained_training_updates_contributed'] == 0, 'Calibration cannot replace formal training')
    require(value['target_old_policy_kl'] is None and value['negative_tail_limit'] is None
            and value['probability_tolerance_evidence'] == [], 'Production/probability thresholds are not established')
    for name in ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
                 'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance'):
        require(value[name] is None, 'Current actual probability review must choose tolerances')
    require(value['prefix_mismatch_limit'] == value['action_token_mismatch_limit'] == 0,
            'Structural identity cannot use a tolerance')
    require(value['training_submitted'] is False and value['calibration_submitted'] is False, 'Draft binding cannot claim a submission')
    for name, filename in (('resource_guard', 'resource_guard_v2.py'), ('resource_ownership', 'resource_ownership_v2.py')):
        entry = value[name]
        require(entry['path'] == str(HELPERS / filename), 'Unbound guard path')
        found = [item for item in value['files'] if item['path'] == entry['path']]
        require(len(found) == 1 and found[0]['sha256'] == entry['sha256'], 'Guard lacks unique full-file binding')
    require(len(value['files']) == len({r['path'] for r in value['files']})
            and len(value['evidence']) == len({r['path'] for r in value['evidence']}), 'Duplicate closure paths')


def validate_native_payload(acceptance, result, review, native_binding, guard):
    require(acceptance is not None and acceptance['status'] == 'PASS' and acceptance['code_commit'] == SOURCE,
            'No new actual native acceptance: calibration must not launch')
    job = str(result['job'])
    require(job.isdigit() and job not in ('3259', '3243') and acceptance['job_id'] == job,
            'Failed/obsolete native job cannot gate calibration')
    require(result['status'] == 'PASS' and result['validation_scope'] == NATIVE_SCOPE
            and result['code_commit'] == SOURCE and result['restore_manifest_sha256'] == MANIFEST,
            'Missing current source/full25 actual native PASS')
    require(result['cpu_exit'] == result['gpu_exit'] == 0 and result['fixtures_passed']
            and result['actual_outer_updates'] == 0 and result['cleanup_residual_states'] == {}
            and result['final_gpu']['processes'] == [], 'Incomplete native/cleanup acceptance')
    require(result['guard_identity'] == acceptance['guard_identity'] == guard, 'Actual native guard differs')
    require(review['status'] == 'PASS' and review['scope'] == NATIVE_SCOPE and review['code_commit'] == SOURCE
            and review['job_id'] == job and review['restore_manifest_sha256'] == MANIFEST
            and review['result_sha256'] == acceptance['result_sha256'], 'Native independent review differs from raw result')
    require(native_binding['code_commit'] == SOURCE and native_binding['helpers'] == str(HELPERS)
            and result['binding_sha256'] == acceptance['native_binding_sha256'], 'Native raw binding identity differs')
    require(len(result['phases']) == 2 and len(result['oracle_records']) == 8, 'Tiny-only native is insufficient')
    require({r['rank'] for r in result['oracle_records']} == set(range(8))
            and all(r['status'] == 'PASS' and r['sequence_pg_kl_dp8_micro2_equal']
                    and r['production_denominator_per_mini'] == 32 and r['global_row_denominator'] == 256
                    and r['single_invalid_rank_synchronous_refusal'] and r['no_extra_adam_after_refusal']
                    for r in result['oracle_records']), 'Incomplete native eight-rank oracle scope')
    for phase, rows in zip(('A', 'B'), result['phases']):
        require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8)), 'Missing actual official3B ranks')
        for row in rows:
            require(row['phase'] == phase and row['status'] == 'PASS' and row['full_saved_state_equal']
                    and row['official_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
                    and row['source_manifest_sha256'] == MANIFEST and row['optimizer_steps_before'] == 500
                    and row['optimizer_steps_after'] == 510 and row['scheduler'] == 25
                    and row['native_mini_optimizer_steps'] == 10 and row['gradient_micro_batches'] == 160
                    and row['monitor_guard_count'] == 20 and row['actual_outer_updates'] == 0,
                    'Official3B full25 restore/ten-mini native incomplete')
    for left, right in zip(*result['phases']):
        for field in ('restored_identity', 'next_mini_identity'):
            require({k: v for k, v in left[field].items() if k not in ('pid', 'start_ticks')} ==
                    {k: v for k, v in right[field].items() if k not in ('pid', 'start_ticks')},
                    'Official A/B full-state replay differs')
        require(left['next_mini_identity']['pid'] != right['next_mini_identity']['pid']
                and left['python_numpy_rng_sha256'] == right['python_numpy_rng_sha256'], 'Native replay is not independent/exact')


def read_native_acceptance(acceptance, value):
    require(acceptance is not None, 'Native acceptance null: calibration launch refused')
    result = read_bound(dict(path=acceptance['result_path'], sha256=acceptance['result_sha256']))
    review = read_bound(dict(path=acceptance['review_path'], sha256=acceptance['review_sha256']))
    binding = read_bound(dict(path=acceptance['native_binding_path'], sha256=acceptance['native_binding_sha256']))
    guard = {key: value[key] for key in ('resource_guard', 'resource_ownership')}
    validate_native_payload(acceptance, result, review, binding, guard)
    return result


def validate_ready(value):
    validate_controls(value)
    require(value['status'] == 'REVIEWED_FOR_NATIVE_ONLY'
            and value['readiness'] == 'READY_FOR_PROBABILITY_PROBE_THEN_WAIT_ONLY', 'No independently frozen calibration READY')
    read_native_acceptance(value['source_native_acceptance'], value)
    require(isinstance(value['pad_token_id'], int) and value['pad_token_id'] >= 0, 'Actual official tokenizer pad identity missing')
    require(all(isinstance(r['bytes'], int) and r['bytes'] >= 0 for r in value['files'] + value['evidence']),
            'Physical closure bytes not established')
