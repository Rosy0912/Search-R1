"""Immutable calibration deployment gates. No claim that CPU/static evidence is GPU acceptance."""
import hashlib
import base64
import gzip
import json
import os
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'
NATIVE_HELPERS = ROOT / 'tools/policy-safety-native-52befe9-v2'
HELPERS = ROOT / 'tools/policy-safety-calibration-52befe9-v6'
CONTROLS = ROOT / 'tools/policy-safety-calibration-controls-52befe9-v6'
SCOPE = 'GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION'
NATIVE_SCOPE = 'GIGPO_POLICY_SAFETY_NATIVE_ONLY'
MANIFEST = '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CHECKPOINT = str(ROOT / 'runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25')
STARTUP_SCOPE = 'BOUNDED_POST_RETRIEVER_IDLE_RECHECK_ONLY'
STARTUP_PROOF_ROLES = ('runtime_manifest', 'runtime_diff', 'runtime_cpu', 'runtime_review',
                       'failure_payload', 'failure_review', 'failure_cgroup')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_bound(item, compressed=False):
    path = Path(item['path'])
    require(path.is_absolute() and path.is_file() and sha(path) == item['sha256'], 'Missing/changed proof: ' + str(path))
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if compressed else payload)


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
    require(value['controls'] == str(CONTROLS), 'Wrong isolated deployment control path')
    runtime = value['runtime_helpers']
    require(len(runtime) == 13 and len({r['path'] for r in runtime}) == 13
            and all(Path(r['path']).parent == HELPERS for r in runtime), 'Runtime copies not separately isolated')
    by_path = {r['path']: r for r in value['files']}
    require(all(by_path.get(r['path']) == r for r in runtime), 'Runtime copy missing from full physical file closure')
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
        require(entry['path'] == str(NATIVE_HELPERS / filename), 'Unbound actual native guard path')
        found = [item for item in value['files'] if item['path'] == entry['path']]
        require(len(found) == 1 and found[0]['sha256'] == entry['sha256'], 'Guard lacks unique full-file binding')
    require(len(value['files']) == len({r['path'] for r in value['files']})
            and len(value['evidence']) == len({r['path'] for r in value['evidence']}), 'Duplicate closure paths')
    require(value['startup_change_scope'] == STARTUP_SCOPE
            and value['runtime_changed_filenames'] == ['calibration_run_v1.py'],
            'Only bounded startup cooldown may differ from frozen v4 runtime')
    proofs = value['startup_control_proofs']
    require(set(proofs) == set(STARTUP_PROOF_ROLES)
            and len({record['path'] for record in proofs.values()}) == len(proofs),
            'Incomplete/duplicate explicit startup/failure proof binding')
    evidence = {record['path']: record for record in value['evidence']}
    require(all(Path(record['path']).parent == ROOT / 'logs'
                and evidence.get(record['path']) == record for record in proofs.values()),
            'Startup proof lacks exact immutable evidence coverage')


def validate_terminal_payload(acceptance, result, review, native_binding, terminal, physical):
    """Require saved native terminal originals, not another currently running job."""
    job = str(result['job'])
    run = str(ROOT / ('runs/gigpo-e1-policy-safety-native-job' + job))
    require(acceptance['result_path'] == str(Path(run) / 'result.json'), 'Native result not in its actual job directory')
    require(acceptance['terminal_path'] == str(ROOT / ('logs/quality10-native-terminal-' + job + '-raw-v1.json')),
            'Terminal receipt path not the exact native job')
    fields = terminal['fields']
    require(fields['JobId'] == job and fields['NodeList'] == 'h100-8'
            and fields['JobState'] == 'COMPLETED' and fields['ExitCode'] == '0:0'
            and str(fields['Requeue']) == str(fields['Restarts']) == '0',
            'Native did not actually complete exactly once on h100-8 with exit0:0')
    require(acceptance['terminal_state'] == 'COMPLETED' and acceptance['terminal_exit_code'] == '0:0'
            and str(acceptance['terminal_requeue']) == str(acceptance['terminal_restarts']) == '0'
            and acceptance['cgroup_removed'] is True, 'Native acceptance specification contradicts actual clean terminal')
    require(review['terminal_path'] == acceptance['terminal_path']
            and review['terminal_sha256'] == acceptance['terminal_sha256']
            and review['terminal_state'] == 'COMPLETED' and review['terminal_exit_code'] == '0:0'
            and str(review['terminal_requeue']) == str(review['terminal_restarts']) == '0'
            and review['cgroup_removed'] is True
            and review['physical_payload_path'] == acceptance['physical_payload_path']
            and review['physical_payload_sha256'] == acceptance['physical_payload_sha256'],
            'Native independent terminal/physical/cgroup acceptance missing or contradictory')
    require(physical['status'] == 'ORIGINAL_NATIVE_TERMINAL_AND_PHYSICAL_EVIDENCE_UNREVIEWED'
            and physical['job'] == job and physical['run'] == run
            and physical['result'] == result and physical['terminal'] == terminal,
            'Actual original native physical/terminal payload identity differs')
    require(physical['own_cgroups'] == [] and physical['own_job_queue'].strip() == '',
            'Native cgroup/job queue not released')
    # An empty glob cannot distinguish unreadable ancestors from true absence.
    release = physical['cgroup_release']
    cgroup = '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_' + job
    require('..' not in Path(result['cgroup']).parts
            and (result['cgroup'] == cgroup or result['cgroup'].startswith(cgroup + '/'))
            and release['result_cgroup'] == result['cgroup'] and release['job_id'] == job and release['path'] == cgroup
            and release['ancestor_path'] == '/sys/fs/cgroup/freezer/slurm_h100-8'
            and release['ancestor_readable'] is True and release['stat_outcome'] == 'ENOENT'
            and release['scope_stat_outcome'] == 'ENOENT' and release['removed'] is True,
            'Native actual readable-ancestor job and original scope disappearance not proved')
    require(physical['current_compute']['returncode'] == 0 and isinstance(physical['current_node_queue'], str),
            'Actual target GPU/queue observation unreadable')
    # Current other-job GPU processes are not historical native allocation residue.
    require(result['cleanup_residual_states'] == {} and result['final_gpu']['processes'] == [],
            'Saved native exclusive-allocation cleanup was incomplete')
    originals = physical['originals']
    require(len(originals) == len({r['path'] for r in originals}), 'Duplicate native original paths')
    required = {
        acceptance['result_path']: (acceptance['result_sha256'], result),
        acceptance['terminal_path']: (acceptance['terminal_sha256'], terminal),
        acceptance['native_binding_path']: (acceptance['native_binding_sha256'], native_binding),
    }
    actual = {}
    for record in originals:
        payload = base64.b64decode(record['content_base64'], validate=True)
        require(Path(record['path']).is_absolute() and len(payload) == record['bytes']
                and hashlib.sha256(payload).hexdigest() == record['sha256'], 'Native original bytes/SHA corrupt')
        actual[record['path']] = record
        if record['path'] in required:
            expected, value = required[record['path']]
            require(record['sha256'] == expected and json.loads(payload) == value,
                    'Native required original differs from separately read bound receipt')
    require(set(required).issubset(actual), 'Missing required actual native result/terminal/binding original')
    paths = physical['physical']
    require(len(paths) == len({r['path'] for r in paths}) and paths
            and all('..' not in Path(r['path']).parts and Path(r['path']).is_relative_to(run)
                    and isinstance(r['bytes'], int) and r['bytes'] >= 0
                    and isinstance(r['sha256'], str) and len(r['sha256']) == 64 for r in paths),
            'Incomplete or foreign native physical file inventory')
    result_record = [r for r in paths if r['path'] == acceptance['result_path']]
    require(result_record == [{k: actual[acceptance['result_path']][k] for k in ('path', 'bytes', 'sha256')}],
            'Native result actual physical identity does not match original')
    by_path = {r['path']: r for r in paths}
    for path, record in actual.items():
        if Path(path).is_relative_to(run):
            require(by_path.get(path) == {k: record[k] for k in ('path', 'bytes', 'sha256')},
                    'Native raw run artifact and physical inventory disagree')


def validate_native_payload(acceptance, result, review, native_binding, guard, terminal, physical):
    require(acceptance is not None and acceptance['status'] == 'PASS' and acceptance['code_commit'] == SOURCE,
            'No new actual native acceptance: calibration must not launch')
    job = str(result['job'])
    require(job.isdigit() and job not in ('3259', '3243') and acceptance['job_id'] == job,
            'Failed/obsolete native job cannot gate calibration')
    require(result['status'] == 'PASS' and result['validation_scope'] == NATIVE_SCOPE
            and result['code_commit'] == SOURCE and result['restore_step'] == 25
            and result['restore_manifest_sha256'] == MANIFEST,
            'Missing current source/full25 actual native PASS')
    require(result['cpu_exit'] == result['gpu_exit'] == 0 and result['fixtures_passed']
            and result['actual_outer_updates'] == 0 and result['cleanup_residual_states'] == {}
            and result['final_gpu']['processes'] == [], 'Incomplete native/cleanup acceptance')
    require(result['guard_identity'] == acceptance['guard_identity'] == guard, 'Actual native guard differs')
    require(review['status'] == 'PASS' and review['scope'] == NATIVE_SCOPE and review['code_commit'] == SOURCE
            and review['job_id'] == job and review['restore_manifest_sha256'] == MANIFEST
            and review['result_sha256'] == acceptance['result_sha256'], 'Native independent review differs from raw result')
    require(native_binding['code_commit'] == SOURCE and native_binding['helpers'] == str(NATIVE_HELPERS)
            and native_binding['restore_step'] == 25 and native_binding['restore_checkpoint'] == CHECKPOINT
            and native_binding['restore_manifest_sha256'] == MANIFEST
            and native_binding['fresh_model_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
            and result['binding_sha256'] == acceptance['native_binding_sha256']
            and {k: native_binding[k] for k in ('resource_guard', 'resource_ownership')} == guard,
            'Native raw binding identity differs')
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
    validate_terminal_payload(acceptance, result, review, native_binding, terminal, physical)


def read_native_acceptance(acceptance, value):
    require(acceptance is not None, 'Native acceptance null: calibration launch refused')
    result = read_bound(dict(path=acceptance['result_path'], sha256=acceptance['result_sha256']))
    review = read_bound(dict(path=acceptance['review_path'], sha256=acceptance['review_sha256']))
    binding = read_bound(dict(path=acceptance['native_binding_path'], sha256=acceptance['native_binding_sha256']))
    terminal = read_bound(dict(path=acceptance['terminal_path'], sha256=acceptance['terminal_sha256']))
    physical = read_bound(dict(path=acceptance['physical_payload_path'], sha256=acceptance['physical_payload_sha256']), compressed=True)
    guard = {key: value[key] for key in ('resource_guard', 'resource_ownership')}
    validate_native_payload(acceptance, result, review, binding, guard, terminal, physical)
    return result


def validate_ready(value):
    validate_controls(value)
    require(value['status'] == 'REVIEWED_FOR_NATIVE_ONLY'
            and value['readiness'] == 'READY_FOR_PROBABILITY_PROBE_THEN_WAIT_ONLY', 'No independently frozen calibration READY')
    read_native_acceptance(value['source_native_acceptance'], value)
    require(isinstance(value['pad_token_id'], int) and value['pad_token_id'] >= 0, 'Actual official tokenizer pad identity missing')
    require(all(isinstance(r['bytes'], int) and r['bytes'] >= 0 for r in value['files'] + value['evidence']),
            'Physical closure bytes not established')
