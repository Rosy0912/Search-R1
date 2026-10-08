"""Exact final fresh200 deployment inputs. Stdlib only; no native/GPU gate."""
import hashlib
import json
import os
import re
import stat
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
COMMON = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
SHARED = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo')
CONTROLS = ROOT / 'tools/fresh200-deployment-controls-v8'
SOURCE_RUNTIME_CPU_HELPER_V3 = dict(path=str(ROOT / 'tools/management/continuous-cpu-once-v3.py'), bytes=19553,
    sha256='46f254bf0827cdbd000b3711d83782c00203c990ff98b9f70d493aaf5f18117c')
SOURCE_RUNTIME_CPU_HELPER = dict(path=str(ROOT / 'tools/management/continuous-cpu-once-v4.py'), bytes=19553,
    sha256='520819b9e7954ecb66b45c17a673af40497b0e4a8f6ee045efb78352f84ad46f')
STATE = ROOT / 'logs/fresh200-live200-submit-v8'
CANDIDATE, PHYSICAL, READY, BINDING, INTENT, RECEIPT = (
    STATE / name for name in ('CANDIDATE.json', 'PHYSICAL.json', 'READY.json',
                             'BINDING.json', 'SUBMIT-INTENT.json', 'SUBMIT-RECEIPT.json'))
CALIBRATION_SCOPE = 'GIGPO_POLICY_SAFETY_OFFICIAL_INSTRUCT_FRESH200_CALIBRATION_V1'
SCOPE = 'GIGPO_POLICY_SAFETY_OFFICIAL_INSTRUCT_FRESH200_LIVE_V1'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
NAMES = ('deployment_contract_v8.py', 'prepare_candidate_v8.py', 'verify_physical_v8.py',
         'freeze_ready_v8.py', 'submit_once_v8.py', 'test_deployment_controls_v8.py')
ROLES = ('source_manifest', 'source_review', 'source_cpu', 'source_cpu_review',
         'runtime_manifest', 'runtime_review', 'runtime_cpu', 'runtime_cpu_review',
         'controls_manifest', 'controls_review', 'controls_cpu', 'asset_manifest', 'asset_metadata', 'asset_origin_candidate')
ASSET_PINS = {
    'asset_manifest': (10425149, 'eab80e1bb33d6cf9f2f92d67aa110549fb29929251621207fc31a95b3af9fb84'),
    'asset_origin_candidate': (10913143, 'dbcecfa0ddf62ef6fb717f71efd1708892e1fda2e94c36af0257f55c1a10fa58'),
    'asset_metadata': (14701486, '02c96a172561257afc96dcb2875fb69c51c8e63a9fd9017830b710a766b9f6b5')}
ASSET_GROUPS = dict(environment=39161, headers=164, official_model=13,
                    common_data=19, retrieval_overlay=41, retrieval_assets=10)
INTERPRETER = dict(path=str(COMMON / 'env/bin/python'), bytes=5937672,
                   sha256='94df6ff4b32aafdba90e38c0aed70880fe44ee4cf88ecdf411641158f120c1af')
PROTECTED = {
    'calibration_v1.sbatch': (865, '966a0b2e848fc5a3c353227a001f18521203839d86e893a1be7b23f7181b1d6e'),
    'owned_runtime_resume_v1.py': (5265, 'c5271d6f170da1c2e9332f28458322feb67e47e4032b56ae4815e33ac8053030')}
GUARDS = {'resource_guard_v2.py': '9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444',
          'resource_ownership_v2.py': '7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed'}
HTTP_SHA = '344e65fa2f8662676b95a6b0f4534a90746298e4e2b2d5802208253c910bb343'
REFERENCE_SHA = 'b1de45354aa5f5014c8bb5341da7471a39d345c905979832b319c332bb5fc038'
UUID_SHA = '0a5a2f204b4f7b22758c7c297837581e0992b06f2b8e4c72d7890ad23b78a0ed'
UUIDS = ['GPU-53210089-2b52-bf37-b574-f91e8892a012', 'GPU-9b68e8d5-643e-25d0-99fa-6cdcd810082c',
         'GPU-85473c01-4753-bf6f-080f-9c89017440dd', 'GPU-9f380cfa-4239-a891-2e73-4bc48f32f06a',
         'GPU-56b1a759-2d63-8c03-fd6f-97883cd6a369', 'GPU-898426c9-149c-ecf1-1cd6-450c8d36a548',
         'GPU-fc12be2b-bff3-6df7-9eca-656755fdf7c1', 'GPU-cef6b05a-4e86-cab0-b75e-c417c30cf67e']


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def core(record):
    return {key: record[key] for key in ('path', 'bytes', 'sha256')}


def record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def closure(records, label):
    require(isinstance(records, list) and records, 'Missing ' + label)
    result = {}
    for item in records:
        require(isinstance(item, dict) and {'path', 'bytes', 'sha256'} <= set(item), 'Incomplete ' + label)
        path = Path(item['path'])
        require(path.is_absolute() and '..' not in path.parts and str(path) == item['path'], 'Invalid path: ' + label)
        require(item['path'] not in result and type(item['bytes']) is int and item['bytes'] >= 0
                and re.fullmatch('[0-9a-f]{64}', str(item['sha256'])), 'Invalid/duplicate ' + label)
        result[item['path']] = core(item)
    return result


def records_digest(records):
    items = sorted(closure(records, 'digest').values(), key=lambda item: item['path'])
    return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_installed_controls(candidate):
    wanted = closure(candidate['control_files'], 'installed controls')
    require(set(wanted) == {str(CONTROLS / name) for name in NAMES}, 'Incomplete installed controls')
    for path, item in wanted.items():
        require(Path(path).resolve() == Path(path) and record(path) == item, 'Different installed control bytes')


def validate_human_kl(candidate):
    require(candidate['token_kl_threshold_source'] == 'human'
            and type(candidate['token_kl_human_goal_updated_at']) is int
            and candidate['token_kl_human_goal_updated_at'] == 1791282908
            and type(candidate['candidate_old_policy_kl']) is float and candidate['candidate_old_policy_kl'] == .2
            and type(candidate['target_old_policy_kl']) is float and candidate['target_old_policy_kl'] == .2
            and candidate['negative_tail_limit'] is None,
            'Different explicit human sampled-policy KL0.2/tail-pending selection')


def live_launch_fields():
    return dict(live_goal_updated_at=1791299045, live_authorization_goal_updated_at=1791293201,
        initialization_human_goal_updated_at=1791275767, official_training_outer_target=200,
        initial_calibration_counted_outer_limit=5, first_full_acceptance_step=5,
        first5_count_toward_goal=True, restart_after_calibration=False,
        first5_known_token_kl_enforced=True, first5_tail_observation_only=True,
        production_tail_required_before_outer6=True, final_validation_rows=51713,
        retained_training_updates_contributed=0)


def validate_live_launch_plan(candidate):
    require(candidate.get('validation_scope') == SCOPE and candidate.get('calibration_scope') == CALIBRATION_SCOPE
            and candidate.get('allowed_safety_modes') == ['calibration', 'production']
            and 'calibration_actual_outer_budget' not in candidate,
            'Old isolated calibration authority cannot launch the live first5-to200 job')
    for key, expected in live_launch_fields().items():
        require(key in candidate and type(candidate[key]) is type(expected) and candidate[key] == expected,
                'Different exact live first5-to200 launch plan: ' + key)


def validate_live_deployment_review(review, candidate):
    require(review.get('live_goal_updated_at') == candidate['live_goal_updated_at']
            and review.get('live_authorization_goal_updated_at') == candidate['live_authorization_goal_updated_at']
            and type(review.get('live_goal_updated_at')) is int
            and type(review.get('live_authorization_goal_updated_at')) is int
            and all(review.get(key) is True for key in ('one_owned_live_job_launch_only',
                'before_first_Adam_probability_gate_required', 'first5_live_acceptance_before_outer6_required',
                'same_process_continuation_required'))
            and all(review.get(key) is False for key in ('fresh_restart_after_first5_allowed',
                'extra_formal200_submission_allowed', 'actual_gpu_acceptance', 'actual_probability_acceptance')),
            'Different/inflated exact owned live launch review')


def validate_canonical_uuid(candidate, files):
    expected = dict(path=str(COMMON / 'tools/expected-gpu-uuids.json'), bytes=352, sha256=UUID_SHA)
    require(core(candidate['canonical_uuid_inventory_record']) == expected and files.get(expected['path']) == expected
            and candidate['expected_gpu_uuid_inventory'] == UUIDS
            and candidate['canonical_uuid_inventory_is_hardware_only'] is True
            and candidate['actual_slurm_selected_uuid_comparison_before_guard_required'] is True,
            'Canonical plain-eight hardware UUID input is not exact/live-allocation comparison required')


def read_bound_bytes(item):
    """One no-follow open: validate and consume the same immutable buffer."""
    closure([item], 'bound bytes')
    path = Path(item['path'])
    require(path.resolve(strict=True) == path, 'Noncanonical proof: ' + str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size == item['bytes'], 'Bound input type/length differs')
        chunks = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, item['bytes'] - total + 1))
            if not block:
                break
            total += len(block)
            require(total <= item['bytes'], 'Bound input grew during read')
            chunks.append(block)
        raw = b''.join(chunks)
        after = os.fstat(descriptor); named = os.lstat(path)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                                 info.st_mtime_ns, info.st_ctime_ns)
        require(identity(before) == identity(after) == identity(named)
                and path.resolve(strict=True) == path, 'Bound input identity changed during read')
        require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'],
                'Changed bound bytes: ' + str(path))
        return raw
    finally:
        os.close(descriptor)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read_bound(item):
    raw = read_bound_bytes(item)
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def write_once(path, value):
    path = Path(path)
    require(path.parent.resolve() == path.parent and path.parent.is_dir(), 'Output parent not provisioned/canonical')
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def mapped_manifest(manifest, directory):
    result = []
    for item in manifest['files']:
        name = item['relative_path'] if 'relative_path' in item else Path(item['path']).name
        require(isinstance(name, str) and Path(name).name == name and name not in ('', '.', '..'), 'Nested/invalid helper name')
        result.append(dict(path=str(Path(directory) / name), bytes=item['bytes'], sha256=item['sha256']))
    return closure(result, 'mapped manifest')


def relative_runtime_closure(rows, directory):
    """Explicit relative-only review layout; never invent a missing path field."""
    relative_source_digest(rows)
    require(all(Path(row['relative_path']).name == row['relative_path'] for row in rows),
            'Nested runtime review record')
    return closure([dict(path=str(Path(directory) / row['relative_path']), bytes=row['bytes'], sha256=row['sha256'])
                    for row in rows], 'original relative runtime closure')


def validate_successor_independence(review):
    independence = review['independence']
    require(review.get('independent_review') is True
            and all(independence.get(key) is False for key in ('root_conversation_inherited',
                'root_conversation_used_as_acceptance', 'implementation_authorship',
                'prior_source_or_runtime_PASS_used_as_substitute', 'coordination_messages_used_as_acceptance',
                'implementation_and_tests_imported', 'implementation_and_tests_executed', 'implementation_edited',
                'remote_access_performed', 'network_used', 'GPU_Ray_Slurm_or_READY_execution'))
            and all(independence.get(key) is True for key in ('review_started_from_explicit_standalone_handoff',
                'independent_post_freeze_review', 'only_original_review_outputs_written'))
            and independence.get('original_fork_turns') == 'none',
            'Successor original review lacks exact independent static-only provenance')
    require(review.get('blocking_findings') == []
            and all(review.get(key) is False for key in ('actual_gpu_acceptance', 'production_READY', 'submitted'))
            and type(review.get('formal_iterations')) is int and review['formal_iterations'] == 0,
            'Successor original review inflates findings/execution scope')


def validate_static_cpu_decision(review, whole_key):
    decision = review['decision']
    choices = [value for key, value in decision.items() if key in (
        'CPU_test_execution_authorized_by_this_report', 'CPU_test_execution_authorized_by_this_PASS_report',
        'CPU_test_execution_authorized_by_this_static_review')]
    require(type(decision.get('blocking_findings')) is int and decision['blocking_findings'] == 0
            and decision.get(whole_key) is True and len(choices) == 1 and choices[0] is True
            and all(decision.get(key) is False for key in ('execution_authorized',
                'runtime_calibration_execution_authorized', 'ready', 'GPU_or_Ray_cluster_execution_authorized',
                'probability_acceptance', 'actual_two_outer_acceptance', 'native_execution_authorized',
                'formal200_authorized', 'single_submit_authorized', 'deployment_or_submission_authorized'))
            and type(decision.get('formal_iterations_contributed')) is int and decision['formal_iterations_contributed'] == 0,
            'Original static CPU-only decision inflates execution scope')


def validate_reviewed_source_snapshot(original, candidate, proofs):
    rows = original.get('full_tracked_source_files')
    manifest = proofs['source_manifest']
    require((original.get('bytes'), original.get('sha256')) ==
            (candidate['proofs']['source_manifest']['bytes'], candidate['proofs']['source_manifest']['sha256'])
            and isinstance(original.get('path'), str) and Path(original['path']).is_absolute()
            and original.get('actual_hash_match') is True and original.get('source_clean_verified') is True
            and original.get('full_tracked_set_matches') is True and original.get('actual_commit') == candidate['source_commit']
            and original.get('source_directory') == manifest['source_repo']
            and original.get('mismatches') == [] and type(original.get('source_file_count')) is int
            and type(original.get('actual_source_hash_matches')) is int
            and original['source_file_count'] == original['actual_source_hash_matches'] == candidate['source_file_count']
            and isinstance(rows, list) and len(rows) == candidate['source_file_count']
            and relative_source_digest(rows) == original.get('actual_full_source_digest') == manifest['source_tracked_files_digest']
            and relative_source_digest(rows) == relative_source_digest(manifest['full_tracked_source_files']),
            'Successor original full source snapshot/closure differs')
    covered_source_records(rows, candidate, 'successor original full source snapshot')


def validate_reviewed_runtime_snapshot(runtime, candidate, proofs):
    frozen = runtime['frozen_inputs']; original = frozen['runtime']; manifest = frozen['manifest']
    rows = original.get('records'); declared = proofs['runtime_manifest']
    expected = closure(candidate['runtime_files'], 'successor candidate runtime closure')
    require((manifest.get('bytes'), manifest.get('sha256')) ==
            (candidate['proofs']['runtime_manifest']['bytes'], candidate['proofs']['runtime_manifest']['sha256'])
            and isinstance(manifest.get('path'), str) and Path(manifest['path']).is_absolute()
            and manifest.get('actual_hash_match') is True and manifest.get('mismatches') == []
            and type(manifest.get('runtime_file_count')) is int and type(manifest.get('actual_runtime_hash_matches')) is int
            and manifest['runtime_file_count'] == manifest['actual_runtime_hash_matches'] == len(expected)
            and type(original.get('actual_file_count')) is int and original['actual_file_count'] == len(expected)
            and original.get('actual_directory_file_names_match') is True
            and original.get('directory') == declared['runtime_root' if candidate['runtime_version'] in (10, 11, 12) else 'runtime_directory']
            and relative_runtime_closure(rows, candidate['helpers']) == expected,
            'Successor original complete runtime manifest/relative records differ')
    local = relative_runtime_closure(rows, original['directory'])
    actual_digest = relative_source_digest(rows) if candidate['runtime_version'] in (9, 10, 11, 12) else records_digest(list(local.values()))
    require(actual_digest == original.get('actual_full_runtime_digest') == declared['runtime_files_digest'],
            'Original version-specific runtime digest differs from frozen manifest')


def validate_successor_cpu_policy(review, kind, candidate, proofs):
    policy = review['approved_cpu_policy']; manifest = proofs[kind + '_manifest']
    expected = closure(candidate['source_files'], 'successor CPU source inputs')
    helpers = closure(candidate['runtime_files'], 'successor CPU runtime inputs') if kind == 'runtime' else {}
    require(policy.get('role') == kind and policy.get('execution_scope') ==
            {'source': 'ALL_CURRENT_CPU_REGRESSIONS', 'runtime': 'FRESH200_RUNTIME_CPU_REGRESSIONS'}[kind]
            and policy.get('snapshot_sha256') == candidate['proofs'][kind + '_manifest']['sha256']
            and policy.get('source_snapshot_sha256') == candidate['proofs']['source_manifest']['sha256']
            and policy.get('source_repo') == policy.get('source_directory') == candidate['repo']
            and policy.get('code_commit') == candidate['source_commit']
            and policy.get('cwd') == (candidate['repo'] if kind == 'source' else candidate['helpers'])
            and policy.get('single_execution_only') is True
            and policy.get('GPU_allowed') is False and policy.get('READY_allowed') is False
            and policy.get('Slurm_submission_allowed') is False
            and type(policy.get('formal_iterations_contributed')) is int and policy['formal_iterations_contributed'] == 0
            and closure(policy.get('source_input_files'), 'policy source inputs') == expected
            and (closure(policy['runtime_input_files'], 'policy runtime inputs') == helpers if helpers
                 else policy.get('runtime_input_files') == [])
            and closure(policy.get('input_file_records'), 'policy full source/runtime inputs') == {**expected, **helpers}
            and policy.get('input_file_records') == policy['source_input_files'] + policy['runtime_input_files']
            and policy.get('source_input_files_digest') == records_digest(policy['source_input_files'])
            and policy.get('original_review_path') == candidate['proofs'][kind + '_review']['path']
            and (policy['source_snapshot_record']['bytes'], policy['source_snapshot_record']['sha256']) ==
                (candidate['proofs']['source_manifest']['bytes'], candidate['proofs']['source_manifest']['sha256'])
            and policy.get('test_files') == manifest['cpu_required_test_files'],
            'Successor static review CPU policy has different complete inputs/scope/selection')
    approved_cpu_commands(policy)
    if candidate['source_version'] in (10, 11):
        expected_helper = SOURCE_RUNTIME_CPU_HELPER if candidate['source_version'] == 11 else SOURCE_RUNTIME_CPU_HELPER_V3
        require(policy.get('management_helper') == expected_helper,
                'Successor CPU policy is not the exact version-bound full-directory management helper')
        source_rows = [dict(path=str(Path(candidate['repo']) / row['relative_path']), bytes=row['bytes'], sha256=row['sha256'])
                       for row in proofs['source_manifest']['full_tracked_source_files']]
        runtime_rows = [dict(path=str(Path(candidate['helpers']) / row['relative_path']), bytes=row['bytes'], sha256=row['sha256'])
                        for row in proofs['runtime_manifest']['files']] if kind == 'runtime' else []
        require(policy['source_input_files'] == source_rows and policy['runtime_input_files'] == runtime_rows,
                'Successor original policy differs from exact manifest inventory order')


def covered_source_records(items, candidate, label):
    wanted = closure(candidate['source_files'], 'source')
    require(isinstance(items, list) and items, 'Missing ' + label)
    for item in items:
        name = item['relative_path']
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid covered source path')
        path = str(Path(candidate['repo']) / name)
        require(wanted.get(path) == dict(path=path, bytes=item['bytes'], sha256=item['sha256']), 'Covered source changed: ' + label)
    require(len({item['relative_path'] for item in items}) == len(items), 'Duplicate covered source: ' + label)


def relative_source_digest(rows):
    require(isinstance(rows, list) and rows, 'Missing full tracked source inventory')
    normalized = []
    for row in rows:
        name = row['relative_path']; path = Path(name)
        require(isinstance(name, str) and name not in ('', '.', '..') and not path.is_absolute()
                and '..' not in path.parts and path.as_posix() == name, 'Invalid relative tracked source path')
        require(type(row['bytes']) is int and row['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}', str(row['sha256'])),
                'Invalid full tracked source byte record')
        normalized.append({key: row[key] for key in ('relative_path', 'bytes', 'sha256')})
    require(len({row['relative_path'] for row in normalized}) == len(normalized), 'Duplicate full tracked source')
    normalized.sort(key=lambda row: row['relative_path'])
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_final_source(candidate, manifest, review):
    version = candidate['source_version']
    manifest_schema = ('gigpo-e1-fresh200-source-v' + str(version) + '-static-snapshot'
                       if version in (9, 10, 11) else 'fresh200-source-static-v' + str(version))
    require(type(version) is int and version in (6, 7, 8, 9, 10, 11) and manifest['schema'] == manifest_schema
            and review['schema'] == candidate['source_review_schema'] == 'gigpo-e1-fresh200-v' + str(version) + '-preexec-review',
            'Original source manifest/report version differs; old v4 cannot be relabelled')
    require(manifest['new_code_commit'] == review['new_code_commit'] == candidate['source_commit']
            and (manifest.get('authoring_stopped') is True and manifest.get('live_scope') == SCOPE
                 and manifest.get('canonical_future_source_repo') == candidate['repo'] if version in (9, 10, 11)
                 else manifest.get('source_clean') is True), 'Final frozen source commit/clean identity differs')
    rows = manifest['full_tracked_source_files']; digest = relative_source_digest(rows)
    covered_source_records(rows, candidate, 'full frozen resulting source')
    require(len(rows) == manifest['full_tracked_source_file_count'] == candidate['source_file_count']
            and {str(Path(candidate['repo']) / row['relative_path']) for row in rows} ==
                set(closure(candidate['source_files'], 'full resulting source'))
            and digest == manifest['source_tracked_files_digest'] == review['source_tracked_files_digest'],
            'Full frozen resulting source closure/digest differs or has uncovered changes')


def validate_assets(candidate, proofs):
    for role, (size, digest) in ASSET_PINS.items():
        require(candidate['proofs'][role]['bytes'] == size and candidate['proofs'][role]['sha256'] == digest,
                'Unpinned original asset/provenance record: ' + role)
    manifest = proofs['asset_manifest']
    require(set(manifest['groups']) == set(ASSET_GROUPS) and manifest['group_counts'] == ASSET_GROUPS
            and manifest['file_count'] == 39408
            and all(len(manifest['groups'][key]) == count for key, count in ASSET_GROUPS.items()),
            'Fixed six-group 39408 accepted asset closure missing/subset')
    expected = closure([row for rows in manifest['groups'].values() for row in rows], 'fixed accepted assets')
    require(len(expected) == 39408 and expected == closure(candidate['asset_files'], 'actual complete assets'),
            'Candidate assets differ from complete immutable accepted closure')
    origins = manifest['accepted_asset_provenance_records']
    require(len(origins) == 2 and {(row['bytes'], row['sha256']) for row in origins} ==
            {ASSET_PINS[key] for key in ('asset_origin_candidate', 'asset_metadata')}
            and manifest['historical_native_acceptance_borrowed'] is False,
            'Accepted data-only original asset provenance differs')
    original_files = closure(proofs['asset_origin_candidate']['files'], 'pinned original asset candidate')
    original_physical = closure(proofs['asset_metadata']['metadata'], 'pinned original physical')
    require(all(original_files.get(path) == row == original_physical.get(path) for path, row in expected.items()),
            'Accepted asset SHA differs from exact original candidate/physical receipts')
    physical_rows = {row['path']: row for row in proofs['asset_metadata']['metadata']}
    require(all(all(type(physical_rows[path][key]) is int and physical_rows[path][key] >= 0
                    for key in ('device', 'inode', 'mtime_ns')) for path in expected), 'Missing original asset physical identity')
    require(core(manifest['environment_interpreter']) == INTERPRETER
            and closure(candidate['support_files'], 'support interpreter').get(INTERPRETER['path']) == INTERPRETER
            and manifest['environment_interpreter']['resolved_path'] == '/usr/bin/python3.10'
            and manifest['environment_entry_alias']['resolved_path'] == str(SHARED / 'env'),
            'Substituted accepted environment/interpreter')


def junit_cases(raw):
    require(b'<!DOCTYPE' not in raw and b'<!ENTITY' not in raw, 'Unexpected JUnit DTD/entity')
    root = ET.fromstring(raw)
    require(root.tag in ('testsuites', 'testsuite'), 'Unrecognized actual JUnit root')
    suites = [root] if root.tag == 'testsuite' else list(root)
    require(suites and all(suite.tag == 'testsuite' and suite.find('testsuite') is None for suite in suites),
            'Ambiguous nested actual JUnit suites')
    counts = dict(collected=0, passed=0, failed=0, errors=0, skipped=0)
    failure_ids = []; test_ids = []; hosts = set()
    for suite in suites:
        local = dict(collected=0, passed=0, failed=0, errors=0, skipped=0)
        require(isinstance(suite.get('hostname'), str) and suite.get('hostname'), 'JUnit actual host missing')
        hosts.add(suite.get('hostname'))
        for case in suite.findall('testcase'):
            name = case.get('name'); classname = case.get('classname')
            require(isinstance(name, str) and name and isinstance(classname, str) and classname, 'JUnit testcase identity missing')
            identity = classname + '::' + name; test_ids.append(identity)
            markers = [(tag, len(case.findall(tag))) for tag in ('failure', 'error', 'skipped')]
            require(sum(number for _, number in markers) <= 1, 'Ambiguous actual testcase outcome')
            outcome = next(({'failure': 'failed', 'error': 'errors', 'skipped': 'skipped'}[tag]
                            for tag, number in markers if number), 'passed')
            local['collected'] += 1; local[outcome] += 1
            if outcome in ('failed', 'errors'):
                failure_ids.append(identity)
        for xml_key, count_key in (('tests', 'collected'), ('failures', 'failed'), ('errors', 'errors'), ('skipped', 'skipped')):
            require(suite.get(xml_key) is not None and int(suite.get(xml_key)) == local[count_key],
                    'JUnit suite totals differ from actual testcases')
        for key in counts:
            counts[key] += local[key]
    require(counts['collected'] > 0 and len(test_ids) == len(set(test_ids)) and len(hosts) == 1,
            'Empty/duplicate/multi-host actual CPU test evidence')
    return dict(counts=counts, failure_ids=sorted(failure_ids), test_ids=sorted(test_ids), hostname=hosts.pop())


def original_cpu_flag(actual, names, expected):
    """Read explicit equivalent original field names without repairing raw history."""
    present = [actual[name] for name in names if name in actual]
    return bool(present) and all(value is expected for value in present)


def original_cpu_command(actual):
    commands = [actual[name] for name in ('command', 'argv', 'executed_command') if name in actual]
    require(commands and all(value == commands[0] for value in commands),
            'Original executed CPU command missing/different; wrapper cannot invent it')
    return commands[0]


def original_cpu_artifact(actual, name, item):
    """Support raw path+SHA and raw core-record forms; conflicting forms fail."""
    present = []
    if name in actual:
        present.append(actual[name] == item['path'])
    if name + '_record' in actual:
        present.append(actual[name + '_record'] == core(item))
    if name + '_sha256' in actual:
        present.append(actual[name + '_sha256'] == item['sha256'])
    return bool(present) and all(present) and (name in actual or name + '_record' in actual) \
        and (name + '_record' in actual or name + '_sha256' in actual)


def validate_actual_cpu_run(run, expected_commit, allowed_scopes, evidence, read=read_bound, read_bytes=read_bound_bytes):
    for key in ('record', 'log_record', 'junit_record'):
        require(evidence.get(run[key]['path']) == core(run[key]), 'Actual CPU original outside evidence: ' + key)
    actual = read(run['record']); cases = junit_cases(read_bytes(run['junit_record']))
    log = read_bytes(run['log_record'])
    require(log and actual['code_commit'] == expected_commit and actual['scope'] == run['execution_scope']
            and actual['scope'] in allowed_scopes and actual['node'] == cases['hostname']
            and original_cpu_flag(actual, ('gpu_acceptance', 'actual_gpu_acceptance'), False)
            and original_cpu_flag(actual, ('production_ready', 'production_READY'), False)
            and type(actual['formal_iterations']) is int and actual['formal_iterations'] == 0,
            'Actual original CPU scope/commit/host differs')
    command = original_cpu_command(actual)
    require(isinstance(command, list) and command and all(isinstance(part, str) and part for part in command)
            and command == run['executed_command']
            and (Path(command[0]).name == 'pytest' or any(command[index:index + 2] == ['-m', 'pytest']
                                                       for index in range(len(command) - 1))),
            'Original executed CPU command missing/different; wrapper cannot invent it')
    require(isinstance(actual['test_files'], list) and actual['test_files']
            and len(set(actual['test_files'])) == len(actual['test_files'])
            and actual['test_files'] == run['test_files'] and all(path in command for path in actual['test_files']),
            'Actual CPU command/test selection differs')
    require(original_cpu_artifact(actual, 'log', run['log_record'])
            and original_cpu_artifact(actual, 'junit', run['junit_record']),
            'Actual CPU RESULT raw log/JUnit identity differs')
    counts = cases['counts']
    require(all(type(run[key]) is int and run[key] == counts[key] for key in counts)
            and run['failure_ids'] == cases['failure_ids'], 'Declared CPU counts/failure IDs differ from actual JUnit')
    if 'actual_cases' in actual:
        require(type(actual['actual_cases']) is dict and actual['actual_cases'] == counts
                and all(type(value) is int for value in actual['actual_cases'].values()),
                'Original RESULT case totals differ from actual JUnit')
    require(all(key not in actual or type(actual[key]) is int and actual[key] == counts[key] for key in counts)
            and ('failure_ids' not in actual or actual['failure_ids'] == cases['failure_ids']),
            'Original RESULT top-level counts/failure IDs differ from actual JUnit')
    exit_code = actual['exit_code']
    require(type(exit_code) is int and exit_code == run['exit_code']
            and (exit_code == 0 if counts['failed'] + counts['errors'] == 0 else exit_code in (1, 2)),
            'Original CPU exit status contradicts raw testcase outcomes')
    cases['original_result'] = actual
    return cases


def approved_cpu_commands(policy):
    """One reviewed pytest mode, with exact files/options and no option injection."""
    require(isinstance(policy, dict) and policy.get('schema') == 'EXACT_FULL_PYTEST_INVOCATION_V1'
            and policy.get('config_policy') == 'EMPTY_DEV_NULL'
            and policy.get('plugin_policy') == 'BUILTIN_ONLY_AUTOLOAD_DISABLED_NO_CONFTEST',
            'Missing approved exact CPU invocation policy')
    for key in ('cwd', 'collection_basetemp', 'run_basetemp', 'junit_path'):
        value = policy.get(key)
        require(isinstance(value, str) and Path(value).is_absolute() and '..' not in Path(value).parts
                and str(Path(value)) == value and '::' not in value, 'Invalid approved CPU path: ' + key)
    interpreter = policy.get('interpreter')
    require(isinstance(interpreter, dict) and set(interpreter) == {'path', 'bytes', 'sha256', 'argv0', 'resolved_path'}
            and interpreter['resolved_path'] == interpreter['path']
            and isinstance(interpreter['argv0'], str) and Path(interpreter['argv0']).is_absolute()
            and '..' not in Path(interpreter['argv0']).parts and str(Path(interpreter['argv0'])) == interpreter['argv0'],
            'Unapproved CPU interpreter identity')
    closure([core(interpreter)], 'approved CPU interpreter')
    files = policy.get('test_files')
    require(isinstance(files, list) and files and len(set(files)) == len(files)
            and all(isinstance(name, str) and name and not Path(name).is_absolute() and not name.startswith('-') and '::' not in name
                    and '[' not in name and name.endswith('.py') and '..' not in Path(name).parts
                    and str(Path(name)) == name for name in files), 'Unapproved CPU file selection')
    inputs = closure(policy.get('input_file_records'), 'approved full CPU input closure')
    test_records = policy.get('test_file_records')
    require(isinstance(test_records, list) and test_records, 'Missing approved CPU test file records')
    if all(isinstance(row, dict) and 'relative_path' in row and 'path' not in row for row in test_records):
        relative_source_digest(test_records)
        selected = closure([dict(path=str(Path(policy['cwd']) / row['relative_path']), bytes=row['bytes'], sha256=row['sha256'])
                            for row in test_records], 'approved relative CPU fixture closure')
    else:
        selected = closure(test_records, 'approved CPU fixture closure')
    paths = {str(Path(name) if Path(name).is_absolute() else Path(policy['cwd']) / name) for name in files}
    require(set(selected) == paths and all(inputs.get(path) == row for path, row in selected.items())
            and policy.get('input_files_digest') == records_digest(policy['input_file_records']),
            'Approved CPU test selection/input bytes differ')
    environment = policy.get('execution_environment')
    require(isinstance(environment, dict) and environment and all(isinstance(key, str) and key
            and isinstance(value, str) for key, value in environment.items()), 'Missing exact CPU environment')
    require(environment.get('PYTEST_ADDOPTS') == '' and environment.get('PYTEST_PLUGINS', '') == ''
            and environment.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
            and environment.get('PYTHONNOUSERSITE') == '1'
            and environment.get('PYTHONDONTWRITEBYTECODE') == '1' and environment.get('CUDA_VISIBLE_DEVICES') == ''
            and not any(key.startswith('PYTEST_') and key not in
                        ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD') for key in environment)
            and not any(key.upper().endswith('_PROXY') or key in ('LD_PRELOAD', 'LD_LIBRARY_PATH',
                'DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH', 'PYTHONSTARTUP', 'PYTHONHOME', 'PYTHONUSERBASE')
                for key in environment), 'Unapproved inherited pytest/plugin/loader environment')
    prefix = [interpreter['argv0'], '-m', 'pytest']
    common = ['-q', '--disable-warnings', '-p', 'no:cacheprovider', '-c', '/dev/null', '--noconftest',
              '--rootdir', policy['cwd'], '--basetemp']
    collected = prefix + ['--collect-only'] + common + [policy['collection_basetemp']] + files
    executed = prefix + common + [policy['run_basetemp'], '--junitxml', policy['junit_path']] + files
    require(('collection_command' not in policy or policy['collection_command'] == collected)
            and ('executed_command' not in policy or policy['executed_command'] == executed)
            and ('rootdir' not in policy or policy['rootdir'] == policy['cwd']), 'Approved CPU argv/rootdir differs from fixed policy')
    return collected, executed


def canonical_collected_case(node_id, test_files):
    """Map standard pytest .py::Class::name[param] without altering parameters."""
    require(isinstance(node_id, str) and '\n' not in node_id and '\r' not in node_id,
            'Invalid original collection node ID')
    matches = [name for name in test_files if node_id.startswith(name + '::')]
    require(len(matches) == 1, 'Collection node outside exact approved test files')
    filename = matches[0]; suffix = node_id[len(filename) + 2:]
    before, bracket, parameters = suffix.partition('[')
    parts = before.split('::')
    require(parts and all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) for part in parts)
            and parts[-1].startswith('test') and (not bracket or parameters.endswith(']')),
            'Unrecognized standard pytest collection identity')
    # Pytest JUnit uses the file spelling with slash->dot and .py removed.
    classname = filename[:-3].replace('/', '.')
    if len(parts) > 1:
        classname += '.' + '.'.join(parts[:-1])
    return classname + '::' + parts[-1] + (bracket + parameters if bracket else '')


def raw_full_collection(raw, test_files):
    require(isinstance(raw, bytes) and raw, 'Missing actual full collection log')
    lines = raw.decode('utf-8').splitlines(); nodes = []; summaries = []
    for line in lines:
        if any(line.startswith(name + '::') for name in test_files):
            canonical_collected_case(line, test_files); nodes.append(line)
        elif re.match(r'\S+\.py::', line):
            raise ValueError('Collection log contains undeclared test path')
        match = re.fullmatch(r'(\d+) tests? collected in [0-9.]+s', line.strip())
        if match:
            summaries.append(int(match.group(1)))
    require(nodes and len(nodes) == len(set(nodes)) and summaries == [len(nodes)]
            and not any('deselected' in line.lower() or 'error collecting' in line.lower() for line in lines),
            'Actual full collection incomplete/duplicate/filtered')
    cases = [canonical_collected_case(node, test_files) for node in nodes]
    require(len(cases) == len(set(cases)), 'Duplicate canonical collection identity')
    return nodes, sorted(cases)


def validate_cpu_environment(actual, policy):
    require(actual.get('execution_environment') == policy['execution_environment'],
            'Actual CPU exact environment differs from approval')
    sanitization = actual.get('environment_sanitization')
    environment = policy['execution_environment']
    require(isinstance(sanitization, dict) and sanitization.get('strategy') == 'EXACT_ALLOWLIST_NO_INHERITANCE'
            and sanitization.get('inherited_environment_forwarded') is False
            and sanitization.get('cleared_inputs') == dict(PYTEST_ADDOPTS='empty',
                PYTEST_PLUGINS='empty' if 'PYTEST_PLUGINS' in environment else 'absent')
            and isinstance(sanitization.get('removed_input_names'), list)
            and all(isinstance(key, str) and key for key in sanitization['removed_input_names'])
            and sanitization['removed_input_names'] == sorted(set(sanitization['removed_input_names'])),
            'Actual CPU inherited pytest/plugin inputs not cleared and recorded')


def cpu_collection_record(run):
    records = [run[key] for key in ('collection_result_record', 'collection_record') if key in run]
    require(records and all(item == records[0] for item in records), 'Missing/conflicting original collection record')
    return records[0]


def validate_full_cpu_run(run, expected_commit, allowed_scopes, evidence, policy,
                          read=read_bound, read_bytes=read_bound_bytes):
    """Production current-run acceptance from separately approved original collection."""
    collection_command, execution_command = approved_cpu_commands(policy)
    collection_record = cpu_collection_record(run)
    for item in (collection_record, run['collection_log_record']):
        require(evidence.get(item['path']) == core(item), 'Actual CPU collection outside evidence')
    collection = read(collection_record)
    snapshot_field = {'FRESH200_DEPLOYMENT_CONTROLS_CPU_ONLY': 'controls_manifest_sha256',
                      'FRESH200_RUNTIME_CPU_REGRESSIONS': 'runtime_snapshot_sha256'}.get(run['execution_scope'], 'source_snapshot_sha256')
    require(re.fullmatch('[0-9a-f]{64}', str(policy.get('snapshot_sha256')))
            and original_cpu_command(collection) == collection_command and collection.get('cwd') == policy['cwd']
            and collection.get('interpreter') == policy['interpreter']
            and collection.get('test_files') == policy['test_files']
            and collection.get('test_file_records') == policy['test_file_records']
            and collection.get(snapshot_field) == policy['snapshot_sha256']
            and ('input_file_records' not in collection or collection['input_file_records'] == policy['input_file_records'])
            and ('input_files_digest' not in collection or collection['input_files_digest'] == policy['input_files_digest'])
            and collection.get('config_policy') == policy['config_policy']
            and collection.get('rootdir') == policy['cwd']
            and ('plugin_policy' not in collection or collection['plugin_policy'] == policy['plugin_policy'])
            and ('dependency_receipt' not in policy or collection.get('dependency_receipt') == policy['dependency_receipt'])
            and collection.get('code_commit') == expected_commit and collection.get('scope') == run['execution_scope']
            and collection.get('scope') in allowed_scopes and type(collection.get('exit_code')) is int
            and collection['exit_code'] == 0
            and ('actual_execution' not in collection or collection['actual_execution'] is True)
            and original_cpu_flag(collection, ('gpu_acceptance', 'actual_gpu_acceptance'), False)
            and original_cpu_flag(collection, ('production_ready', 'production_READY'), False)
            and type(collection.get('formal_iterations')) is int and collection['formal_iterations'] == 0
            and original_cpu_artifact(collection, 'log', run['collection_log_record'])
            and collection.get('collection_log') == run['collection_log_record']['path'],
            'Actual original full collection command/input/scope differs')
    validate_cpu_environment(collection, policy)
    nodes, identities = raw_full_collection(read_bytes(run['collection_log_record']), policy['test_files'])
    declared_cases = collection.get('canonical_case_ids')
    require(collection.get('node_ids') == nodes and isinstance(declared_cases, list)
            and len(declared_cases) == len(set(declared_cases)) and sorted(declared_cases) == identities
            and ('collected' not in collection or type(collection['collected']) is int and collection['collected'] == len(nodes)),
            'Original collection cases differ from full raw log')
    current = validate_actual_cpu_run(run, expected_commit, allowed_scopes, evidence, read, read_bytes)
    actual = current['original_result']
    collection_records = [actual[key] for key in ('collection_result_record', 'collection_record') if key in actual]
    executed_ids = [actual[key] for key in ('junit_case_ids', 'executed_canonical_case_ids') if key in actual]
    require(original_cpu_command(actual) == execution_command and actual.get('cwd') == policy['cwd']
            and actual.get('interpreter') == policy['interpreter'] and actual.get('test_files') == policy['test_files']
            and actual.get('test_file_records') == policy['test_file_records']
            and actual.get(snapshot_field) == policy['snapshot_sha256']
            and ('input_file_records' not in actual or actual['input_file_records'] == policy['input_file_records'])
            and ('input_files_digest' not in actual or actual['input_files_digest'] == policy['input_files_digest'])
            and actual.get('config_policy') == policy['config_policy'] and actual.get('rootdir') == policy['cwd']
            and ('plugin_policy' not in actual or actual['plugin_policy'] == policy['plugin_policy'])
            and ('dependency_receipt' not in policy or actual.get('dependency_receipt') == policy['dependency_receipt'])
            and collection_records and all(item == core(collection_record) for item in collection_records)
            and actual.get('collection_log_record') == core(run['collection_log_record'])
            and executed_ids and all(isinstance(value, list) and len(value) == len(set(value))
                and sorted(value) == current['test_ids'] for value in executed_ids)
            and ('collected_canonical_case_ids' not in actual or sorted(actual['collected_canonical_case_ids']) == identities)
            and actual.get('node') == collection.get('node'),
            'Actual CPU exact approved argv/interpreter/cwd/inputs differ')
    validate_cpu_environment(actual, policy)
    counts = current['counts']
    require(current['test_ids'] == identities and counts['collected'] == counts['passed'] == len(identities)
            and all(counts[key] == 0 for key in ('failed', 'errors', 'skipped')) and current['failure_ids'] == [],
            'Actual CPU does not execute exact full collection with zero skips/fails/errors')
    current['collection_result'] = collection
    return current


def successor_cpu_identity(candidate, kind):
    source = closure(candidate['source_files'], 'successor CPU source identity')
    rows = [dict(relative_path=Path(path).relative_to(candidate['repo']).as_posix(), bytes=row['bytes'], sha256=row['sha256'])
            for path, row in source.items()]
    identity = dict(commit=candidate['source_commit'], clean=True, source_files=len(source),
        actual_source_input_digest=records_digest(list(source.values())), source_digest=relative_source_digest(rows),
        runtime_files=len(candidate['runtime_files']) if kind == 'runtime' else 0,
        runtime_manifest_sha256=candidate['proofs']['runtime_manifest']['sha256'] if kind == 'runtime' else None,
        source_snapshot_sha256=candidate['proofs']['source_manifest']['sha256'])
    if candidate['source_version'] in (10, 11):
        identity['source_directory_inventory'] = dict(directory=candidate['repo'], actual_file_count=len(source),
            actual_directory_file_names=sorted(row['relative_path'] for row in rows), exact_declared_inventory=True)
        identity['runtime_directory_inventory'] = (dict(directory=candidate['helpers'],
            actual_file_count=len(candidate['runtime_files']), actual_directory_file_names=sorted(
                Path(row['path']).relative_to(candidate['helpers']).as_posix() for row in candidate['runtime_files']),
            exact_declared_inventory=True) if kind == 'runtime' else None)
    return identity


def validate_successor_cpu_originals(started, current, policy, candidate, kind, read):
    static = read(candidate['proofs'][kind + '_review']); approved = static['approved_cpu_policy']
    keys = ('schema', 'cwd', 'rootdir', 'interpreter', 'test_files', 'test_file_records', 'input_file_records',
            'input_files_digest', 'snapshot_sha256', 'source_snapshot_sha256', 'execution_environment',
            'config_policy', 'plugin_policy', 'collection_basetemp', 'run_basetemp', 'junit_path',
            'collection_command', 'executed_command', 'management_helper')
    require(all(key in policy and policy[key] == approved[key] for key in keys),
            'Successor current CPU approval differs from original static exact invocation')
    identity = successor_cpu_identity(candidate, kind)
    require(started.get('source_before') == current['original_result'].get('source_before') ==
            current['original_result'].get('source_after') == identity,
            'Successor CPU source/runtime before/after identity differs')
    for original in (started, current['collection_result'], current['original_result']):
        require(original.get('actual_execution') is True
                and original.get('management_helper') == approved['management_helper']
                and original.get('input_file_records') == approved['input_file_records']
                and original.get('input_files_digest') == approved['input_files_digest']
                and original.get('source_snapshot_sha256') == candidate['proofs']['source_manifest']['sha256']
                and original.get('review_sha256') == candidate['proofs'][kind + '_review']['sha256'],
                'Successor original CPU helper/input/snapshot/review identity differs')
        validate_cpu_environment(original, approved)
    require(type(current['original_result'].get('actual_cases')) is dict,
            'Successor original CPU case totals are not a count dictionary')


def validate_cpu(receipt, review, kind, candidate, evidence, read=read_bound, read_bytes=read_bound_bytes):
    require(receipt['status'] == review['status'] == 'PASS' and receipt['gpu_used'] is False
            and receipt['actual_execution'] is True and type(receipt['exit_code']) is int
            and receipt['exit_code'] == 0 and type(receipt['failed']) is int and receipt['failed'] == 0
            and type(receipt['passed']) is int and receipt['passed'] > 0,
            'Actual successful CPU evidence missing: ' + kind)
    require(receipt['source_commit'] == review['source_commit'] == candidate['source_commit'], 'Different CPU source')
    manifest_sha = candidate['proofs'][kind + '_manifest']['sha256']
    require(receipt['manifest_sha256'] == review['manifest_sha256'] == manifest_sha
            and review['cpu_receipt_sha256'] == candidate['proofs'][kind + '_cpu']['sha256']
            and review['independent_review'] is True and review['complete_required_cpu_coverage_accepted'] is True,
            'Different/incomplete independent CPU acceptance: ' + kind)
    originals = closure(receipt['raw_execution_records'], 'CPU originals')
    require(all(evidence.get(path) == item for path, item in originals.items()), 'CPU originals outside evidence closure')
    scopes = {'source': ('ALL_CURRENT_CPU_REGRESSIONS', 'RELEVANT_CHANGED_CPU_REGRESSIONS',
                         'REVIEWED_CPU_ONLY_NO_CUDA_RAY_OR_REAL_DISTRIBUTED'),
              'runtime': ('FRESH200_RUNTIME_CPU_REGRESSIONS',)}[kind]
    policy = review['approved_current_invocation']
    if candidate.get('source_version') in (10, 11):
        expected_helper = SOURCE_RUNTIME_CPU_HELPER if candidate['source_version'] == 11 else SOURCE_RUNTIME_CPU_HELPER_V3
        require(policy.get('management_helper') == expected_helper
                and evidence.get(expected_helper['path']) == expected_helper,
                'Successor actual CPU management helper is outside the exact frozen evidence closure')
    require(policy.get('snapshot_sha256') == manifest_sha, 'CPU approval is not the reviewed final snapshot')
    current = validate_full_cpu_run(receipt['current_run'], candidate['source_commit'], scopes, evidence, policy, read, read_bytes)
    base = Path(candidate['repo'] if kind == 'source' else candidate['helpers'])
    accepted = closure(candidate['source_files' if kind == 'source' else 'runtime_files'], 'current CPU reviewed inputs')
    relocated = {str(Path(policy['cwd']) / Path(path).relative_to(base)):
                 dict(path=str(Path(policy['cwd']) / Path(path).relative_to(base)), bytes=row['bytes'], sha256=row['sha256'])
                 for path, row in accepted.items()}
    if kind == 'runtime':
        require(policy['cwd'] == candidate['helpers'], 'Actual runtime CPU cwd differs from exact reviewed helper namespace')
        source_inputs = closure(candidate['source_files'], 'runtime CPU full source dependency closure')
        require(not set(source_inputs).intersection(relocated), 'Runtime CPU source/helper input groups overlap')
        relocated.update(source_inputs)
    require(closure(policy['input_file_records'], 'actual current CPU full input closure') == relocated,
            'Actual current CPU inputs differ from reviewed full source/runtime closure')
    require(all(type(receipt[key]) is int and receipt[key] == current['counts'][key] for key in current['counts'])
            and current['failure_ids'] == [] and current['original_result']['scope'] == receipt['scope'],
            'Current CPU summary/scope differs from actual raw run')
    require(all(originals.get(receipt['current_run'][key]['path']) == core(receipt['current_run'][key])
                for key in ('record', 'log_record', 'junit_record', 'collection_log_record'))
            and originals.get(cpu_collection_record(receipt['current_run'])['path']) == core(cpu_collection_record(receipt['current_run']))
            and current['original_result']['review_sha256'] == candidate['proofs'][kind + '_review']['sha256']
            and current['original_result'][kind + '_snapshot_sha256'] == manifest_sha,
            'Actual current CPU run is not the original reviewed final source/runtime snapshot')
    if kind == 'source':
        require(receipt['scope'] in ('ALL_CURRENT_CPU_REGRESSIONS', 'RELEVANT_CHANGED_CPU_REGRESSIONS'), 'Wrong source CPU scope')
        covered_source_records(receipt['current_source_and_test_files'], candidate, 'current CPU source/test')
        started_record = receipt['current_run']['started_record']
        require(evidence.get(started_record['path']) == originals.get(started_record['path']) == core(started_record),
                'Original CPU STARTED record unbound')
        started = read(started_record)
        rows = [dict(relative_path=Path(path).relative_to(base).as_posix(), bytes=row['bytes'], sha256=row['sha256'])
                for path, row in accepted.items()]
        successor = candidate.get('source_version') in (9, 10, 11)
        identity = successor_cpu_identity(candidate, kind) if successor else dict(
            commit=candidate['source_commit'], clean=True, files=len(rows), digest=relative_source_digest(rows))
        require(started.get('source_before') == identity and current['original_result'].get('source_after') == identity
                and started.get('source_snapshot_sha256') == manifest_sha
                and started.get('review_sha256') == candidate['proofs']['source_review']['sha256']
                and started.get('command') == original_cpu_command(current['original_result'])
                and started.get('collection_command') == original_cpu_command(current['collection_result'])
                and started.get('cwd') == policy['cwd'] and started.get('interpreter') == policy['interpreter']
                and started.get('test_files') == policy['test_files'] and started.get('test_file_records') == policy['test_file_records'],
                'Actual current CPU source before/after/full closure identity differs')
        validate_cpu_environment(started, policy)
        if successor:
            require(receipt['scope'] == 'ALL_CURRENT_CPU_REGRESSIONS', 'Successor source CPU must execute the complete declared suite')
            validate_successor_cpu_originals(started, current, policy, candidate, kind, read)
        require(('dependency_receipt' not in policy or started.get('dependency_receipt') == policy['dependency_receipt'])
                and started.get('config_policy') == policy['config_policy'] and started.get('rootdir') == policy['cwd'],
                'Original CPU STARTED dependency/config inputs differ')
        covered_test_paths = {str(Path(candidate['repo']) / row['relative_path']) for row in receipt['current_source_and_test_files']}
        require(all(str(Path(name) if Path(name).is_absolute() else Path(candidate['repo']) / name) in covered_test_paths
                    for name in current['original_result']['test_files']), 'Current actual CPU test file SHA not covered')
        require(review['scope'] == 'FRESH200_SOURCE_CPU_EVIDENCE_ONLY', 'CPU review scope inflated')
        if receipt['scope'] == 'RELEVANT_CHANGED_CPU_REGRESSIONS':
            require(receipt['claims_all_prior_cases_rerun_on_current_commit'] is False, 'Prior cases cannot be called rerun')
            patch = core(receipt['minimal_patch_record'])
            require(evidence.get(patch['path']) == patch, 'Unbound exact minimal patch')
            covered_source_records(receipt['unchanged_source_files'], candidate, 'unchanged prior CPU source')
            require(receipt['prior_runs'] and review['superseded_prior_failure_ids'] == receipt['superseded_prior_failure_ids'],
                    'Unreviewed replacement of historical CPU failure')
            prior_unchanged = {}; superseded = []
            for prior in receipt['prior_runs']:
                item = core(prior['record'])
                manifest = core(prior['source_manifest_record'])
                require(evidence.get(item['path']) == item and re.fullmatch('[0-9a-f]{40}', prior['source_commit'])
                        and type(prior['passed']) is int and prior['passed'] > 0 and type(prior['failed']) is int
                        and prior['failed'] >= 0 and evidence.get(manifest['path']) == manifest
                        and len(prior['superseded_failure_ids']) == prior['failed'] + prior['errors'], 'Missing preserved actual prior CPU evidence')
                parsed_prior = validate_actual_cpu_run(prior, prior['source_commit'], scopes, evidence, read, read_bytes)
                require(prior['superseded_failure_ids'] == parsed_prior['failure_ids'],
                        'Every original CPU failure must be explicitly independently superseded')
                require(all(originals.get(prior[key]['path']) == core(prior[key])
                            for key in ('record', 'log_record', 'junit_record')), 'Preserved prior CPU originals omitted')
                actual_prior = read(prior['record']); prior_source = read(prior['source_manifest_record'])
                require(actual_prior['code_commit'] == prior_source['source_commit'] == prior['source_commit']
                        and actual_prior['gpu_acceptance'] is False and actual_prior['production_ready'] is False,
                        'Preserved actual CPU/source identity differs')
                old_files = closure(prior_source['files'], 'prior CPU source identity')
                old_repo = Path(prior_source['repo'])
                require(all(str(Path(name) if Path(name).is_absolute() else old_repo / name) in old_files
                            for name in actual_prior['test_files']), 'Prior actual CPU selected test missing from original source identity')
                for row in prior['unchanged_source_files']:
                    path = str(old_repo / row['relative_path'])
                    require(old_files.get(path) == dict(path=path, bytes=row['bytes'], sha256=row['sha256']),
                            'Prior unchanged source not present in bound historical source identity')
                covered_source_records(prior['unchanged_source_files'], candidate, 'prior immutable source')
                for row in prior['unchanged_source_files']:
                    name = row['relative_path']
                    require(name not in prior_unchanged or prior_unchanged[name] == row, 'Conflicting prior source SHA')
                    prior_unchanged[name] = row
                superseded.extend(prior['superseded_failure_ids'])
            require(set(prior_unchanged) == {r['relative_path'] for r in receipt['unchanged_source_files']}
                    and len(set(superseded)) == len(superseded)
                    and sorted(superseded) == sorted(receipt['superseded_prior_failure_ids']), 'Incomplete prior CPU coverage/replacement')
            require(review['unchanged_source_files_digest'] == records_digest([
                dict(path=str(Path(candidate['repo']) / r['relative_path']), bytes=r['bytes'], sha256=r['sha256'])
                for r in receipt['unchanged_source_files']]), 'Independent unchanged CPU source binding differs')
    else:
        require(receipt['scope'] == 'FRESH200_RUNTIME_CPU_REGRESSIONS' and review['scope'] == 'FRESH200_RUNTIME_CPU_EVIDENCE_ONLY',
                'Wrong runtime CPU scope')
        require(receipt['runtime_files_digest'] == records_digest(candidate['runtime_files']), 'Different tested runtime bytes')
        started_record = receipt['current_run']['started_record']
        require(evidence.get(started_record['path']) == originals.get(started_record['path']) == core(started_record),
                'Original runtime CPU STARTED record unbound')
        started = read(started_record)
        source_rows = [dict(relative_path=Path(path).relative_to(candidate['repo']).as_posix(),
                            bytes=row['bytes'], sha256=row['sha256']) for path, row in source_inputs.items()]
        successor = candidate.get('runtime_version') in (10, 11, 12)
        identity = successor_cpu_identity(candidate, kind) if successor else dict(
            commit=candidate['source_commit'], clean=True, source_files=len(source_inputs),
            source_digest=relative_source_digest(source_rows), runtime_files=len(accepted), runtime_manifest_sha256=manifest_sha)
        require(started.get('source_before') == current['original_result'].get('source_after') == identity
                and all(original.get('source_snapshot_sha256') == candidate['proofs']['source_manifest']['sha256']
                        and original.get('runtime_snapshot_sha256') == manifest_sha
                        and original.get('review_sha256') == candidate['proofs']['runtime_review']['sha256']
                        and original.get('input_file_records') == policy['input_file_records']
                        and original.get('input_files_digest') == policy['input_files_digest']
                        for original in (started, current['collection_result'], current['original_result']))
                and started.get('command') == original_cpu_command(current['original_result'])
                and started.get('collection_command') == original_cpu_command(current['collection_result'])
                and started.get('cwd') == policy['cwd'] and started.get('interpreter') == policy['interpreter']
                and started.get('test_files') == policy['test_files'] and started.get('test_file_records') == policy['test_file_records']
                and started.get('config_policy') == policy['config_policy'] and started.get('rootdir') == policy['cwd']
                and ('dependency_receipt' not in policy or started.get('dependency_receipt') == policy['dependency_receipt']),
                'Actual runtime CPU before/after/source+helper full closure identity differs')
        validate_cpu_environment(started, policy)
        if successor:
            validate_successor_cpu_originals(started, current, policy, candidate, kind, read)
        runtime_files = closure(candidate['runtime_files'], 'actual CPU runtime test selection')
        require(all(str(Path(name) if Path(name).is_absolute() else Path(candidate['helpers']) / name) in runtime_files
                    for name in current['original_result']['test_files']), 'Actual runtime CPU test outside final helper closure')


def validate_static_reviews(candidate, proofs):
    """Consume the original strict reports; no replacement generic PASS is made."""
    c = candidate; source = proofs['source_review']; runtime = proofs['runtime_review']
    validate_final_source(c, proofs['source_manifest'], source)
    if c['source_version'] in (9, 10, 11):
        require(source['schema'] == c['source_review_schema'] == 'gigpo-e1-fresh200-v' + str(c['source_version']) + '-preexec-review'
                and source['status'] == 'PASS_STATIC_ONLY_CPU_SCOPE'
                and source['scope'] == 'FRESH200_SOURCE_V' + str(c['source_version']) + '_STATIC_CPU_ONLY',
                'Different original typed successor source9/10/11 static review')
        validate_successor_independence(source)
        validate_static_cpu_decision(source, 'whole_source_static_pass')
        validate_reviewed_source_snapshot(source['frozen_inputs']['source_snapshot'], c, proofs)
        validate_successor_cpu_policy(source, 'source', c, proofs)
    else:
        require(source['schema'] == c['source_review_schema']
                and source['status'] == 'PASS_STATIC_ONLY_CPU_SCOPE'
                and source['reviewer']['root_conversation_inherited'] is False
                and source['reviewer']['independent_after_v' + str(c['source_version']) + '_freeze'] is True
                and source['blocking_findings'] == [],
                'Original strict final source static review missing')
        freeze = source['freeze']; manifest = proofs['source_manifest']
        if isinstance(freeze, list):
            require(any(item['sha256'] == c['proofs']['source_manifest']['sha256']
                        and item['independently_hash_matched'] is True for item in freeze), 'Source review is a different snapshot')
        else:
            # Explicit original v8 dictionary layout; preserve the original report.
            require(c['source_version'] == 8 and isinstance(freeze, dict)
                    and freeze.get('snapshot_sha256') == c['proofs']['source_manifest']['sha256']
                    and isinstance(freeze.get('snapshot_path'), str) and Path(freeze['snapshot_path']).is_absolute()
                    and freeze.get('independently_hash_matched') is True and freeze.get('source_clean') is True
                    and freeze.get('actual_new_code_commit') == c['source_commit']
                    and freeze.get('full_tracked_records_exactly_matched') is True
                    and type(freeze.get('full_tracked_file_count')) is int
                    and freeze['full_tracked_file_count'] == manifest['full_tracked_source_file_count'] == c['source_file_count']
                    and freeze.get('actual_source_tracked_files_digest') == manifest['source_tracked_files_digest'],
                    'Original v8 dictionary freeze/full source closure differs')
            changed = source['file_receipts']; unchanged = source.get('unchanged_files_receipts')
            require(relative_source_digest(changed) == relative_source_digest(manifest['files'])
                    and relative_source_digest(unchanged) == relative_source_digest(manifest['unchanged_files'])
                    and not {row['relative_path'] for row in changed}.intersection(row['relative_path'] for row in unchanged)
                    and all(row.get('same_as_base') is True for row in unchanged),
                    'Original v8 changed/unchanged receipts differ from original snapshot')
            covered_source_records(unchanged, c, 'original v8 unchanged source receipts')
        covered_source_records(source['file_receipts'], c, 'strict source review original files')
        reviewed_receipts = {r['relative_path']: (r['bytes'], r['sha256']) for r in source['file_receipts']}
        require(reviewed_receipts in [
                {r['relative_path']: (r['bytes'], r['sha256']) for r in proofs['source_manifest'][key]}
                for key in ('files', 'full_tracked_source_files')],
                'Strict source review omits final snapshot file')
        require(source['cpu_execution_decision']['authorized_by_this_static_review'] is True
                and source['cpu_execution_decision']['actual_8GPU_acceptance'] is False
                and source['cpu_execution_decision']['production_READY'] is False, 'Source static scope inflated')
    schema = runtime['schema']; status = runtime['status']
    require(type(c['runtime_version']) is int and c['runtime_version'] in (5, 6, 7, 8, 9, 10, 11, 12)
            and type(schema) is type(c['runtime_review_schema']) and schema == c['runtime_review_schema']
            and type(status) is str, 'Failed/non-static original runtime review')
    version = str(c['runtime_version'])
    if type(schema) is int and schema == 1 and c['runtime_version'] in (9, 10, 11, 12):
        expected_scope = ('INDEPENDENT_FROZEN_FRESH200_CALIBRATION_V9_STATIC_PREEXEC_REVIEW_ONLY'
                          if c['runtime_version'] == 9 else
                          'INDEPENDENT_FROZEN_FRESH200_CONTINUATION_V' + version + '_STATIC_PREEXEC_REVIEW_ONLY')
        require(status == 'PASS' and type(runtime.get('runtime_version')) is int
                and runtime['runtime_version'] == c['runtime_version'] and runtime.get('scope') == expected_scope,
                'Different original typed runtime9/10/11/12 report version/status/scope')
        validate_successor_independence(runtime)
        validate_static_cpu_decision(runtime, 'whole_runtime_static_pass')
        validate_reviewed_runtime_snapshot(runtime, c, proofs)
        source_original = runtime['frozen_inputs']['source_snapshot']
        if c['runtime_version'] == 9:
            require(runtime['independence'].get('completed_CPU_and_runtime8_reports_preserved_unchanged') is True
                    and source_original.get('full_tracked_set_matches') is True
                    and source_original.get('actual_hash_match') is True
                    and source_original.get('source_clean_verified') is True
                    and source_original.get('actual_commit') == c['source_commit']
                    and source_original.get('sha256') == c['proofs']['source_manifest']['sha256']
                    and type(source_original.get('source_file_count')) is int
                    and type(source_original.get('actual_source_hash_matches')) is int
                    and source_original['source_file_count'] == source_original['actual_source_hash_matches'] == c['source_file_count']
                    and source_original.get('mismatches') == []
                    and isinstance(source_original.get('full490_records'), list)
                    and len(source_original['full490_records']) == c['source_file_count']
                    and relative_source_digest(source_original['full490_records']) == source_original.get('actual_full_source_digest') ==
                        proofs['source_manifest']['source_tracked_files_digest'],
                    'Genuine original runtime9 full source proof differs')
            covered_source_records(source_original['full490_records'], c, 'genuine runtime9 original full source proof')
        else:
            require((c['source_version'], c['runtime_version']) in ((9, 10), (10, 11), (11, 12))
                    and proofs['runtime_manifest']['live_scope'] == SCOPE,
                    'Continuation runtime10/11/12 cannot borrow a different source/version/scope')
            validate_reviewed_source_snapshot(source_original, c, proofs)
            validate_successor_cpu_policy(runtime, 'runtime', c, proofs)
    elif type(schema) is int and schema == 1:
        require(status == 'PASS' or status.startswith('PASS_STATIC'), 'Failed/non-static original runtime review')
        require(runtime['scope'] == 'INDEPENDENT_FROZEN_FRESH200_CALIBRATION_V' + version + '_STATIC_PREEXEC_REVIEW_ONLY',
                'Different original standalone runtime review scope')
        independence = runtime['independence']; decision = runtime['decision']; frozen = runtime['frozen_inputs']
        if status == 'PASS':
            require(type(runtime.get('runtime_version')) is int and runtime['runtime_version'] == c['runtime_version']
                    and runtime.get('blocking_findings') == []
                    and runtime.get('actual_gpu_acceptance') is False and runtime.get('production_READY') is False
                    and type(runtime.get('formal_iterations')) is int and runtime['formal_iterations'] == 0,
                    'Literal PASS runtime report has mismatched version/findings or inflated acceptance')
            require(independence.get('root_conversation_inherited') is False
                    and independence.get('independent_post_freeze_review') is True
                    and independence.get('original_fork_turns') == 'none'
                    and independence.get('completed_source_CPU_report_used_as_runtime_acceptance') is False,
                    'Literal PASS runtime report lacks exact original post-freeze independence')
            require(all(decision.get(key) is False for key in ('execution_authorized',
                'runtime_calibration_execution_authorized', 'native_execution_authorized',
                'single_submit_authorized', 'deployment_or_submission_authorized')),
                'Literal PASS runtime report widens original CPU-only permission')
            source_original = frozen['source_snapshot']; runtime_original = frozen['runtime']
            require(type(decision.get('formal_iterations_contributed')) is int
                    and decision['formal_iterations_contributed'] == 0
                    and type(runtime_original.get('actual_file_count')) is int
                    and runtime_original['actual_file_count'] == len(c['runtime_files'])
                    and isinstance(runtime_original.get('file_receipts'), list)
                    and all(row.get('actual_hash_match') is True for row in runtime_original['file_receipts'])
                    and mapped_manifest(dict(files=runtime_original['file_receipts']), c['helpers']) ==
                        closure(c['runtime_files'], 'literal PASS original runtime closure'),
                    'Literal PASS original full runtime closure differs')
            rows = source_original.get('full490_records')
            require(source_original.get('source_clean_verified') is True
                    and source_original.get('actual_commit') == c['source_commit']
                    and type(source_original.get('source_file_count')) is int
                    and type(source_original.get('actual_source_hash_matches')) is int
                    and source_original['source_file_count'] == source_original['actual_source_hash_matches'] == c['source_file_count']
                    and source_original.get('mismatches') == []
                    and isinstance(rows, list) and len(rows) == c['source_file_count']
                    and relative_source_digest(rows) == source_original.get('actual_full_source_digest') ==
                        proofs['source_manifest']['source_tracked_files_digest'],
                    'Literal PASS original full source closure differs')
            covered_source_records(rows, c, 'literal PASS original full source closure')
        cpu_decisions = [value for key, value in decision.items() if key in
                         ('CPU_test_execution_authorized_by_this_report',
                          'CPU_test_execution_authorized_by_this_PASS_report',
                          'CPU_test_execution_authorized_by_this_static_review')]
        require(independence['root_conversation_used_as_acceptance'] is False
                and independence['implementation_authorship'] is False
                and independence['prior_source_or_runtime_PASS_used_as_substitute'] is False
                and independence['review_started_from_explicit_standalone_handoff'] is True,
                'Original runtime review is not independent after freeze')
        require(type(decision['blocking_findings']) is int and decision['blocking_findings'] == 0
                and decision['whole_runtime_static_pass'] is True
                and len(cpu_decisions) == 1 and cpu_decisions[0] is True
                and all(decision[key] is False for key in ('ready', 'GPU_or_Ray_cluster_execution_authorized',
                    'probability_acceptance', 'actual_two_outer_acceptance', 'formal200_authorized'))
                and decision['formal_iterations_contributed'] == 0, 'Runtime static decision scope inflated')
        manifest = frozen['manifest']; source_snapshot = frozen['source_snapshot']
        require(manifest['sha256'] == c['proofs']['runtime_manifest']['sha256'] and manifest['actual_hash_match'] is True
                and manifest['runtime_file_count'] == manifest['actual_runtime_hash_matches'] == len(c['runtime_files'])
                and manifest['mismatches'] == []
                and source_snapshot['sha256'] == c['proofs']['source_manifest']['sha256']
                and source_snapshot['actual_hash_match'] is True and source_snapshot['source_import_or_execution'] is False,
                'Original runtime reviewed physical manifest/source snapshot differs')
    else:
        require(type(schema) is str and status.startswith('PASS_STATIC'), 'Failed/non-static original runtime review')
        require(runtime['schema'] == 'STRICT_CLEAN_CALIBRATION_PREEXEC_REVIEW_V' + version
                and runtime['scope'] == CALIBRATION_SCOPE and runtime['runtime_version'] == c['runtime_version']
                and runtime['independent_after_freeze'] is True and runtime['root_conversation_inherited'] is False
                and runtime['review_inputs']['manifest']['sha256'] == c['proofs']['runtime_manifest']['sha256']
                and runtime['review_inputs']['source_snapshot']['sha256'] == c['proofs']['source_manifest']['sha256']
                and runtime['snapshot_checks']['all_runtime_bytes_and_sha_match'] is True
                and runtime['snapshot_checks']['runtime_mismatches'] == []
                and runtime['snapshot_checks']['all_source_dependency_bytes_and_sha_match'] is True,
                'Original strict final runtime static review missing')
        require(mapped_manifest(dict(files=runtime['reviewed_runtime_files']), c['helpers']) == closure(c['runtime_files'], 'strict runtime'),
                'Strict runtime review is not the final actual helper bytes')
        auth = runtime['authorization']
        require(auth['cpu_execution_allowed'] is True and all(auth[key] is False for key in
            ('gpu_execution_allowed', 'calibration_execution_allowed', 'formal200_execution_allowed', 'READY_allowed')),
            'Runtime static review cannot claim actual GPU/READY')


def validate_controls_cpu(control_cpu, review, manifest, candidate, evidence,
                          read=read_bound, read_bytes=read_bound_bytes):
    """Use full raw collection identities; AST function names are documentation only."""
    c = candidate; policy = review['approved_cpu_policy']
    require(policy.get('cwd') == str(CONTROLS) and policy.get('test_files') == ['test_deployment_controls_v8.py']
            and policy.get('snapshot_sha256') == c['proofs']['controls_manifest']['sha256']
            and closure(policy.get('input_file_records'), 'approved six controls inputs') ==
                closure(c['control_files'], 'frozen six controls inputs'),
            'Controls CPU approval is not exact single fixture/six-module closure')
    require(policy.get('role') == 'controls'
            and policy.get('execution_scope') == 'FRESH200_DEPLOYMENT_CONTROLS_CPU_ONLY'
            and policy.get('single_execution_only') is True and policy.get('GPU_allowed') is False
            and policy.get('READY_allowed') is False and policy.get('Slurm_submission_allowed') is False
            and type(policy.get('formal_iterations_contributed')) is int and policy['formal_iterations_contributed'] == 0
            and policy.get('management_helper') == manifest['management_helper']
            and evidence.get(policy['management_helper']['path']) == core(policy['management_helper'])
            and manifest['cpu_required_test_files'] == policy['test_files'],
            'Controls CPU management helper/single-use scope differs from original approval')
    require(control_cpu['status'] == 'PASS' and control_cpu['scope'] == 'FRESH200_DEPLOYMENT_CONTROLS_CPU_ONLY'
            and control_cpu['actual_execution'] is True and control_cpu['gpu_used'] is False
            and control_cpu['actual_slurm_submitted'] is False and control_cpu['source_or_runtime_imported'] is False
            and type(control_cpu['exit_code']) is int and control_cpu['exit_code'] == 0
            and all(type(control_cpu[key]) is int and control_cpu[key] == 0 for key in ('failed', 'errors', 'skipped'))
            and type(control_cpu['passed']) is int and control_cpu['passed'] > 0
            and control_cpu['manifest_sha256'] == c['proofs']['controls_manifest']['sha256']
            and control_cpu['controls_files_digest'] == records_digest(c['control_files']),
            'Controls CPU not these tested bytes or skips/fails/errors present')
    originals = closure(control_cpu['raw_execution_records'], 'control CPU originals')
    require(all(evidence.get(path) == item for path, item in originals.items())
            and all(originals.get(control_cpu['current_run'][key]['path']) == core(control_cpu['current_run'][key])
                    for key in ('record', 'log_record', 'junit_record', 'collection_log_record'))
            and originals.get(cpu_collection_record(control_cpu['current_run'])['path']) == core(cpu_collection_record(control_cpu['current_run'])),
            'Control CPU originals unbound')
    start_record = control_cpu['current_run']['started_record']
    require(evidence.get(start_record['path']) == originals.get(start_record['path']) == core(start_record),
            'Original controls CPU STARTED record unbound')
    started = read(start_record)
    require(started.get('command') == policy['executed_command']
            and started.get('collection_command') == policy['collection_command']
            and started.get('cwd') == policy['cwd'] and started.get('interpreter') == policy['interpreter']
            and started.get('test_files') == policy['test_files'] and started.get('test_file_records') == policy['test_file_records']
            and started.get('management_helper') == policy['management_helper']
            and started.get('actual_execution') is True
            and started.get('review_sha256') == c['proofs']['controls_review']['sha256']
            and started.get('controls_manifest_sha256') == c['proofs']['controls_manifest']['sha256']
            and started.get('input_file_records_before') == policy['input_file_records']
            and started.get('input_files_digest_before') == policy['input_files_digest']
            and started.get('source_or_runtime_imported') is False and started.get('actual_slurm_submitted') is False,
            'Original controls CPU STARTED exact helper/invocation/input identity differs')
    validate_cpu_environment(started, policy)
    current = validate_full_cpu_run(control_cpu['current_run'], None,
        ('FRESH200_DEPLOYMENT_CONTROLS_CPU_ONLY',), evidence, policy, read, read_bytes)
    require(all(type(control_cpu[key]) is int and control_cpu[key] == current['counts'][key] for key in current['counts'])
            and control_cpu.get('executed_case_ids') == current['test_ids']
            and current['original_result']['controls_manifest_sha256'] == c['proofs']['controls_manifest']['sha256']
            and current['original_result']['controls_files_digest'] == records_digest(c['control_files'])
            and current['original_result']['source_or_runtime_imported'] is False
            and current['original_result']['actual_slurm_submitted'] is False
            and current['collection_result']['controls_manifest_sha256'] == c['proofs']['controls_manifest']['sha256']
            and current['collection_result']['controls_files_digest'] == records_digest(c['control_files'])
            and current['collection_result']['source_or_runtime_imported'] is False
            and current['collection_result']['actual_slurm_submitted'] is False
            and all(original.get('management_helper') == policy['management_helper']
                    and original.get('actual_execution') is True
                    and original.get('input_file_records') == policy['input_file_records']
                    and original.get('input_files_digest') == policy['input_files_digest']
                    for original in (current['collection_result'], current['original_result']))
            and type(current['original_result'].get('actual_cases')) is dict,
            'Controls CPU raw result/full fixture collection differs')
    require(current['original_result'].get('input_file_records_after') == policy['input_file_records']
            and current['original_result'].get('input_files_digest_after') == policy['input_files_digest']
            and started.get('controls_directory_names_before') == sorted(NAMES)
            and current['original_result'].get('controls_directory_names_before') == sorted(NAMES)
            and current['original_result'].get('controls_directory_names_after') == sorted(NAMES),
            'Controls CPU final input records differ from reviewed frozen six bytes')
    return current


def validate_candidate(candidate, read=read_bound):
    c = candidate
    require(c.get('schema') == 'FINAL_FRESH200_LIVE200_DEPLOYMENT_V8'
            and c.get('status') == 'FINAL_CANDIDATE_NOT_READY' and c.get('final_source') is True
            and c.get('final_runtime') is True, 'Historical/pending candidate cannot execute')
    validate_live_launch_plan(c)
    require(c['code_commit'] == c['source_commit'] and re.fullmatch('[0-9a-f]{40}', c['source_commit']), 'Unbound final source')
    repo = Path(c['repo'])
    require(repo.parent == ROOT and (repo.name == 'repo' or re.fullmatch(r'repo-[A-Za-z0-9_-]+', repo.name))
            and c['root'] == str(ROOT) and c['controls'] == str(CONTROLS)
            and c['common_root'] == str(COMMON) and c['environment_root'] == str(COMMON / 'env'), 'Wrong physical namespace')
    helpers = Path(c['helpers'])
    require(helpers == ROOT / 'tools/fresh200-continuation-v12'
            and type(c['runtime_version']) is int and c['runtime_version'] == 12
            and type(c['source_version']) is int and c['source_version'] == 11
            and repo.name == 'repo-v11-' + c['source_commit'][:7], 'Unreviewed legacy runtime/source')
    expected = dict(origin_kind='OFFICIAL_MODEL_INITIALIZATION', official_revision=REVISION, common_manifest_sha256=DATA,
        initial_outer_step=0, initial_adam_steps=0, initial_scheduler_step=0, initial_data_cursor=0,
        restore_checkpoint=None, restore_manifest_sha256=None, ppo_mini_batch_size=128,
        reference_kl_coefficient=0.0,
        optimizer_opportunities_per_outer=20, diagnostic_total_optimizer_opportunities=100,
        no_skip_adam_per_outer_per_rank=20, no_skip_gradient_micro_per_outer_per_rank=160,
        counters_are_no_skip_reference=True, full_outer_requires_all_opportunities_processed=True,
        full_outer_requires_at_least_one_actual_adam_write=True, actual_write_counts_must_be_reported_separately=True,
        zero_write_or_abandoned_outer_counts=False, retained_training_updates_contributed=0,
        counting_rules_source='human', counting_human_goal_updated_at=1791283983,
        outer_counting_authorization_source='human', outer_counting_human_goal_updated_at=1791283983,
        production_thresholds_selected=False,
        actor_token_identity='sampled_action_prefix_v1', capture_actual_runner_inputs=True, capture_actual_normalizer_inputs=True,
        calibration_phase='probability_probe_then_wait', candidate_old_policy_kl=0.2, target_old_policy_kl=0.2,
        debug_evaluation_rows=512, extra_stage64_evaluation=False, prefix_mismatch_limit=0, action_token_mismatch_limit=0)
    for key, value in expected.items():
        require(key in c and type(c[key]) is type(value) and c[key] == value, 'Different fresh protocol: ' + key)
    validate_human_kl(c)
    for key in ('source_native_acceptance', 'negative_tail_limit', 'actor_recompute_abs_tolerance',
                'actor_vllm_abs_tolerance', 'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance',
                'actual_gpu_acceptance', 'actual_probability_acceptance', 'production_thresholds'):
        require(key in c and c[key] is None, 'Unknown acceptance cannot be filled: ' + key)
    require(c['READY'] is False and c['submitted'] is False and c['probability_tolerance_evidence'] == [], 'Candidate scope inflated')
    require(type(c['probability_review_wait_seconds']) is int and 0 < c['probability_review_wait_seconds'] <= 43200, 'Invalid wait')
    require(c['official_model_path'] == str(COMMON / 'assets/qwen2.5-3b-instruct-aa8e725')
            and c['pad_token_id'] == 151643 and c['eos_token_id'] == 151645, 'Wrong official model/tokenizer identity')
    files = closure(c['files'], 'files'); evidence = closure(c['evidence'], 'evidence')
    parts = [closure(c[key], key) for key in ('source_files', 'runtime_files', 'control_files', 'asset_files', 'support_files')]
    combined = {}
    for part in parts:
        require(not set(combined).intersection(part), 'File groups overlap')
        combined.update(part)
    require(combined == files, 'Incomplete file group closure')
    require({str(CONTROLS / name) for name in NAMES} == set(parts[2]), 'All control/test bytes must be bound')
    require(type(c['source_file_count']) is int and c['source_file_count'] > 0
            and all(Path(path).is_relative_to(repo) for path in parts[0])
            and len(parts[0]) == c['source_file_count'], 'Incomplete source')
    require(all(Path(path).parent == helpers for path in parts[1]), 'Runtime outside exact helper namespace')
    require(set(c['proofs']) == set(ROLES) and all(evidence.get(r['path']) == core(r) for r in c['proofs'].values()), 'Proof closure incomplete')
    proofs = {name: read(item) for name, item in c['proofs'].items()}
    validate_static_reviews(c, proofs)
    for kind in ('source', 'runtime'):
        validate_cpu(proofs[kind + '_cpu'], proofs[kind + '_cpu_review'], kind, c, evidence, read)
    covered_source_records(proofs['source_manifest']['changed_or_new_source_files'], c, 'final reviewed source manifest')
    runtime_manifest = proofs['runtime_manifest']
    require(relative_runtime_closure(runtime_manifest['files'], helpers) == parts[1]
            and runtime_manifest['schema'] == 'gigpo-e1-fresh200-continuation-runtime-static-manifest'
            and runtime_manifest['live_scope'] == SCOPE and runtime_manifest['authoring_stopped'] is True
            and type(runtime_manifest['runtime_version']) is int and runtime_manifest['runtime_version'] == 12
            and type(runtime_manifest['file_count']) is int and runtime_manifest['file_count'] == len(parts[1])
            and runtime_manifest['canonical_future_runtime_root'] == str(helpers)
            and runtime_manifest['canonical_future_source_repo'] == str(repo)
            and runtime_manifest['source_commit'] == c['source_commit']
            and (runtime_manifest['source_snapshot_record']['bytes'], runtime_manifest['source_snapshot_record']['sha256']) ==
                (c['proofs']['source_manifest']['bytes'], c['proofs']['source_manifest']['sha256'])
            and runtime_manifest['source_file_count'] == c['source_file_count']
            and runtime_manifest['source_tracked_files_digest'] == proofs['source_manifest']['source_tracked_files_digest']
            and relative_source_digest(runtime_manifest['files']) == runtime_manifest['runtime_files_digest'],
            'Final runtime12 manifest/source dependency/full relative byte closure differs')
    require(mapped_manifest(proofs['controls_manifest'], CONTROLS) == parts[2], 'Controls manifest bytes differ')
    review = proofs['controls_review']
    require(review['schema'] == 'STRICT_CLEAN_DEPLOYMENT_CONTROLS_PREEXEC_REVIEW_V8'
            and review['status'] in ('PASS', 'PASS_STATIC_ONLY_CPU_SCOPE')
            and review['scope'] == 'FRESH200_DEPLOYMENT_CONTROLS_PREEXEC_STATIC_ONLY'
            and review['independent_review'] is True and review['manifest_sha256'] == c['proofs']['controls_manifest']['sha256']
            and review['blocking_findings'] == [] and review['actual_gpu_acceptance'] is False
            and review['production_READY'] is False and review['submitted'] is False
            and type(review['formal_iterations']) is int and review['formal_iterations'] == 0
            and review['controls_files_digest'] == records_digest(c['control_files'])
            and review['reviewer']['fork_turns'] == 'none' and review['reviewer']['root_conversation_inherited'] is False
            and review['reviewer']['implementation_authorship'] is False and review['reviewer']['independent_after_freeze'] is True
            and type(review['decision']['blocking_findings']) is int and review['decision']['blocking_findings'] == 0
            and review['decision']['whole_controls_static_pass'] is True
            and review['decision']['CPU_fixture_execution_authorized_by_this_report'] is True
            and all(review['decision'][key] is False for key in ('READY_allowed', 'GPU_execution_allowed',
                'calibration_execution_allowed', 'single_submit_allowed', 'formal200_execution_allowed',
                'probability_acceptance', 'actual_two_outer_acceptance')),
            'No exact original controls static CPU-only decision')
    validate_controls_cpu(proofs['controls_cpu'], review, proofs['controls_manifest'], c, evidence, read)
    for name, (size, digest) in PROTECTED.items():
        require(parts[1].get(str(helpers / name)) == dict(path=str(helpers / name), bytes=size, sha256=digest), 'Protected runtime/header changed')
    for key, name in (('resource_guard', 'resource_guard_v2.py'), ('resource_ownership', 'resource_ownership_v2.py')):
        r = c[key]
        require(Path(r['path']).parent == ROOT / 'tools/fresh200-guard-v2' and Path(r['path']).name == name
                and r['sha256'] == GUARDS[name] and files[r['path']]['sha256'] == r['sha256'], 'Protected guard changed')
    require(c['retrieval_verifier_path'] == str(COMMON / 'tools/check_http_retrieval.py')
            and files[c['retrieval_verifier_path']]['sha256'] == HTTP_SHA
            and files[str(ROOT / 'logs/cpu-gpu-retrieval-comparison.json')]['sha256'] == REFERENCE_SHA
            and files[str(COMMON / 'tools/expected-gpu-uuids.json')]['sha256'] == UUID_SHA, 'HTTP/reference/canonical UUID dependency changed')
    validate_canonical_uuid(c, files)
    require(c['hf_verifier_path'] == str(COMMON / 'tools/verify-gigpo-checkpoint.py')
            and str(COMMON / 'tools/checkpoint_semantics.py') in files, 'HF semantic verifier dependency missing')
    allowed_support = {str(COMMON / 'tools' / name) for name in ('check_http_retrieval.py', 'verify-gigpo-checkpoint.py',
        'checkpoint_semantics.py', 'common-retrieval-identity.json', 'expected-gpu-uuids.json')}
    allowed_support.update({str(COMMON / 'env/bin/python'), str(ROOT / 'logs/cpu-gpu-retrieval-comparison.json')})
    allowed_support.update(str(ROOT / 'tools/fresh200-guard-v2' / name) for name in GUARDS)
    require(set(parts[4]) == allowed_support, 'Extra/missing support file or old restore source')
    validate_assets(c, proofs)
    allowed_assets = (SHARED / 'env', SHARED / 'retriever-gpu-lib', SHARED / 'tools/python310-headers-v1',
                      SHARED / 'assets-root/assets', COMMON / 'assets/qwen2.5-3b-instruct-aa8e725', COMMON / 'common-data')
    require(all(any(Path(path).is_relative_to(prefix) for prefix in allowed_assets) for path in parts[3]), 'Old source/checkpoint is not an asset')
    return proofs


def validate_physical(value, candidate, candidate_sha):
    require(value['status'] == 'PHYSICAL_NEW_CODE_SHA_INHERITED_ASSET_IDENTITY_ONLY_NOT_GPU'
            and value['candidate_sha256'] == candidate_sha and value['source_commit'] == candidate['source_commit']
            and value['source_clean'] is True and value['source_tracked_file_count'] == candidate['source_file_count']
            and value['hostname'] == 'h100-8' and value['source_imported'] is False and value['cuda_initialized'] is False,
            'Physical scope/source differs')
    require(closure(value['metadata'], 'physical files') == closure(candidate['files'], 'candidate files')
            and closure(value['evidence'], 'physical evidence') == closure(candidate['evidence'], 'candidate evidence'), 'Physical bytes/SHA closure differs')
    require(value['asset_metadata_sha256'] == candidate['proofs']['asset_metadata']['sha256']
            and value['runtime_all_file_sha_before_gpu_required'] is True
            and value['expected_gpu_uuids'] == candidate['expected_gpu_uuid_inventory']
            and value['canonical_uuid_inventory_record'] == candidate['canonical_uuid_inventory_record']
            and value['canonical_uuid_plain_array_read_only'] is True
            and value['future_actual_exclusive_slurm_uuid_ofd_admission_required'] is True
            and value['old_canonical_inventory_is_not_allocation_or_lease'] is True
            and value['actual_gpu_acceptance'] is False and value['actual_probability_acceptance'] is False
            and value['submitted'] is False, 'Physical dependency binding differs')
    require(value['environment_root_resolved'] == str(SHARED / 'env')
            and value['environment_python_resolved'] == '/usr/bin/python3.10', 'Different environment identity')


def validate_ready(value, read=read_bound):
    require(value['status'] == 'REVIEWED_FOR_FRESH200_LIVE_TRAINING_ONLY' and value['READY'] is True
            and value['readiness'] == 'READY_FOR_FRESH200_SAME_PROCESS_FIRST5_THEN200_GATED_LAUNCH', 'No fresh live READY')
    candidate = read(value['candidate_record']); validate_candidate(candidate, read)
    physical = read(value['physical_record']); validate_physical(physical, candidate, value['candidate_record']['sha256'])
    review = read(value['deployment_review'])
    require(review['status'] == 'PASS' and review['scope'] == SCOPE and review['independent_review'] is True
            and review['candidate_sha256'] == value['candidate_record']['sha256']
            and review['physical_sha256'] == value['physical_record']['sha256']
            and review['source_commit'] == candidate['source_commit']
            and review['controls_manifest_sha256'] == candidate['proofs']['controls_manifest']['sha256']
            and review['controls_cpu_sha256'] == candidate['proofs']['controls_cpu']['sha256']
            and review['actual_gpu_acceptance'] is False and review['actual_probability_acceptance'] is False,
            'Different/inflated deployment acceptance')
    validate_live_deployment_review(review, candidate)
    for key in candidate:
        require(key in ('status', 'READY', 'evidence') or value[key] == candidate[key], 'READY changed candidate field: ' + key)
    wanted = list(candidate['evidence']) + [value[key] for key in ('candidate_record', 'physical_record', 'deployment_review')]
    require(closure(value['evidence'], 'READY evidence') == closure(wanted, 'required READY evidence'), 'READY evidence changed')
    require(review['freeze_helper_sha256'] == closure(candidate['control_files'], 'controls')[str(CONTROLS / 'freeze_ready_v8.py')]['sha256']
            and review['submit_helper_sha256'] == closure(candidate['control_files'], 'controls')[str(CONTROLS / 'submit_once_v8.py')]['sha256'],
            'Reviewed freeze/submit bytes differ')
    return candidate
