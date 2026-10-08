"""Exact final fresh200 deployment inputs. Stdlib only; no native/GPU gate."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
COMMON = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
SHARED = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo')
CONTROLS = ROOT / 'tools/fresh200-deployment-controls-v1'
STATE = ROOT / 'logs/fresh200-calibration2-submit-v1'
CANDIDATE, PHYSICAL, READY, BINDING, INTENT, RECEIPT = (
    STATE / name for name in ('CANDIDATE.json', 'PHYSICAL.json', 'READY.json',
                             'BINDING.json', 'SUBMIT-INTENT.json', 'SUBMIT-RECEIPT.json'))
SCOPE = 'GIGPO_POLICY_SAFETY_OFFICIAL_INSTRUCT_FRESH200_CALIBRATION_V1'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
NAMES = ('deployment_contract_v1.py', 'prepare_candidate_v1.py', 'verify_physical_v1.py',
         'freeze_ready_v1.py', 'submit_once_v1.py', 'test_deployment_controls_v1.py')
ROLES = ('source_manifest', 'source_review', 'source_cpu', 'source_cpu_review',
         'runtime_manifest', 'runtime_review', 'runtime_cpu', 'runtime_cpu_review',
         'controls_manifest', 'controls_review', 'controls_cpu', 'asset_manifest', 'asset_metadata')
PROTECTED = {
    'calibration_v1.sbatch': (879, 'b7d4be4767002f2b70081b59a37ecec8b1c43ed23d1588fbdb64e433544206af'),
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


def validate_canonical_uuid(candidate, files):
    expected = dict(path=str(COMMON / 'tools/expected-gpu-uuids.json'), bytes=352, sha256=UUID_SHA)
    require(core(candidate['canonical_uuid_inventory_record']) == expected and files.get(expected['path']) == expected
            and candidate['expected_gpu_uuid_inventory'] == UUIDS
            and candidate['canonical_uuid_inventory_is_hardware_only'] is True
            and candidate['actual_slurm_selected_uuid_comparison_before_guard_required'] is True,
            'Canonical plain-eight hardware UUID input is not exact/live-allocation comparison required')


def read_bound(item):
    closure([item], 'bound JSON')
    path = Path(item['path'])
    require(path.resolve() == path and not path.is_symlink() and path.is_file(), 'Noncanonical proof: ' + str(path))
    require(record(path) == core(item), 'Changed bound JSON: ' + str(path))
    return json.loads(path.read_bytes())


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
        name = item.get('relative_path', Path(item['path']).name)
        require(isinstance(name, str) and Path(name).name == name and name not in ('', '.', '..'), 'Nested/invalid helper name')
        result.append(dict(path=str(Path(directory) / name), bytes=item['bytes'], sha256=item['sha256']))
    return closure(result, 'mapped manifest')


def covered_source_records(items, candidate, label):
    wanted = closure(candidate['source_files'], 'source')
    require(isinstance(items, list) and items, 'Missing ' + label)
    for item in items:
        name = item['relative_path']
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid covered source path')
        path = str(Path(candidate['repo']) / name)
        require(wanted.get(path) == dict(path=path, bytes=item['bytes'], sha256=item['sha256']), 'Covered source changed: ' + label)
    require(len({item['relative_path'] for item in items}) == len(items), 'Duplicate covered source: ' + label)


def validate_cpu(receipt, review, kind, candidate, evidence, read=read_bound):
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
    if kind == 'source':
        require(receipt['scope'] in ('ALL_CURRENT_CPU_REGRESSIONS', 'RELEVANT_CHANGED_CPU_REGRESSIONS'), 'Wrong source CPU scope')
        covered_source_records(receipt['current_source_and_test_files'], candidate, 'current CPU source/test')
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
                        and len(prior['superseded_failure_ids']) == prior['failed'], 'Missing preserved actual prior CPU evidence')
                actual_prior = read(prior['record']); prior_source = read(prior['source_manifest_record'])
                require(actual_prior['code_commit'] == prior_source['source_commit'] == prior['source_commit']
                        and actual_prior['gpu_acceptance'] is False and actual_prior['production_ready'] is False,
                        'Preserved actual CPU/source identity differs')
                old_files = closure(prior_source['files'], 'prior CPU source identity')
                old_repo = Path(prior_source['repo'])
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


def validate_static_reviews(candidate, proofs):
    """Consume the original strict reports; no replacement generic PASS is made."""
    c = candidate; source = proofs['source_review']; runtime = proofs['runtime_review']
    require(source['schema'] == c['source_review_schema']
            and isinstance(source['schema'], str) and source['schema'].startswith('gigpo-e1-fresh200-')
            and source['schema'].endswith('-preexec-review')
            and source['status'] == 'PASS_STATIC_ONLY_CPU_SCOPE'
            and source['reviewer']['root_conversation_inherited'] is False
            and any(value is True for key, value in source['reviewer'].items()
                    if key.startswith('independent_after_') and key.endswith('_freeze'))
            and source['blocking_findings'] == [],
            'Original strict final source static review missing')
    require(any(item['sha256'] == c['proofs']['source_manifest']['sha256']
                and item['independently_hash_matched'] is True for item in source['freeze']), 'Source review is a different snapshot')
    covered_source_records(source['file_receipts'], c, 'strict source review original files')
    require({r['relative_path']: (r['bytes'], r['sha256']) for r in source['file_receipts']} ==
            {r['relative_path']: (r['bytes'], r['sha256']) for r in proofs['source_manifest']['files']},
            'Strict source review omits final snapshot file')
    require(source['cpu_execution_decision']['authorized_by_this_static_review'] is True
            and source['cpu_execution_decision']['actual_8GPU_acceptance'] is False
            and source['cpu_execution_decision']['production_READY'] is False, 'Source static scope inflated')
    require(runtime['schema'] == c['runtime_review_schema'] and isinstance(runtime['status'], str)
            and runtime['status'].startswith('PASS_STATIC'), 'Failed/non-static original runtime review')
    version = str(c['runtime_version'])
    if runtime['schema'] == 1:
        require(runtime['scope'] == 'INDEPENDENT_FROZEN_FRESH200_CALIBRATION_V' + version + '_STATIC_PREEXEC_REVIEW_ONLY',
                'Different original standalone runtime review scope')
        independence = runtime['independence']; decision = runtime['decision']; frozen = runtime['frozen_inputs']
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
        require(runtime['schema'] == 'STRICT_CLEAN_CALIBRATION_PREEXEC_REVIEW_V' + version
                and runtime['scope'] == SCOPE and runtime['runtime_version'] == c['runtime_version']
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


def validate_candidate(candidate, read=read_bound):
    c = candidate
    require(c.get('schema') == 'FINAL_FRESH200_CALIBRATION2_DEPLOYMENT_V1'
            and c.get('status') == 'FINAL_CANDIDATE_NOT_READY' and c.get('final_source') is True
            and c.get('final_runtime') is True, 'Historical/pending candidate cannot execute')
    require(c['validation_scope'] == c['calibration_scope'] == SCOPE and c['allowed_safety_modes'] == ['calibration'], 'Wrong scope')
    require(c['code_commit'] == c['source_commit'] and re.fullmatch('[0-9a-f]{40}', c['source_commit']), 'Unbound final source')
    repo = Path(c['repo'])
    require(repo.parent == ROOT and (repo.name == 'repo' or re.fullmatch(r'repo-[A-Za-z0-9_-]+', repo.name))
            and c['root'] == str(ROOT) and c['controls'] == str(CONTROLS)
            and c['common_root'] == str(COMMON) and c['environment_root'] == str(COMMON / 'env'), 'Wrong physical namespace')
    helpers = Path(c['helpers'])
    require(helpers.parent == ROOT / 'tools' and helpers.name.startswith('fresh200-calibration-v')
            and type(c['runtime_version']) is int and c['runtime_version'] >= 5
            and type(c['source_version']) is int and c['source_version'] >= 5, 'Unreviewed legacy runtime/source')
    expected = dict(origin_kind='OFFICIAL_MODEL_INITIALIZATION', official_revision=REVISION, common_manifest_sha256=DATA,
        initial_outer_step=0, initial_adam_steps=0, initial_scheduler_step=0, initial_data_cursor=0,
        restore_checkpoint=None, restore_manifest_sha256=None, ppo_mini_batch_size=128,
        reference_kl_coefficient=0.0, calibration_actual_outer_budget=2,
        optimizer_opportunities_per_outer=20, diagnostic_total_optimizer_opportunities=40,
        no_skip_adam_per_outer_per_rank=20, no_skip_gradient_micro_per_outer_per_rank=160,
        counters_are_no_skip_reference=True, full_outer_requires_all_opportunities_processed=True,
        full_outer_requires_at_least_one_actual_adam_write=True, actual_write_counts_must_be_reported_separately=True,
        zero_write_or_abandoned_outer_counts=False, retained_training_updates_contributed=0,
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
    covered_source_records(proofs['source_manifest']['files'], c, 'final reviewed source manifest')
    require(mapped_manifest(proofs['runtime_manifest'], helpers) == parts[1], 'Runtime manifest bytes differ')
    runtime_manifest = proofs['runtime_manifest']; source_dependencies = runtime_manifest['source_dependency_snapshot']
    snapshots = [item for key, item in source_dependencies.items() if key.startswith('frozen_')
                 and key.endswith('_snapshot') and isinstance(item, dict) and 'sha256' in item]
    require(runtime_manifest['scope'] == SCOPE and runtime_manifest['runtime_version'] == c['runtime_version']
            and len(snapshots) == 1 and snapshots[0]['sha256'] == c['proofs']['source_manifest']['sha256'],
            'Runtime manifest is not bound to this final source snapshot')
    covered_source_records(source_dependencies['files'], c, 'runtime frozen source dependency')
    covered_source_records(source_dependencies['supplemental_dependencies'], c, 'runtime supplemental source dependency')
    require({r['relative_path']: (r['bytes'], r['sha256']) for r in source_dependencies['files']} ==
            {r['relative_path']: (r['bytes'], r['sha256']) for r in proofs['source_manifest']['files']},
            'Runtime source dependency omits final reviewed snapshot bytes')
    require(mapped_manifest(proofs['controls_manifest'], CONTROLS) == parts[2], 'Controls manifest bytes differ')
    review = proofs['controls_review']
    require(review['status'] == 'PASS' and review['scope'] == 'FRESH200_DEPLOYMENT_CONTROLS_PREEXEC_STATIC_ONLY'
            and review['independent_review'] is True and review['manifest_sha256'] == c['proofs']['controls_manifest']['sha256']
            and review['controls_files_digest'] == records_digest(c['control_files']), 'No exact controls static review')
    control_cpu = proofs['controls_cpu']
    require(control_cpu['status'] == 'PASS' and control_cpu['scope'] == 'FRESH200_DEPLOYMENT_CONTROLS_CPU_ONLY'
            and control_cpu['actual_execution'] is True and control_cpu['gpu_used'] is False
            and control_cpu['actual_slurm_submitted'] is False and control_cpu['source_or_runtime_imported'] is False
            and type(control_cpu['exit_code']) is int and control_cpu['exit_code'] == 0
            and type(control_cpu['failed']) is int and control_cpu['failed'] == 0
            and type(control_cpu['passed']) is int and control_cpu['passed'] > 0
            and control_cpu['test_functions_executed'] == proofs['controls_manifest']['static_checks']['test_functions_unexecuted']
            and control_cpu['manifest_sha256'] == c['proofs']['controls_manifest']['sha256']
            and control_cpu['controls_files_digest'] == records_digest(c['control_files']), 'Controls CPU not these tested bytes')
    require(all(evidence.get(path) == item for path, item in closure(control_cpu['raw_execution_records'], 'control CPU originals').items()),
            'Control CPU originals unbound')
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
    expected_assets = [item for rows in proofs['asset_manifest']['groups'].values() for item in rows]
    require(closure(expected_assets, 'accepted assets') == parts[3], 'Inherited immutable asset closure differs')
    prior = closure(proofs['asset_metadata']['metadata'], 'inherited physical metadata')
    require(all(prior.get(path) == item for path, item in parts[3].items()), 'Asset SHA lacks physical provenance')
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
    require(value['status'] == 'REVIEWED_FOR_CALIBRATION_ONLY' and value['READY'] is True
            and value['readiness'] == 'READY_FOR_FRESH200_CALIBRATION2_PROBABILITY_PROBE_THEN_WAIT_ONLY', 'No fresh READY')
    candidate = read(value['candidate_record']); validate_candidate(candidate, read)
    physical = read(value['physical_record']); validate_physical(physical, candidate, value['candidate_record']['sha256'])
    review = read(value['deployment_review'])
    require(review['status'] == 'PASS' and review['scope'] == SCOPE and review['independent_review'] is True
            and review['candidate_sha256'] == value['candidate_record']['sha256']
            and review['physical_sha256'] == value['physical_record']['sha256']
            and review['source_commit'] == candidate['source_commit']
            and review['controls_manifest_sha256'] == candidate['proofs']['controls_manifest']['sha256']
            and review['controls_cpu_sha256'] == candidate['proofs']['controls_cpu']['sha256']
            and review['calibration_launch_only'] is True and review['formal200_authorized'] is False
            and review['actual_gpu_acceptance'] is False and review['actual_probability_acceptance'] is False,
            'Different/inflated deployment acceptance')
    for key in candidate:
        require(key in ('status', 'READY', 'evidence') or value[key] == candidate[key], 'READY changed candidate field: ' + key)
    wanted = list(candidate['evidence']) + [value[key] for key in ('candidate_record', 'physical_record', 'deployment_review')]
    require(closure(value['evidence'], 'READY evidence') == closure(wanted, 'required READY evidence'), 'READY evidence changed')
    require(review['freeze_helper_sha256'] == closure(candidate['control_files'], 'controls')[str(CONTROLS / 'freeze_ready_v1.py')]['sha256']
            and review['submit_helper_sha256'] == closure(candidate['control_files'], 'controls')[str(CONTROLS / 'submit_once_v1.py')]['sha256'],
            'Reviewed freeze/submit bytes differ')
    return candidate
