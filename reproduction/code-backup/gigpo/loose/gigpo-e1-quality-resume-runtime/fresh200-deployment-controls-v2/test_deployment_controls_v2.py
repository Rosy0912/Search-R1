"""CPU fixtures only, currently unexecuted. Never invoke real Slurm or source."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import pytest
import deployment_contract_v2 as contract
import submit_once_v2 as submission


def test_frozen_ff7_observation_cannot_become_final_candidate():
    with pytest.raises(ValueError, match='Historical/pending candidate'):
        contract.validate_candidate(dict(status='CANDIDATE_NOT_READY_FINAL_RUNTIME_DEPENDENCY_PENDING'))


def test_bound_original_rejects_same_length_different_bytes(tmp_path):
    path = tmp_path / 'original.json'; path.write_text('{"value":1}')
    item = contract.record(path); path.write_text('{"value":2}')
    with pytest.raises(ValueError, match='Changed bound bytes'):
        contract.read_bound(item)


def test_verified_json_is_parsed_without_reopening_path(tmp_path, monkeypatch):
    path = tmp_path / 'one-buffer.json'; path.write_bytes(b'{"value":1}')
    item = contract.record(path)
    def forbidden_reopen(*args, **kwargs):
        raise AssertionError('JSON parsing must use the verified open-descriptor buffer')
    monkeypatch.setattr(Path, 'read_bytes', forbidden_reopen)
    assert contract.read_bound(item) == dict(value=1)


def test_replacement_during_bound_read_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'review.json'; path.write_bytes(b'{"value":1}')
    item = contract.record(path); replacement = tmp_path / 'replacement.json'; replacement.write_bytes(b'{"value":2}')
    original = contract.os.read; replaced = []
    def replace_after_read(descriptor, size):
        block = original(descriptor, size)
        if block and not replaced:
            replacement.replace(path); replaced.append(True)
        return block
    monkeypatch.setattr(contract.os, 'read', replace_after_read)
    with pytest.raises(ValueError, match='identity changed during read'):
        contract.read_bound(item)
    assert replaced


def test_duplicate_review_json_key_is_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'; path.write_bytes(b'{"status":"FAIL","status":"PASS"}')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        contract.read_bound(contract.record(path))


def test_symlink_bound_proof_cannot_alias_a_review(tmp_path):
    path = tmp_path / 'original.json'; path.write_bytes(b'{}')
    link = tmp_path / 'alias.json'; link.symlink_to(path)
    with pytest.raises(ValueError, match='Noncanonical proof'):
        contract.read_bound(contract.record(link))


def test_exclusive_output_preserves_first_intent(tmp_path):
    path = tmp_path / 'intent.json'
    contract.write_once(path, dict(status='CLAIMED_NO_RESUBMISSION'))
    first = path.read_bytes()
    with pytest.raises(FileExistsError):
        contract.write_once(path, dict(status='replacement'))
    assert path.read_bytes() == first


def test_duplicate_closure_cannot_hide_omitted_rank_or_file():
    item = dict(path='/absolute/test.json', bytes=1, sha256='a' * 64)
    with pytest.raises(ValueError, match='Invalid/duplicate'):
        contract.closure([item, dict(item)], 'files')


def human_kl_fixture():
    return dict(token_kl_threshold_source='human', token_kl_human_goal_updated_at=1791282908,
                candidate_old_policy_kl=.2, target_old_policy_kl=.2, negative_tail_limit=None)


def test_human_point_two_is_explicit_and_tail_is_still_pending():
    contract.validate_human_kl(human_kl_fixture())


def test_obsolete_point_zero_one_cannot_replace_latest_human_choice():
    value = human_kl_fixture(); value['candidate_old_policy_kl'] = .01
    with pytest.raises(ValueError, match='human sampled-policy KL0.2'):
        contract.validate_human_kl(value)


def test_unmeasured_tail_cannot_be_filled_by_token_kl_choice():
    value = human_kl_fixture(); value['negative_tail_limit'] = .2
    with pytest.raises(ValueError, match='tail-pending'):
        contract.validate_human_kl(value)


def canonical_uuid_fixture():
    item = dict(path=str(contract.COMMON / 'tools/expected-gpu-uuids.json'), bytes=352, sha256=contract.UUID_SHA)
    return dict(canonical_uuid_inventory_record=item, expected_gpu_uuid_inventory=list(contract.UUIDS),
                canonical_uuid_inventory_is_hardware_only=True,
                actual_slurm_selected_uuid_comparison_before_guard_required=True), contract.closure([item], 'UUID fixture')


def test_exact_plain_canonical_eight_requires_later_actual_guard_comparison():
    value, files = canonical_uuid_fixture(); contract.validate_canonical_uuid(value, files)
    value['actual_slurm_selected_uuid_comparison_before_guard_required'] = False
    with pytest.raises(ValueError, match='live-allocation comparison required'):
        contract.validate_canonical_uuid(value, files)


def test_same_count_different_uuid_is_not_hardware_acceptance():
    value, files = canonical_uuid_fixture(); value['expected_gpu_uuid_inventory'][0] = 'GPU-other'
    with pytest.raises(ValueError, match='Canonical plain-eight'):
        contract.validate_canonical_uuid(value, files)


def relevant_cpu_fixture(current_scope='RELEVANT_CHANGED_CPU_REGRESSIONS'):
    repo = '/new/repo'
    changed = dict(relative_path='search_r1/baselines/instruct300.py', bytes=12, sha256='a' * 64)
    test = dict(relative_path='tests/test_checkpoint_controls.py', bytes=13, sha256='b' * 64)
    unchanged = dict(relative_path='search_r1/baselines/policy_safety.py', bytes=14, sha256='c' * 64)
    def source(item):
        return dict(path=repo + '/' + item['relative_path'], bytes=item['bytes'], sha256=item['sha256'])
    buffers = {}
    def bind(path, raw):
        buffers[path] = raw
        return dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    def bind_json(path, value):
        return bind(path, json.dumps(value, sort_keys=True).encode())
    failure_id = 'tests.fixture::failed_original_case'
    def make_run(label, commit, passed, failed, scope):
        counts = dict(collected=passed + failed, passed=passed, failed=failed, errors=0, skipped=0)
        cases = ''.join('<testcase classname="tests.fixture" name="case' + str(index) + '" />' for index in range(passed))
        if failed:
            cases += '<testcase classname="tests.fixture" name="failed_original_case"><failure message="preserved" /></testcase>'
        xml = ('<testsuites><testsuite hostname="h100-8" tests="' + str(passed + failed) + '" failures="' + str(failed) +
               '" errors="0" skipped="0">' + cases + '</testsuite></testsuites>').encode()
        log = bind('/proofs/' + label + '.log', b'CPU fixture raw log; no actual training execution\n')
        junit = bind('/proofs/' + label + '.xml', xml)
        command = ['python3', '-m', 'pytest', test['relative_path'], '--junitxml=' + junit['path']]
        actual = dict(code_commit=commit, scope=scope, node='h100-8', command=command, test_files=[test['relative_path']],
            log=log['path'], log_sha256=log['sha256'], junit=junit['path'], junit_sha256=junit['sha256'],
            exit_code=int(bool(failed)), actual_cases=counts, gpu_acceptance=False, production_ready=False, formal_iterations=0,
            review_sha256='6' * 64, source_snapshot_sha256='1' * 64)
        result = bind_json('/proofs/' + label + '-RESULT.json', actual)
        return dict(record=result, log_record=log, junit_record=junit, execution_scope=scope,
            executed_command=command, test_files=[test['relative_path']], exit_code=actual['exit_code'],
            failure_ids=[failure_id] if failed else [], **counts)
    current = make_run('current', '9' * 40, 41, 0, current_scope)
    prior = make_run('preserved-175', '8' * 40, 174, 1, 'REVIEWED_CPU_ONLY_NO_CUDA_RAY_OR_REAL_DISTRIBUTED')
    patch = bind('/proofs/minimal-semantic.patch', b'Exact synthetic CPU fixture patch only\n')
    old_manifest = bind_json('/proofs/actual-source-v3-identity.json', dict(source_commit='8' * 40, repo='/historical/repo',
        files=[dict(path='/historical/repo/' + row['relative_path'], bytes=row['bytes'], sha256=row['sha256'])
               for row in (unchanged, test)]))
    candidate = dict(repo=repo, source_commit='9' * 40, source_files=[source(x) for x in (changed, test, unchanged)],
        proofs=dict(source_manifest=dict(sha256='1' * 64), source_cpu=dict(sha256='2' * 64), source_review=dict(sha256='6' * 64)))
    prior.update(source_commit='8' * 40, source_manifest_record=old_manifest, unchanged_source_files=[unchanged],
                 superseded_failure_ids=[failure_id])
    raw_records = [run[key] for run in (current, prior) for key in ('record', 'log_record', 'junit_record')]
    receipt = dict(status='PASS', scope=current_scope, actual_execution=True,
        gpu_used=False, exit_code=0, failed=0, passed=41, errors=0, skipped=0, collected=41, source_commit=candidate['source_commit'],
        manifest_sha256='1' * 64, raw_execution_records=raw_records, current_run=current, current_source_and_test_files=[changed, test],
        claims_all_prior_cases_rerun_on_current_commit=False, minimal_patch_record=patch,
        unchanged_source_files=[unchanged], prior_runs=[prior], superseded_prior_failure_ids=[failure_id])
    review = dict(status='PASS', scope='FRESH200_SOURCE_CPU_EVIDENCE_ONLY', source_commit=candidate['source_commit'],
        manifest_sha256='1' * 64, cpu_receipt_sha256='2' * 64, independent_review=True,
        complete_required_cpu_coverage_accepted=True,
        superseded_prior_failure_ids=receipt['superseded_prior_failure_ids'],
        unchanged_source_files_digest=contract.records_digest([source(unchanged)]))
    def fake_bytes(item):
        raw = buffers[item['path']]
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
        return raw
    def fake_read(item):
        return json.loads(fake_bytes(item))
    return candidate, receipt, review, contract.closure(raw_records + [patch, old_manifest], 'CPU evidence'), fake_read, fake_bytes


def test_changed_cpu_plus_preserved_174_pass_one_superseded_fail_is_explicit():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    contract.validate_cpu(receipt, review, 'source', candidate, evidence, read, read_bytes)
    assert receipt['prior_runs'][0]['failed'] == 1
    assert receipt['claims_all_prior_cases_rerun_on_current_commit'] is False


def test_unchanged_source_sha_drift_invalidates_prior_cpu():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    receipt['unchanged_source_files'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Covered source changed'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read, read_bytes)


def test_failed_current_semantic_loader_never_accepts_prior_cpu():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture(); receipt['failed'] = 1
    with pytest.raises(ValueError, match='Actual successful CPU'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read, read_bytes)


def test_prior_174_cannot_be_reported_as_current_commit_rerun():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    receipt['claims_all_prior_cases_rerun_on_current_commit'] = True
    with pytest.raises(ValueError, match='Prior cases cannot be called rerun'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read, read_bytes)


def test_prior_original_result_commit_cannot_be_relabelled():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    original_read = read
    def different_commit(item):
        result = original_read(item)
        if item == receipt['prior_runs'][0]['record']:
            result['code_commit'] = '7' * 40
        return result
    with pytest.raises(ValueError, match='Actual original CPU scope/commit/host differs'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, different_commit, read_bytes)


def test_unexecuted_cpu_fixture_status_cannot_be_promoted():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture(); receipt['actual_execution'] = False
    with pytest.raises(ValueError, match='Actual successful CPU evidence missing'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read, read_bytes)


def test_complete_current_cpu_does_not_require_a_historical_composite():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture('ALL_CURRENT_CPU_REGRESSIONS')
    receipt.pop('prior_runs')
    contract.validate_cpu(receipt, review, 'source', candidate, evidence, read, read_bytes)


def test_current_wrapper_cannot_relabel_raw_relevant_run_as_complete_suite():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    receipt['scope'] = 'ALL_CURRENT_CPU_REGRESSIONS'
    with pytest.raises(ValueError, match='Current CPU summary/scope differs'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read, read_bytes)


def test_prior_wrapper_failure_identity_cannot_replace_raw_junit_identity():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    receipt['prior_runs'][0]['failure_ids'] = ['invented::failure']
    with pytest.raises(ValueError, match='counts/failure IDs differ from actual JUnit'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read, read_bytes)


def test_original_result_counts_must_equal_raw_junit(tmp_path):
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    def contradictory_result(item):
        value = read(item)
        if item == receipt['prior_runs'][0]['record']:
            value['actual_cases']['failed'] = 2
        return value
    with pytest.raises(ValueError, match='RESULT case totals differ from actual JUnit'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, contradictory_result, read_bytes)


def test_original_exit_zero_cannot_describe_preserved_failure():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    def contradictory_exit(item):
        value = read(item)
        if item == receipt['prior_runs'][0]['record']:
            value['exit_code'] = 0
        return value
    with pytest.raises(ValueError, match='exit status contradicts raw testcase outcomes'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, contradictory_exit, read_bytes)


def test_original_missing_command_cannot_be_invented_by_prior_wrapper():
    candidate, receipt, review, evidence, read, read_bytes = relevant_cpu_fixture()
    def missing_command(item):
        value = read(item)
        if item == receipt['prior_runs'][0]['record']:
            value.pop('command')
        return value
    with pytest.raises(ValueError, match='Original executed CPU command missing/different'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, missing_command, read_bytes)


def test_junit_declared_total_cannot_hide_extra_failure():
    raw = b'<testsuites><testsuite hostname="h100-8" tests="1" failures="0" errors="0" skipped="0"><testcase classname="t" name="f"><failure /></testcase></testsuite></testsuites>'
    with pytest.raises(ValueError, match='suite totals differ from actual testcases'):
        contract.junit_cases(raw)


def strict_static_fixture():
    candidate, _, _, _, _, _ = relevant_cpu_fixture()
    candidate.update(source_version=6, source_file_count=3, source_review_schema='gigpo-e1-fresh200-v6-preexec-review', runtime_review_schema=1,
        runtime_version=5, helpers='/new/helpers', runtime_files=[dict(path='/new/helpers/runtime.py', bytes=20, sha256='4' * 64)])
    candidate['proofs']['runtime_manifest'] = dict(sha256='5' * 64)
    source_files = [dict(relative_path=Path(item['path']).relative_to(candidate['repo']).as_posix(),
                        bytes=item['bytes'], sha256=item['sha256']) for item in candidate['source_files']]
    digest = contract.relative_source_digest(source_files)
    source = dict(schema=candidate['source_review_schema'], status='PASS_STATIC_ONLY_CPU_SCOPE', new_code_commit=candidate['source_commit'],
        source_tracked_files_digest=digest,
        reviewer=dict(root_conversation_inherited=False, independent_after_v6_freeze=True), blocking_findings=[],
        freeze=[dict(sha256='1' * 64, independently_hash_matched=True)], file_receipts=source_files,
        cpu_execution_decision=dict(authorized_by_this_static_review=True, actual_8GPU_acceptance=False, production_READY=False))
    runtime = dict(schema=1, status='PASS_STATIC_ONLY_CPU_SCOPE',
        scope='INDEPENDENT_FROZEN_FRESH200_CALIBRATION_V5_STATIC_PREEXEC_REVIEW_ONLY',
        independence=dict(root_conversation_used_as_acceptance=False, implementation_authorship=False,
            prior_source_or_runtime_PASS_used_as_substitute=False, review_started_from_explicit_standalone_handoff=True),
        decision=dict(blocking_findings=0, whole_runtime_static_pass=True, CPU_test_execution_authorized_by_this_report=True,
            ready=False, GPU_or_Ray_cluster_execution_authorized=False, probability_acceptance=False,
            actual_two_outer_acceptance=False, formal200_authorized=False, formal_iterations_contributed=0),
        frozen_inputs=dict(manifest=dict(sha256='5' * 64, actual_hash_match=True, runtime_file_count=1,
            actual_runtime_hash_matches=1, mismatches=[]), source_snapshot=dict(sha256='1' * 64,
                actual_hash_match=True, source_import_or_execution=False)))
    return candidate, dict(source_review=source, runtime_review=runtime, source_manifest=dict(schema='fresh200-source-static-v6',
        files=source_files, new_code_commit=candidate['source_commit'], source_clean=True, full_tracked_source_files=source_files,
        full_tracked_source_file_count=3, source_tracked_files_digest=digest))


def test_original_independent_reports_grant_only_cpu_scope():
    candidate, proofs = strict_static_fixture(); contract.validate_static_reviews(candidate, proofs)
    assert proofs['runtime_review']['decision']['actual_two_outer_acceptance'] is False


def test_original_failed_runtime_review_cannot_be_relabelled_by_candidate():
    candidate, proofs = strict_static_fixture(); proofs['runtime_review']['status'] = 'FAIL'
    with pytest.raises(ValueError, match='Failed/non-static original runtime review'):
        contract.validate_static_reviews(candidate, proofs)


def test_runtime_review_of_different_source_snapshot_is_rejected():
    candidate, proofs = strict_static_fixture()
    proofs['runtime_review']['frozen_inputs']['source_snapshot']['sha256'] = '6' * 64
    with pytest.raises(ValueError, match='manifest/source snapshot differs'):
        contract.validate_static_reviews(candidate, proofs)


def test_missing_runtime_file_review_cannot_be_hid_by_same_manifest_name():
    candidate, proofs = strict_static_fixture()
    proofs['runtime_review']['frozen_inputs']['manifest']['actual_runtime_hash_matches'] = 0
    with pytest.raises(ValueError, match='manifest/source snapshot differs'):
        contract.validate_static_reviews(candidate, proofs)


def test_static_review_cannot_claim_actual_gpu_acceptance():
    candidate, proofs = strict_static_fixture()
    proofs['runtime_review']['decision']['GPU_or_Ray_cluster_execution_authorized'] = True
    with pytest.raises(ValueError, match='Runtime static decision scope inflated'):
        contract.validate_static_reviews(candidate, proofs)


def test_old_v4_source_cannot_be_relabelled_as_version_six():
    candidate, proofs = strict_static_fixture()
    proofs['source_manifest']['schema'] = 'fresh200-source-static-v4'
    proofs['source_review']['schema'] = candidate['source_review_schema'] = 'gigpo-e1-fresh200-v4-preexec-review'
    with pytest.raises(ValueError, match='old v4 cannot be relabelled'):
        contract.validate_static_reviews(candidate, proofs)


def test_uncovered_additional_tracked_source_is_rejected():
    candidate, proofs = strict_static_fixture()
    candidate['source_files'].append(dict(path=candidate['repo'] + '/unreviewed.py', bytes=1, sha256='0' * 64))
    candidate['source_file_count'] += 1
    with pytest.raises(ValueError, match='closure/digest differs or has uncovered changes'):
        contract.validate_static_reviews(candidate, proofs)


def test_final_commit_must_be_the_original_reviewed_frozen_commit():
    candidate, proofs = strict_static_fixture(); candidate['source_commit'] = '7' * 40
    with pytest.raises(ValueError, match='frozen source commit/clean identity differs'):
        contract.validate_static_reviews(candidate, proofs)


def asset_structure_fixture():
    # Synthetic in-memory structural fixture only. It bypasses JSON reading;
    # real candidate entry points additionally enforce the three original SHAs.
    groups = {}; rows = []
    for group, count in contract.ASSET_GROUPS.items():
        groups[group] = [dict(path=str(contract.SHARED / 'assets-root/assets' / group / str(index)), bytes=1, sha256='a' * 64)
                         for index in range(count)]
        rows.extend(groups[group])
    candidate = dict(asset_files=rows, support_files=[dict(contract.INTERPRETER)],
        proofs={key: dict(path='/pinned/' + key + '.json', bytes=pin[0], sha256=pin[1]) for key, pin in contract.ASSET_PINS.items()})
    manifest = dict(groups=groups, group_counts=dict(contract.ASSET_GROUPS), file_count=39408,
        accepted_asset_provenance_records=[candidate['proofs'][key] for key in ('asset_origin_candidate', 'asset_metadata')],
        historical_native_acceptance_borrowed=False,
        environment_interpreter=dict(contract.INTERPRETER, resolved_path='/usr/bin/python3.10'),
        environment_entry_alias=dict(resolved_path=str(contract.SHARED / 'env')))
    proofs = dict(asset_manifest=manifest, asset_origin_candidate=dict(files=rows),
        asset_metadata=dict(metadata=[dict(row, device=1, inode=index + 1, mtime_ns=1) for index, row in enumerate(rows)]))
    return candidate, proofs


def test_complete_fixed_asset_geometry_and_origins_are_checked():
    candidate, proofs = asset_structure_fixture(); contract.validate_assets(candidate, proofs)


def test_asset_subset_cannot_match_a_subset_of_caller_proofs():
    candidate, proofs = asset_structure_fixture(); candidate['asset_files'] = candidate['asset_files'][1:]
    with pytest.raises(ValueError, match='complete immutable accepted closure'):
        contract.validate_assets(candidate, proofs)


def test_missing_environment_group_is_not_accepted_asset_geometry():
    candidate, proofs = asset_structure_fixture(); proofs['asset_manifest']['groups'].pop('environment')
    with pytest.raises(ValueError, match='Fixed six-group 39408'):
        contract.validate_assets(candidate, proofs)


def test_changed_original_physical_receipt_is_rejected():
    candidate, proofs = asset_structure_fixture(); candidate['proofs']['asset_metadata']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Unpinned original asset/provenance record'):
        contract.validate_assets(candidate, proofs)


def test_substituted_interpreter_is_not_accepted_environment():
    candidate, proofs = asset_structure_fixture(); candidate['support_files'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Substituted accepted environment/interpreter'):
        contract.validate_assets(candidate, proofs)


def test_missing_original_candidate_asset_is_rejected():
    candidate, proofs = asset_structure_fixture(); proofs['asset_origin_candidate']['files'] = proofs['asset_origin_candidate']['files'][1:]
    with pytest.raises(ValueError, match='exact original candidate/physical receipts'):
        contract.validate_assets(candidate, proofs)


def install_submit_fixture(tmp_path, monkeypatch):
    controls = tmp_path / 'controls'; controls.mkdir(); state = tmp_path / 'state'; state.mkdir()
    for name in contract.NAMES:
        (controls / name).write_text('# CPU fixture identity only\n')
    helper = controls / 'submit_once_v2.py'; helper.write_text('# CPU fixture identity only\n')
    ready_path = state / 'READY.json'; ready_path.write_text('{}\n')
    review_path = state / 'review.json'; review_path.write_text('{}\n')
    ready_record, review_record = contract.record(ready_path), contract.record(review_path)
    proof = dict(status='PASS', scope='FRESH200_CALIBRATION2_SINGLE_SUBMIT_REVIEW', independent_review=True,
        ready_sha256=ready_record['sha256'], submit_helper_sha256=contract.sha(helper), controls_manifest_sha256='a' * 64,
        effective_submission_environment=dict(submission.SAFE_ENVIRONMENT), sbatch_options=list(submission.SBATCH_OPTIONS))
    physical_record = dict(path=str(state / 'physical.json'), bytes=1, sha256='b' * 64)
    candidate = dict(repo='/exact/new/repo', helpers='/exact/new/fresh200-calibration-v4',
        environment_root='/exact/old/readonly/env', source_commit='c' * 40, asset_files=[],
        control_files=[contract.record(controls / name) for name in contract.NAMES],
        proofs=dict(controls_manifest=dict(sha256='a' * 64)))
    ready = dict(files=[], evidence=[], physical_record=physical_record)
    def fake_read(item):
        if item == ready_record: return ready
        if item == review_record: return proof
        if item == physical_record: return dict(metadata=[])
        raise AssertionError('Unknown fixture proof')
    monkeypatch.setattr(submission, 'CONTROLS', controls); monkeypatch.setattr(submission, '__file__', str(helper))
    monkeypatch.setattr(contract, 'CONTROLS', controls)
    monkeypatch.setattr(submission, 'READY', ready_path)
    for name in ('BINDING', 'INTENT', 'RECEIPT'):
        monkeypatch.setattr(submission, name, state / (name + '.json'))
    monkeypatch.setattr(submission, 'read_bound', fake_read)
    monkeypatch.setattr(submission, 'validate_ready', lambda value: candidate)
    for key in list(submission.os.environ):
        if key.startswith(('SBATCH_', 'SLURM_', 'SALLOC_', 'SRUN_', 'GIGPO_QUALITY_')): monkeypatch.delenv(key)
    return ready_record, review_record, candidate


def test_exactly_one_submission_has_five_runtime_arguments(tmp_path, monkeypatch):
    ready, review, candidate = install_submit_fixture(tmp_path, monkeypatch); calls = []
    def fake_run(command, **kwargs):
        assert kwargs['env'] == submission.SAFE_ENVIRONMENT
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '12345;cluster\n' if command[0] == 'sbatch' else '', '')
    result = submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    command = calls[-1]
    header = candidate['helpers'] + '/calibration_v1.sbatch'; position = command.index(header)
    assert command[:position] == ['sbatch', *submission.SBATCH_OPTIONS]
    assert len(command[position + 1:]) == 5 and command[-1] == candidate['environment_root']
    assert result['status'] == 'SUBMITTED_EXACTLY_ONCE' and result['actual_gpu_acceptance'] is False
    with pytest.raises(ValueError, match='Existing claim/uncertainty'):
        submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    assert sum(command[0] == 'sbatch' for command in calls) == 1


def test_unknown_sbatch_output_keeps_durable_intent_and_blocks_retry(tmp_path, monkeypatch):
    ready, review, _ = install_submit_fixture(tmp_path, monkeypatch); calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '12345\n12346\n' if command[0] == 'sbatch' else '', '')
    with pytest.raises(ValueError, match='Unknown sbatch outcome'):
        submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    assert submission.INTENT.is_file() and submission.BINDING.is_file()
    assert json.loads(submission.RECEIPT.read_text())['status'] == 'UNCERTAIN_NO_RESUBMISSION'
    with pytest.raises(ValueError, match='Existing claim/uncertainty'):
        submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    assert sum(command[0] == 'sbatch' for command in calls) == 1


def test_sbatch_timeout_preserves_uncertainty_without_second_call(tmp_path, monkeypatch):
    ready, review, _ = install_submit_fixture(tmp_path, monkeypatch); calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == 'sbatch': raise subprocess.TimeoutExpired(command, 60, output='123')
        return subprocess.CompletedProcess(command, 0, '', '')
    with pytest.raises(subprocess.TimeoutExpired):
        submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    assert json.loads(submission.RECEIPT.read_text())['status'] == 'UNCERTAIN_NO_RESUBMISSION'
    with pytest.raises(ValueError, match='Existing claim/uncertainty'):
        submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    assert sum(command[0] == 'sbatch' for command in calls) == 1


def test_occupied_node_creates_no_submission_claim(tmp_path, monkeypatch):
    ready, review, _ = install_submit_fixture(tmp_path, monkeypatch); calls = []
    def fake_run(command, **kwargs):
        calls.append(command); return subprocess.CompletedProcess(command, 0, 'occupied job\n', '')
    with pytest.raises(ValueError, match='Node occupied/unreadable'):
        submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    assert not submission.INTENT.exists() and not submission.BINDING.exists() and not submission.RECEIPT.exists()
    assert not any(command[0] == 'sbatch' for command in calls)


def test_changed_installed_controller_prevents_any_queue_or_submit_call(tmp_path, monkeypatch):
    ready, review, candidate = install_submit_fixture(tmp_path, monkeypatch)
    Path(candidate['control_files'][0]['path']).write_text('# changed after review\n')
    def no_run(*args, **kwargs):
        raise AssertionError('Changed controls must fail before external commands')
    with pytest.raises(ValueError, match='Different installed control bytes'):
        submission.submit(ready, review, run=no_run, hostname=lambda: 'h100-8')
    assert not submission.INTENT.exists() and not submission.BINDING.exists() and not submission.RECEIPT.exists()


@pytest.mark.parametrize('name,value', [('SBATCH_ARRAY_INX', '0-1'), ('SBATCH_CLUSTERS', 'foreign'),
    ('SLURM_CLUSTERS', 'foreign'), ('SBATCH_GRES', 'gpu:a800:1'), ('SBATCH_EXCLUSIVE', ''),
    ('SBATCH_REQUEUE', 'yes'), ('SBATCH_EXPORT', 'ALL'), ('SBATCH_PARTITION', 'other'),
    ('SLURM_CONF', '/unreviewed/slurm.conf'), ('SRUN_OPTIONS', '--array=0-1')])
def test_scheduler_input_overrides_create_no_claim_or_external_call(tmp_path, monkeypatch, name, value):
    ready, review, _ = install_submit_fixture(tmp_path, monkeypatch); monkeypatch.setenv(name, value)
    def no_run(*args, **kwargs):
        raise AssertionError('Inherited scheduler inputs must fail before external calls')
    with pytest.raises(ValueError, match='Inherited scheduler/old authorization inputs forbidden'):
        submission.submit(ready, review, run=no_run, hostname=lambda: 'h100-8')
    assert not submission.INTENT.exists() and not submission.BINDING.exists() and not submission.RECEIPT.exists()


def test_submission_environment_never_passes_unrelated_loader_or_proxy_values():
    result = submission.scheduler_environment(dict(PATH='/unreviewed/bin', LD_PRELOAD='/unreviewed.so',
        BASH_ENV='/unreviewed.sh', ALL_PROXY='secret-proxy-value', PYTHONPATH='/old/native'))
    assert result == submission.SAFE_ENVIRONMENT and set(result) == {'PATH', 'LANG', 'LC_ALL', 'PYTHONDONTWRITEBYTECODE'}


def test_single_submit_review_is_reaudited_after_queue_observations(tmp_path, monkeypatch):
    ready, review, _ = install_submit_fixture(tmp_path, monkeypatch)
    original = submission.read_bound; observed = []
    def changed_review(item):
        value = copy.deepcopy(original(item))
        if observed and item == review:
            value['sbatch_options'] = ['--array=0-1']
        return value
    monkeypatch.setattr(submission, 'read_bound', changed_review)
    def fake_run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, '', '')
    with pytest.raises(ValueError, match='READY/single-submit review changed before intent'):
        submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    assert not submission.INTENT.exists() and not any(command[0] == 'sbatch' for command in observed)
