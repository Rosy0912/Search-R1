"""CPU fixtures only, currently unexecuted. Never invoke real Slurm or source."""
import copy
import json
import subprocess
from pathlib import Path
import pytest
import deployment_contract_v1 as contract
import submit_once_v1 as submission


def test_frozen_ff7_observation_cannot_become_final_candidate():
    with pytest.raises(ValueError, match='Historical/pending candidate'):
        contract.validate_candidate(dict(status='CANDIDATE_NOT_READY_FINAL_RUNTIME_DEPENDENCY_PENDING'))


def test_bound_original_rejects_same_length_different_bytes(tmp_path):
    path = tmp_path / 'original.json'; path.write_text('{"value":1}')
    item = contract.record(path); path.write_text('{"value":2}')
    with pytest.raises(ValueError, match='Changed bound JSON'):
        contract.read_bound(item)


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


def relevant_cpu_fixture():
    repo = '/new/repo'
    changed = dict(relative_path='search_r1/baselines/instruct300.py', bytes=12, sha256='a' * 64)
    test = dict(relative_path='tests/test_checkpoint_controls.py', bytes=13, sha256='b' * 64)
    unchanged = dict(relative_path='search_r1/baselines/policy_safety.py', bytes=14, sha256='c' * 64)
    def source(item):
        return dict(path=repo + '/' + item['relative_path'], bytes=item['bytes'], sha256=item['sha256'])
    raw = dict(path='/proofs/current-cpu.log', bytes=15, sha256='d' * 64)
    prior = dict(path='/proofs/preserved-175-RESULT.json', bytes=16, sha256='e' * 64)
    patch = dict(path='/proofs/minimal-semantic.patch', bytes=17, sha256='f' * 64)
    old_manifest = dict(path='/proofs/actual-source-v3-identity.json', bytes=18, sha256='3' * 64)
    candidate = dict(repo=repo, source_commit='9' * 40, source_files=[source(x) for x in (changed, test, unchanged)],
        proofs=dict(source_manifest=dict(sha256='1' * 64), source_cpu=dict(sha256='2' * 64)))
    receipt = dict(status='PASS', scope='RELEVANT_CHANGED_CPU_REGRESSIONS', actual_execution=True,
        gpu_used=False, exit_code=0, failed=0, passed=41, source_commit=candidate['source_commit'],
        manifest_sha256='1' * 64, raw_execution_records=[raw, prior], current_source_and_test_files=[changed, test],
        claims_all_prior_cases_rerun_on_current_commit=False, minimal_patch_record=patch,
        unchanged_source_files=[unchanged], prior_runs=[dict(record=prior, source_commit='8' * 40, passed=174, failed=1,
            source_manifest_record=old_manifest, unchanged_source_files=[unchanged],
            superseded_failure_ids=['full_hf_semantic_config_identity'])],
        superseded_prior_failure_ids=['full_hf_semantic_config_identity'])
    review = dict(status='PASS', scope='FRESH200_SOURCE_CPU_EVIDENCE_ONLY', source_commit=candidate['source_commit'],
        manifest_sha256='1' * 64, cpu_receipt_sha256='2' * 64, independent_review=True,
        complete_required_cpu_coverage_accepted=True,
        superseded_prior_failure_ids=receipt['superseded_prior_failure_ids'],
        unchanged_source_files_digest=contract.records_digest([source(unchanged)]))
    def fake_read(item):
        if item == prior:
            return dict(code_commit='8' * 40, gpu_acceptance=False, production_ready=False)
        if item == old_manifest:
            return dict(source_commit='8' * 40, repo='/historical/repo', files=[dict(
                path='/historical/repo/' + unchanged['relative_path'], bytes=14, sha256='c' * 64)])
        raise AssertionError('Unknown historical CPU fixture record')
    return candidate, receipt, review, contract.closure([raw, prior, patch, old_manifest], 'CPU evidence'), fake_read


def test_changed_cpu_plus_preserved_174_pass_one_superseded_fail_is_explicit():
    candidate, receipt, review, evidence, read = relevant_cpu_fixture()
    contract.validate_cpu(receipt, review, 'source', candidate, evidence, read)
    assert receipt['prior_runs'][0]['failed'] == 1
    assert receipt['claims_all_prior_cases_rerun_on_current_commit'] is False


def test_unchanged_source_sha_drift_invalidates_prior_cpu():
    candidate, receipt, review, evidence, read = relevant_cpu_fixture()
    receipt['unchanged_source_files'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Covered source changed'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read)


def test_failed_current_semantic_loader_never_accepts_prior_cpu():
    candidate, receipt, review, evidence, read = relevant_cpu_fixture(); receipt['failed'] = 1
    with pytest.raises(ValueError, match='Actual successful CPU'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read)


def test_prior_174_cannot_be_reported_as_current_commit_rerun():
    candidate, receipt, review, evidence, read = relevant_cpu_fixture()
    receipt['claims_all_prior_cases_rerun_on_current_commit'] = True
    with pytest.raises(ValueError, match='Prior cases cannot be called rerun'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read)


def test_prior_original_result_commit_cannot_be_relabelled():
    candidate, receipt, review, evidence, read = relevant_cpu_fixture()
    original_read = read
    def different_commit(item):
        result = original_read(item)
        if item == receipt['prior_runs'][0]['record']:
            result['code_commit'] = '7' * 40
        return result
    with pytest.raises(ValueError, match='Preserved actual CPU/source identity differs'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, different_commit)


def test_unexecuted_cpu_fixture_status_cannot_be_promoted():
    candidate, receipt, review, evidence, read = relevant_cpu_fixture(); receipt['actual_execution'] = False
    with pytest.raises(ValueError, match='Actual successful CPU evidence missing'):
        contract.validate_cpu(receipt, review, 'source', candidate, evidence, read)


def strict_static_fixture():
    candidate, _, _, _, _ = relevant_cpu_fixture()
    candidate.update(source_review_schema='gigpo-e1-fresh200-v5-preexec-review', runtime_review_schema=1,
        runtime_version=5, helpers='/new/helpers', runtime_files=[dict(path='/new/helpers/runtime.py', bytes=20, sha256='4' * 64)])
    candidate['proofs']['runtime_manifest'] = dict(sha256='5' * 64)
    source_files = [dict(relative_path=Path(item['path']).relative_to(candidate['repo']).as_posix(),
                        bytes=item['bytes'], sha256=item['sha256']) for item in candidate['source_files']]
    source = dict(schema=candidate['source_review_schema'], status='PASS_STATIC_ONLY_CPU_SCOPE',
        reviewer=dict(root_conversation_inherited=False, independent_after_v5_freeze=True), blocking_findings=[],
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
    return candidate, dict(source_review=source, runtime_review=runtime, source_manifest=dict(files=source_files))


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


def install_submit_fixture(tmp_path, monkeypatch):
    controls = tmp_path / 'controls'; controls.mkdir(); state = tmp_path / 'state'; state.mkdir()
    for name in contract.NAMES:
        (controls / name).write_text('# CPU fixture identity only\n')
    helper = controls / 'submit_once_v1.py'; helper.write_text('# CPU fixture identity only\n')
    ready_path = state / 'READY.json'; ready_path.write_text('{}\n')
    review_path = state / 'review.json'; review_path.write_text('{}\n')
    ready_record, review_record = contract.record(ready_path), contract.record(review_path)
    proof = dict(status='PASS', scope='FRESH200_CALIBRATION2_SINGLE_SUBMIT_REVIEW', independent_review=True,
        ready_sha256=ready_record['sha256'], submit_helper_sha256=contract.sha(helper), controls_manifest_sha256='a' * 64)
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
        if key.startswith('GIGPO_QUALITY_'): monkeypatch.delenv(key)
    return ready_record, review_record, candidate


def test_exactly_one_submission_has_five_runtime_arguments(tmp_path, monkeypatch):
    ready, review, candidate = install_submit_fixture(tmp_path, monkeypatch); calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '12345;cluster\n' if command[0] == 'sbatch' else '', '')
    result = submission.submit(ready, review, run=fake_run, hostname=lambda: 'h100-8')
    command = calls[-1]
    assert command[:3] == ['sbatch', '--parsable', candidate['helpers'] + '/calibration_v1.sbatch']
    assert len(command[3:]) == 5 and command[-1] == candidate['environment_root']
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
