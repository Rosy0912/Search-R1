"""Datarow closure refusals only; no remote, CUDA, native replay or installation."""
import copy
import json
from pathlib import Path
import sys

import pytest
import deployment_contract_v8 as contract
import prepare_candidate_v8 as prepare

sys.path.append(str(Path(__file__).resolve().parent.with_name('safety-calibration-deployment-v7')))
import test_deployment_v7 as old_fixtures


def candidate8():
    c = old_fixtures.candidate7()
    old_paths = {r['path'] for r in c['runtime_helpers']}
    c['files'] = [r for r in c['files'] if r['path'] not in old_paths]
    c['runtime_helpers'] = [dict(r, path=str(contract.HELPERS / Path(r['path']).name)) for r in c['runtime_helpers']]
    c['files'] += c['runtime_helpers']
    c.update(helpers=str(contract.HELPERS), controls=str(contract.CONTROLS),
             factory_runtime_changed_filenames=copy.deepcopy(contract.FACTORY_CHANGED),
             datarow_change_scope=contract.DATAROW_SCOPE, runtime_changed_filenames=copy.deepcopy(contract.DATAROW_CHANGED),
             datarow_base_runtime_manifest_sha256=contract.DATAROW_BASE_MANIFEST_SHA)
    c['datarow_control_proofs'] = {role: dict(path=str(contract.ROOT / 'logs' / ('unit-datarow-' + role + '.json')),
            bytes=1, sha256='c' * 64) for role in contract.DATAROW_PROOF_ROLES}
    c['evidence'] += list(c['datarow_control_proofs'].values())
    return c


@pytest.mark.parametrize('mutation', ['scope', 'base', 'changed_math', 'factory_history', 'missing_factory',
    'missing_datarow', 'duplicate_datarow', 'unbound_datarow', 'outside_datarow', 'then_update',
    'production_threshold', 'premature_native'])
def test_new_dtype_scope_and_preserved_factory_fail_closed(mutation):
    c = candidate8()
    if mutation == 'scope': c['datarow_change_scope'] = 'algorithm-change'
    elif mutation == 'base': c['datarow_base_runtime_manifest_sha256'] = 'f' * 64
    elif mutation == 'changed_math': c['runtime_changed_filenames'].append('calibration_worker_v1.py')
    elif mutation == 'factory_history': c['factory_runtime_changed_filenames'] = ['calibration_driver_v1.py']
    elif mutation == 'missing_factory': c['factory_control_proofs'].pop('runtime_review')
    elif mutation == 'missing_datarow': c['datarow_control_proofs'].pop('runtime_review')
    elif mutation == 'duplicate_datarow': c['datarow_control_proofs']['runtime_review'] = copy.deepcopy(c['datarow_control_proofs']['runtime_cpu'])
    elif mutation == 'unbound_datarow': c['evidence'].pop()
    elif mutation == 'outside_datarow': c['datarow_control_proofs']['runtime_review']['path'] = '/tmp/other.json'
    elif mutation == 'then_update': c['calibration_phase'] = 'probability_probe_then_update'
    elif mutation == 'production_threshold': c['target_old_policy_kl'] = .01
    else: c['source_native_acceptance'] = {'status': 'PASS'}
    with pytest.raises((ValueError, KeyError)): contract.validate_candidate(c)


def test_isolated_dtype_candidate_keeps_factory_startup_and_null_native():
    c = candidate8(); contract.validate_candidate(c)
    assert len(c['runtime_helpers']) == 14 and c['source_native_acceptance'] is None
    assert c['calibration_phase'] == 'probability_probe_then_wait'
    assert c['factory_control_proofs'] == old_fixtures.candidate7()['factory_control_proofs']
    assert c['startup_control_proofs'] == old_fixtures.candidate7()['startup_control_proofs']


@pytest.mark.parametrize('mutation', ['manifest', 'diff', 'cpu', 'scope', 'extra_file', 'gpu', 'source', 'threshold', 'not_pass'])
def test_runtime_review_is_bound_static_dtype_scope(mutation):
    proofs = {role: dict(sha256=str(i) * 64) for i, role in enumerate(contract.DATAROW_PROOF_ROLES)}
    r = dict(status='PASS', source_commit=contract.SOURCE,
        runtime_manifest_sha256=proofs['runtime_manifest']['sha256'], runtime_diff_sha256=proofs['runtime_diff']['sha256'],
        runtime_cpu_sha256=proofs['runtime_cpu']['sha256'], datarow_change_scope=contract.DATAROW_SCOPE,
        runtime_changed_filenames=copy.deepcopy(contract.DATAROW_CHANGED), gpu_used=False,
        source_modified=False, production_thresholds_selected=False)
    if mutation in ('manifest', 'diff', 'cpu'): r['runtime_' + mutation + '_sha256'] = 'f' * 64
    elif mutation == 'scope': r['datarow_change_scope'] = 'algorithm-change'
    elif mutation == 'extra_file': r['runtime_changed_filenames'].append('calibration_worker_v1.py')
    elif mutation == 'gpu': r['gpu_used'] = True
    elif mutation == 'source': r['source_modified'] = True
    elif mutation == 'threshold': r['production_thresholds_selected'] = True
    else: r['status'] = 'NEEDS_FIX'
    with pytest.raises((ValueError, KeyError)): prepare.validate_runtime_review(proofs, r, contract.DATAROW_CHANGED)


def failure_inputs(tmp, mutation=None):
    names = dict(failure_payload='CALIBRATION3281-FAILURE-PAYLOAD-v2.json',
        failure_review='CALIBRATION3281-FAILURE-INDEPENDENT-REVIEW-v1.json',
        failure_sampling_metadata='PROBABILITY3281-FAILURE-RAW-METADATA-20261005T235846704624Z.json',
        failure_trace_metadata='PROBABILITY3281-TRACE-RAW-METADATA-20261006T000029859946Z.json')
    values = {role: json.loads((prepare.OUT / name).read_text()) for role, name in names.items()}
    p = values['failure_payload']; r = values['failure_review']; s = values['failure_sampling_metadata']['actual']
    t = values['failure_trace_metadata']['actual']
    if mutation == 'adam': r['actual_adam_writes'] = 1
    elif mutation == 'bool_adam': r['actual_adam_writes'] = False
    elif mutation == 'gradient': r['actual_gradient_updates'] = 1
    elif mutation == 'accepted': r['status'] = 'PASS_CALIBRATION'
    elif mutation == 'binding': r['binding_sha256'] = 'f' * 64
    elif mutation == 'cgroup': p['cgroup'][0]['exists'] = True
    elif mutation == 'wrong_audit': p['audit_paths'][0]['path'] = p['run'] + '/audit/rank-0.jsonl'
    elif mutation == 'corrupt_raw': p['originals'][0]['text'] += 'corrupt'
    elif mutation == 'missing_raw': p['originals'].pop()
    elif mutation == 'missing_worker': s['worker_receipt_diagnostics'].pop()
    elif mutation == 'bad_pt_sha': s['worker_receipt_diagnostics'][0]['raw_declared_sha_matches'] = False
    elif mutation == 'missing_source': s['training_prefix_original']['source_set_is_exactly_0_2559'] = False
    elif mutation == 'wrong_group': t['trace_originals'][0]['question_response_group_mapping_valid'] = False
    paths = {role: tmp / (role + '.json') for role in names}
    for role, value in values.items():
        if role != 'failure_review': paths[role].write_text(json.dumps(value))
    r['evidence'][0] = dict(path=str(paths['failure_payload']), sha256=contract.sha(paths['failure_payload']))
    paths['failure_review'].write_text(json.dumps(r))
    proofs = {role: dict(path=str(path), bytes=path.stat().st_size, sha256=contract.sha(path)) for role, path in paths.items()}
    return proofs, paths


@pytest.mark.parametrize('mutation', ['adam', 'bool_adam', 'gradient', 'accepted', 'binding', 'cgroup',
    'wrong_audit', 'corrupt_raw', 'missing_raw', 'missing_worker', 'bad_pt_sha', 'missing_source', 'wrong_group'])
def test_actual_failed3281_identity_zero_update_sampling_release_refuse(tmp_path, mutation):
    with pytest.raises((ValueError, KeyError)): prepare.validate_failure_inputs(*failure_inputs(tmp_path, mutation))


def test_original_failed3281_sampling_metadata_is_preserved_not_probability_acceptance(tmp_path):
    originals, sampled = prepare.validate_failure_inputs(*failure_inputs(tmp_path))
    assert len(originals) == 27 and len(sampled) == 69
    assert sum(x['bytes'] for x in sampled if x['path'].endswith('.pt')) == 348010688


def actual_proofs():
    names = dict(runtime_manifest='CALIBRATION-RUNTIME-IMPLEMENTATION-v8.json',
        runtime_diff='CALIBRATION-RUNTIME-DIFF-v8.patch', runtime_cpu='CALIBRATION-DATAROW-CPU-v8.json',
        runtime_review='CALIBRATION-DATAROW-INDEPENDENT-REVIEW-v8.json',
        failure_payload='CALIBRATION3281-FAILURE-PAYLOAD-v2.json', failure_review='CALIBRATION3281-FAILURE-INDEPENDENT-REVIEW-v1.json',
        failure_sampling_metadata='PROBABILITY3281-FAILURE-RAW-METADATA-20261005T235846704624Z.json',
        failure_trace_metadata='PROBABILITY3281-TRACE-RAW-METADATA-20261006T000029859946Z.json',
        runtime_target_cpu='CALIBRATION-DATAROW-TARGET-CPU-v8.json',
        runtime_target_cpu_raw='CALIBRATION-DATAROW-TARGET-CPU-RAW-v8.json',
        runtime_target_cpu_review='CALIBRATION-DATAROW-TARGET-CPU-INDEPENDENT-REVIEW-v8.json')
    return {role: dict(path=str(prepare.OUT / name), bytes=(prepare.OUT / name).stat().st_size,
                      sha256=contract.sha(prepare.OUT / name)) for role, name in names.items()}


def target_inputs(tmp, mutation=None):
    proofs = actual_proofs()
    roles = ('runtime_target_cpu', 'runtime_target_cpu_raw', 'runtime_target_cpu_review')
    values = {role: json.loads(Path(proofs[role]['path']).read_text()) for role in roles}
    c = values['runtime_target_cpu']; w = values['runtime_target_cpu_raw']; r = values['runtime_target_cpu_review']
    if mutation == 'torch': c['torch_version'] = '2.8.0'
    elif mutation == 'cuda': c['actual_CUDA_initialized'] = True
    elif mutation == 'bool_adam': c['actual_Adam_writes'] = False
    elif mutation == 'wrong_source': c['actual_protocol_path'] = '/tmp/other/protocol.py'
    elif mutation == 'wrong_driver': c['runtime_driver_sha256'] = 'f' * 64
    elif mutation == 'raw_failed': w['actual']['returncode'] = 1
    elif mutation == 'wrong_manifest': r['runtime_manifest_sha256'] = 'f' * 64
    elif mutation == 'review_gpu': r['gpu_used'] = True
    elif mutation == 'review_threshold': r['production_thresholds_selected'] = True
    elif mutation == 'review_node': r['node'] = 'h100-11'
    paths = {role: tmp / (role + '.json') for role in roles}
    paths['runtime_target_cpu'].write_text(json.dumps(c))
    proofs['runtime_target_cpu'] = dict(path=str(paths['runtime_target_cpu']), bytes=paths['runtime_target_cpu'].stat().st_size,
                                       sha256=contract.sha(paths['runtime_target_cpu']))
    w['actual']['stdout'] = json.dumps(c)
    w['actual']['records']['target_cpu'].update(data=c, bytes=proofs['runtime_target_cpu']['bytes'], sha256=proofs['runtime_target_cpu']['sha256'])
    paths['runtime_target_cpu_raw'].write_text(json.dumps(w))
    proofs['runtime_target_cpu_raw'] = dict(path=str(paths['runtime_target_cpu_raw']), bytes=paths['runtime_target_cpu_raw'].stat().st_size,
                                           sha256=contract.sha(paths['runtime_target_cpu_raw']))
    r.update(runtime_target_cpu_sha256=proofs['runtime_target_cpu']['sha256'],
             runtime_target_cpu_raw_sha256=proofs['runtime_target_cpu_raw']['sha256'])
    paths['runtime_target_cpu_review'].write_text(json.dumps(r))
    return proofs, paths


@pytest.mark.parametrize('mutation', ['torch', 'cuda', 'bool_adam', 'wrong_source', 'wrong_driver',
    'raw_failed', 'wrong_manifest', 'review_gpu', 'review_threshold', 'review_node'])
def test_target_cpu_original_and_independent_static_identity_refuse(tmp_path, mutation):
    with pytest.raises((ValueError, KeyError)):
        prepare.validate_target_cpu_inputs(*target_inputs(tmp_path, mutation), '5e4d8d1d4bdd05dcf9c71d4a069a898dbd458151e2951845c2101cb8d7df50dd')


def test_actual_runtime_dtype_only_and_both_local_target_cpu_static_reviews_close():
    proofs = actual_proofs()
    manifest, paths = prepare.validate_runtime_inputs(proofs)
    assert len(manifest['files']) == 14 and manifest['runtime_changed_filenames'] == ['calibration_driver_v1.py']
    assert manifest['scope']['actual_GPU_acceptance'] is False


def test_inherited_factory_cpu_probe_uses_exact_frozen_work_original():
    record=dict(path=str(contract.ROOT/'logs/test_metadata_colocation_v7_actual.py'), bytes=8483,
        sha256='f283fc30e671ec3740bcf3656d36e5417c22704613de84f3799c7d0a06afc9d1')
    assert prepare.inherited_local_proof(record).is_file()
    with pytest.raises(ValueError): prepare.inherited_local_proof(dict(record, sha256='f'*64))
    with pytest.raises(ValueError): prepare.inherited_local_proof(dict(record, path=str(contract.ROOT/'logs/unknown.py')))
