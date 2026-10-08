"""Datarow closure refusals only; no remote, CUDA, native replay or installation."""
import copy
import json
from pathlib import Path
import sys

import pytest
import deployment_contract_v9 as contract
import prepare_candidate_v9 as prepare

sys.path.append(str(Path(__file__).resolve().parent.with_name('safety-calibration-deployment-v7')))
import test_deployment_v7 as old_fixtures


def inherited_candidate8():
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


def candidate9():
    c=inherited_candidate8()
    old={row['path'] for row in c['runtime_helpers']}
    c['files']=[row for row in c['files'] if row['path'] not in old]
    c['runtime_helpers']=[dict(path=str(contract.HELPERS/Path(row['path']).name),bytes=row['bytes'],sha256=row['sha256'])
        for row in json.loads((prepare.OUT/'CALIBRATION-RUNTIME-IMPLEMENTATION-v9.json').read_text())['files']]
    c['files']+=c['runtime_helpers']
    c['inherited_v8_wrapper_sha256']=c.pop('unchanged_runtime_wrapper_sha256')
    c.update(predecessor_calibration_candidate_sha256=contract.PREDECESSOR_CANDIDATE_SHA,
        preserved_v8_ready_binding_sha256=contract.PREDECESSOR_READY_SHA,
        backend_base_runtime_manifest_sha256=contract.BACKEND_BASE_MANIFEST_SHA,
        historical_runtime_proof_versions=dict(startup='v6-v3',factory='v7',datarow='v8'),
        datarow_runtime_changed_filenames=copy.deepcopy(contract.DATAROW_CHANGED),
        runtime_changed_filenames=copy.deepcopy(contract.BACKEND_CHANGED),backend_change_scope=contract.BACKEND_SCOPE,
        backend_unchanged_runtime_files=11,expected_rollout_attention_backend='FLASH_ATTN',
        actual_backend_gate_required_before_restore_baseline_adam=True,
        actual_v9_runtime_wrapper_sha256=c['runtime_helpers'][0]['sha256'],preserved_v8_one_outer_ast_sha256='a'*64)
    for row in c['runtime_helpers']:
        if Path(row['path']).name=='metadata_colocation_v1.py':row['sha256']='39e3226c82132524e57ee1adc964f607da6a82314a784c1081c505fd10ea6b9e'
        if Path(row['path']).name=='calibration_run_v1.py':c['actual_v9_runtime_wrapper_sha256']=row['sha256']
    c['backend_control_proofs']={role:dict(path=str(contract.ROOT/'logs'/('unit-backend-'+role+'.json')),bytes=1,sha256='d'*64)
        for role in contract.BACKEND_PROOF_ROLES}
    c['backend_control_proofs']['runtime_manifest']['path']=str(contract.ROOT/'logs/CALIBRATION-RUNTIME-IMPLEMENTATION-v9.json')
    c['current_driver_target_cpu_proofs']={role:dict(path=str(contract.ROOT/'logs'/name),bytes=1,sha256='e'*64)
        for role,name in zip(contract.CURRENT_DRIVER_CPU_ROLES,('CALIBRATION-DATAROW-TARGET-CPU-v9.json',
            'CALIBRATION-DATAROW-TARGET-CPU-RAW-v9.json','CALIBRATION-DATAROW-TARGET-CPU-INDEPENDENT-REVIEW-v9.json'))}
    c['predecessor_candidate_record']=dict(path=str(contract.ROOT/'logs/parentcandidate.json'),bytes=1,sha256=contract.PREDECESSOR_CANDIDATE_SHA)
    c['predecessor_ready_record']=dict(path=str(contract.ROOT/'logs/parentready.json'),bytes=1,sha256=contract.PREDECESSOR_READY_SHA)
    c['evidence']+=list(c['backend_control_proofs'].values())+list(c['current_driver_target_cpu_proofs'].values())+[c['predecessor_candidate_record'],c['predecessor_ready_record']]
    return c


@pytest.mark.parametrize('mutation', ['scope', 'base', 'changed_math', 'factory_history', 'missing_factory',
    'missing_datarow', 'duplicate_datarow', 'unbound_datarow', 'outside_datarow', 'then_update',
    'production_threshold', 'premature_native'])
def test_new_dtype_scope_and_preserved_factory_fail_closed(mutation):
    c = candidate9()
    if mutation == 'scope': c['datarow_change_scope'] = 'algorithm-change'
    elif mutation == 'base': c['datarow_base_runtime_manifest_sha256'] = 'f' * 64
    elif mutation == 'changed_math': c['runtime_changed_filenames'].append('other.py')
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
    c = candidate9(); contract.validate_candidate(c)
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
    with pytest.raises((ValueError, KeyError)): prepare.validate_inherited_runtime_review(proofs, r, contract.DATAROW_CHANGED)


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
        prepare.validate_inherited_target_cpu_inputs(*target_inputs(tmp_path, mutation), '5e4d8d1d4bdd05dcf9c71d4a069a898dbd458151e2951845c2101cb8d7df50dd')


def test_actual_runtime_dtype_only_and_both_local_target_cpu_static_reviews_close():
    proofs = actual_proofs()
    manifest, paths = prepare.validate_inherited_datarow_inputs(proofs)
    assert len(manifest['files']) == 14 and manifest['runtime_changed_filenames'] == ['calibration_driver_v1.py']
    assert manifest['scope']['actual_GPU_acceptance'] is False


def test_inherited_factory_cpu_probe_uses_exact_frozen_work_original():
    record=dict(path=str(contract.ROOT/'logs/test_metadata_colocation_v7_actual.py'), bytes=8483,
        sha256='f283fc30e671ec3740bcf3656d36e5417c22704613de84f3799c7d0a06afc9d1')
    assert prepare.inherited_local_proof(record).is_file()
    with pytest.raises(ValueError): prepare.inherited_local_proof(dict(record, sha256='f'*64))
    with pytest.raises(ValueError): prepare.inherited_local_proof(dict(record, path=str(contract.ROOT/'logs/unknown.py')))


@pytest.mark.parametrize('mutation',['backend','missing_backend','dup_backend','new_cpu_missing','old_cpu_as_new',
    'unbound_new_cpu','wrapper','factory','parent','ready','base','history','changed','no_gate','no_ast'])
def test_new_backend_and_current_driver_cpu_admission_reject(mutation):
    c=candidate9()
    if mutation=='backend':c['expected_rollout_attention_backend']='XFORMERS'
    elif mutation=='missing_backend':c['backend_control_proofs'].pop('runtime_review')
    elif mutation=='dup_backend':c['backend_control_proofs']['runtime_review']=c['backend_control_proofs']['runtime_cpu']
    elif mutation=='new_cpu_missing':c['current_driver_target_cpu_proofs'].pop('runtime_target_cpu_review')
    elif mutation=='old_cpu_as_new':c['current_driver_target_cpu_proofs']['runtime_target_cpu']=c['datarow_control_proofs']['runtime_target_cpu']
    elif mutation=='unbound_new_cpu':c['evidence']=[row for row in c['evidence'] if row!=c['current_driver_target_cpu_proofs']['runtime_target_cpu']]
    elif mutation=='wrapper':c['actual_v9_runtime_wrapper_sha256']='f'*64
    elif mutation=='factory':next(row for row in c['runtime_helpers'] if Path(row['path']).name=='metadata_colocation_v1.py')['sha256']='f'*64
    elif mutation=='parent':c['predecessor_calibration_candidate_sha256']='f'*64
    elif mutation=='ready':c['preserved_v8_ready_binding_sha256']='f'*64
    elif mutation=='base':c['backend_base_runtime_manifest_sha256']='f'*64
    elif mutation=='history':c['historical_runtime_proof_versions']['datarow']='v9'
    elif mutation=='changed':c['runtime_changed_filenames'].append('metadata_colocation_v1.py')
    elif mutation=='no_gate':c['actual_backend_gate_required_before_restore_baseline_adam']=False
    else:c['preserved_v8_one_outer_ast_sha256']=None
    with pytest.raises((ValueError,KeyError)):contract.validate_candidate(c)


def backend_proofs():
    names=dict(runtime_manifest='CALIBRATION-RUNTIME-IMPLEMENTATION-v9.json',runtime_diff='CALIBRATION-RUNTIME-DIFF-v9-v2.patch',
        runtime_cpu='CALIBRATION-BACKEND-CPU-v9-v2.log',runtime_review='CALIBRATION-BACKEND-INDEPENDENT-REVIEW-v9.json',
        runtime_target_cpu_import='CALIBRATION-BACKEND-TARGET-CPU-IMPORT-v9.json',
        installed_backend_evidence='CALIBRATION-BACKEND-INSTALLED-EVIDENCE-v9.json',diagnosis='PROBABILITY-MISMATCH-3281-DIAGNOSIS-v1.json')
    return {role:dict(path=str(prepare.OUT/name),bytes=(prepare.OUT/name).stat().st_size,sha256=contract.sha(prepare.OUT/name)) for role,name in names.items()}


def test_actual_v9_runtime_and_backend_CPU_bound_without_GPU_claim():
    runtime,paths=prepare.validate_backend_runtime(backend_proofs())
    assert len(runtime['files'])==14 and len(runtime['unchanged_v8_filenames'])==11
    assert runtime['checks']['actual_GPU_acceptance'] is False
    assert prepare.one_outer_ast_sha(prepare.RUNTIME/'calibration_driver_v1.py')==prepare.one_outer_ast_sha(prepare.OLD_RUNTIME/'calibration_driver_v1.py')


@pytest.mark.parametrize('mutation',['manifest','diff','cpu','scope','changed','gpu','source','threshold','status'])
def test_exact_new_backend_review_reject(mutation):
    proofs=backend_proofs();review=json.loads(Path(proofs['runtime_review']['path']).read_text())
    if mutation in ('manifest','diff','cpu'):review['runtime_'+mutation+'_sha256']='f'*64
    elif mutation=='scope':review['backend_change_scope']='GUESS_PARITY'
    elif mutation=='changed':review['runtime_changed_filenames']=['calibration_driver_v1.py']
    elif mutation=='gpu':review['gpu_used']=True
    elif mutation=='source':review['source_modified']=True
    elif mutation=='threshold':review['production_thresholds_selected']=True
    else:review['status']='NEEDS_FIX'
    with pytest.raises((ValueError,KeyError)):prepare.validate_runtime_review(proofs,review,contract.BACKEND_CHANGED)


def current_target_inputs(tmp, mutation=None):
    names=dict(runtime_target_cpu='CALIBRATION-DATAROW-TARGET-CPU-v9.json',runtime_target_cpu_raw='CALIBRATION-DATAROW-TARGET-CPU-RAW-v9.json',
        runtime_target_cpu_review='CALIBRATION-DATAROW-TARGET-CPU-INDEPENDENT-REVIEW-v9.json')
    values={role:json.loads((prepare.OUT/name).read_text()) for role,name in names.items()}
    c=values['runtime_target_cpu'];w=values['runtime_target_cpu_raw'];r=values['runtime_target_cpu_review']
    if mutation=='driver':c['runtime_driver_sha256']='f'*64
    elif mutation=='torch':c['torch_version']='2.8.0'
    elif mutation=='cuda':c['actual_CUDA_initialized']=True
    elif mutation=='adam':c['actual_Adam_writes']=1
    elif mutation=='booladam':c['actual_Adam_writes']=False
    elif mutation=='manifest':r['runtime_manifest_sha256']='f'*64
    elif mutation=='reviewgpu':r['actual_GPU_backend_accepted']=True
    elif mutation=='reviewprob':r['actual_probability_accepted']=True
    elif mutation=='failed':w['actual']['returncode']=1
    paths={role:tmp/(role+'.json') for role in names}
    paths['runtime_target_cpu'].write_text(json.dumps(c))
    proofs={role:dict(path=str(prepare.OUT/name),bytes=(prepare.OUT/name).stat().st_size,sha256=contract.sha(prepare.OUT/name)) for role,name in names.items()}
    proofs['runtime_target_cpu'].update(bytes=paths['runtime_target_cpu'].stat().st_size,sha256=contract.sha(paths['runtime_target_cpu']))
    w['actual']['stdout']=json.dumps(c);w['actual']['records']['target_cpu'].update(data=c,bytes=proofs['runtime_target_cpu']['bytes'],sha256=proofs['runtime_target_cpu']['sha256'])
    import base64
    original=dict(w['actual'],cpu_base64=base64.b64encode(paths['runtime_target_cpu'].read_bytes()).decode())
    if mutation=='corruptbase64':original['cpu_base64']=base64.b64encode(b'wrong').decode()
    if mutation=='rawdiff':original['returncode']=999
    w['raw_stdout']=json.dumps(original);paths['runtime_target_cpu_raw'].write_text(json.dumps(w))
    proofs['runtime_target_cpu_raw'].update(bytes=paths['runtime_target_cpu_raw'].stat().st_size,sha256=contract.sha(paths['runtime_target_cpu_raw']))
    r.update(runtime_target_cpu_sha256=proofs['runtime_target_cpu']['sha256'],runtime_target_cpu_raw_sha256=proofs['runtime_target_cpu_raw']['sha256'])
    paths['runtime_target_cpu_review'].write_text(json.dumps(r))
    return proofs,paths


@pytest.mark.parametrize('mutation',['driver','torch','cuda','adam','booladam','manifest','reviewgpu','reviewprob','failed','corruptbase64','rawdiff'])
def test_new_driver_target_CPU_not_old_bytes_or_GPU_acceptance(tmp_path,mutation):
    with pytest.raises((ValueError,KeyError)):
        prepare.validate_current_target_cpu_inputs(*current_target_inputs(tmp_path,mutation),
            '31f191711844c0c875149502e9c2ddfc9569f40ef88163baf860aa46dbd762b9',backend_proofs()['runtime_manifest']['sha256'])


def test_actual_new_driver_target_CPU_original_accepts_only_CPU_scope(tmp_path):
    prepare.validate_current_target_cpu_inputs(*current_target_inputs(tmp_path),
        '31f191711844c0c875149502e9c2ddfc9569f40ef88163baf860aa46dbd762b9',backend_proofs()['runtime_manifest']['sha256'])


@pytest.mark.parametrize('text,expected',[('47 passed in 0.3s',47),('Ran 47 tests in 0.3s\n\nOK',47),
    ('47 passed, 1 failed',0),('FAILED with 47 passed',0),('47 passed then unexpected',0)])
def test_success_count_rejects_failure_or_nonterminal(text,expected):
    assert prepare.cpu_success_count(text)==expected

import submit_once_v9 as submit


def submit_fixture(tmp, monkeypatch):
    root=tmp/'root';(root/'logs').mkdir(parents=True);(root/'tools').mkdir()
    monkeypatch.setattr(submit,'ROOT',root);monkeypatch.setattr(submit,'HELPERS',root/'tools/runtime-v9')
    monkeypatch.setattr(submit,'INTENT',root/'logs/intent-v9');monkeypatch.setattr(submit,'RECEIPT',root/'logs/receipt-v9')
    monkeypatch.setattr(submit.socket,'gethostname',lambda:'h100-8')
    b=root/'logs/policy-safety-calibration-52befe9-reviewed-binding-v9.json'
    proof=root/'logs/original';proof.write_bytes(b'bound')
    binding=dict(files=[dict(path=str(proof),bytes=5)],evidence=[],source_native_acceptance=dict(job_id='3267'),repo='unchanged52',helpers=str(submit.HELPERS))
    review=dict(status='PASS',scope=submit.SCOPE,binding_sha256='b'*64,submit_helper_sha256=submit.sha(Path(submit.__file__)))
    monkeypatch.setattr(submit,'read_bound',lambda item:binding if item['path']==str(b) else review)
    monkeypatch.setattr(submit,'validate_ready',lambda value:None) # Controller CPU scope only; READY/native gates separately tested.
    monkeypatch.setattr(submit.subprocess,'check_output',lambda command,**kwargs:'')
    calls=[]
    class Completed:
        returncode=0;stdout='9001;h100\\n';stderr=''
    def launch(command,**kwargs):calls.append(command);return Completed()
    monkeypatch.setattr(submit.subprocess,'run',launch)
    return b,review,Completed,calls


@pytest.mark.parametrize('mutation',['intent','receipt','symlink_intent','host','binding','occupied','changed_size','review'])
def test_direct_once_no_claim_or_submission_on_unknown_or_occupied(tmp_path,monkeypatch,mutation):
    b,review,completed,calls=submit_fixture(tmp_path,monkeypatch)
    if mutation=='intent':submit.INTENT.write_text('claimed')
    elif mutation=='receipt':submit.RECEIPT.write_text('uncertain')
    elif mutation=='symlink_intent':submit.INTENT.symlink_to(tmp_path/'missing')
    elif mutation=='host':monkeypatch.setattr(submit.socket,'gethostname',lambda:'h100-11')
    elif mutation=='binding':b=b.with_name('reviewed-binding-v8.json')
    elif mutation=='occupied':monkeypatch.setattr(submit.subprocess,'check_output',lambda *a,**k:'another-job')
    elif mutation=='changed_size':(submit.ROOT/'logs/original').write_bytes(b'changed')
    else:review['binding_sha256']='f'*64
    with pytest.raises((ValueError,KeyError)):submit.main(str(b),'b'*64,'review','r'*64)
    assert calls==[]


@pytest.mark.parametrize('response',['nonsense','9001;h100 extra','9001\\n9002','0','9001;'])
def test_ambiguous_sbatch_preserves_single_claim_and_never_retries(tmp_path,monkeypatch,response):
    b,review,completed,calls=submit_fixture(tmp_path,monkeypatch);completed.stdout=response
    with pytest.raises(ValueError):submit.main(str(b),'b'*64,'review','r'*64)
    assert len(calls)==1 and submit.INTENT.exists()
    assert json.loads(submit.RECEIPT.read_text())['status']=='UNCERTAIN_OR_FAILED_NO_RETRY'
    with pytest.raises(ValueError):submit.main(str(b),'b'*64,'review','r'*64)
    assert len(calls)==1


def test_direct_once_valid_receipt_excludes_duplicate(tmp_path,monkeypatch):
    b,review,completed,calls=submit_fixture(tmp_path,monkeypatch)
    completed.stdout='9001;h100\n'
    result=submit.main(str(b),'b'*64,'review','r'*64)
    assert result['job_id']=='9001' and result['retained_training_updates_contributed']==0 and len(calls)==1
    with pytest.raises(ValueError):submit.main(str(b),'b'*64,'review','r'*64)
    assert len(calls)==1


def test_submit_transport_exception_preserves_claim_and_uncertain_receipt(tmp_path,monkeypatch):
    b,review,completed,calls=submit_fixture(tmp_path,monkeypatch)
    def fail(command,**kwargs):calls.append(command);raise TimeoutError('CPU transport fixture timeout')
    monkeypatch.setattr(submit.subprocess,'run',fail)
    with pytest.raises(TimeoutError):submit.main(str(b),'b'*64,'review','r'*64)
    assert submit.INTENT.exists() and json.loads(submit.RECEIPT.read_text())['status']=='UNCERTAIN_OR_FAILED_NO_RETRY'
    with pytest.raises(ValueError):submit.main(str(b),'b'*64,'review','r'*64)
    assert len(calls)==1


def test_inherited_datarow_work_probe_uses_exact_v8_manifest_original():
    record=dict(path=str(contract.ROOT/'logs/test_calibration_datarow_v8_actual.py'),bytes=13740,
        sha256='dd379a3a50f78a49ba21b8ef805c9a7b7fecf4eba90a9832dca6ea4f37ba87b9')
    assert prepare.inherited_local_proof(record).is_file()
    with pytest.raises(ValueError):prepare.inherited_local_proof(dict(record,sha256='f'*64))
