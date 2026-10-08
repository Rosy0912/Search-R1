"""External v7 factory closure regressions only; no native/GPU/target execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
import deployment_contract_v7 as contract
import prepare_candidate_v7 as prepare

V6=Path(__file__).resolve().parent.with_name('safety-calibration-deployment-v6')
sys.path.append(str(V6))
import test_deployment_v6 as fixtures


def candidate7():
    c=fixtures.isolated_candidate6(fixtures.native_fixture5()[4])
    old={r['path'] for r in c['runtime_helpers']}
    c['files']=[r for r in c['files'] if r['path'] not in old]
    c['runtime_helpers']=[dict(r,path=str(contract.HELPERS/Path(r['path']).name)) for r in c['runtime_helpers']]
    c['runtime_helpers'].append(dict(path=str(contract.HELPERS/'metadata_colocation_v1.py'),bytes=1,sha256='a'*64))
    c['files']+=c['runtime_helpers']
    c.update(helpers=str(contract.HELPERS),controls=str(contract.CONTROLS),factory_change_scope=contract.FACTORY_SCOPE,
        runtime_changed_filenames=copy.deepcopy(contract.FACTORY_CHANGED),startup_runtime_changed_filenames=['calibration_run_v1.py'],
        factory_base_runtime_manifest_sha256=prepare.BASE_RUNTIME_MANIFEST_SHA,
        unchanged_runtime_wrapper_sha256='7aa80984aee6a5589ee44f9b72c4bed0d091e30c38e9c181fe2cc2728d1172c9')
    c['factory_control_proofs']={role:dict(path=str(contract.ROOT/'logs'/('unit-factory-'+role+'.json')),bytes=1,sha256='b'*64)
        for role in contract.FACTORY_PROOF_ROLES}
    c['evidence']+=list(c['factory_control_proofs'].values())
    return c


@pytest.mark.parametrize('mutation',['runtime13','wrong_helper','old_controls','factory_scope','extra_changed_math',
 'missing_added_metadata','wrong_base','changed_wrapper','startup_scope','startup_change',
 'missing_factory_proof','duplicate_factory_proof','unbound_factory_proof','outside_factory_proof',
 'missing_startup_proof','then_update','production_threshold','native_pass_before_ready'])
def test_v7_external_factory_scope_and_inherited_startup_closure_refuse(mutation):
    c=candidate7()
    if mutation=='runtime13':c['runtime_helpers'].pop()
    elif mutation=='wrong_helper':c['helpers']=str(fixtures.contract.HELPERS)
    elif mutation=='old_controls':c['controls']=str(fixtures.contract.CONTROLS)
    elif mutation=='factory_scope':c['factory_change_scope']='algorithm-change'
    elif mutation=='extra_changed_math':c['runtime_changed_filenames'].append('calibration_contract_v1.py')
    elif mutation=='missing_added_metadata':c['runtime_changed_filenames']=['calibration_driver_v1.py']
    elif mutation=='wrong_base':c['factory_base_runtime_manifest_sha256']='f'*64
    elif mutation=='changed_wrapper':c['unchanged_runtime_wrapper_sha256']='f'*64
    elif mutation=='startup_scope':c['startup_change_scope']='relaxed15'
    elif mutation=='startup_change':c['startup_runtime_changed_filenames'].append('guard.py')
    elif mutation=='missing_factory_proof':c['factory_control_proofs'].pop('runtime_review')
    elif mutation=='duplicate_factory_proof':c['factory_control_proofs']['runtime_review']=copy.deepcopy(c['factory_control_proofs']['runtime_cpu'])
    elif mutation=='unbound_factory_proof':c['evidence'].pop()
    elif mutation=='outside_factory_proof':c['factory_control_proofs']['runtime_review']['path']='/tmp/proof.json'
    elif mutation=='missing_startup_proof':c['startup_control_proofs'].pop('failure_cgroup')
    elif mutation=='then_update':c['calibration_phase']='probability_probe_then_update'
    elif mutation=='production_threshold':c['target_old_policy_kl']=1.
    else:c['source_native_acceptance']={'status':'PASS'}
    with pytest.raises((ValueError,KeyError)):contract.validate_candidate(c)


def test_new14_factory_candidate_retains_native_and_no_approval():
    c=candidate7();contract.validate_candidate(c)
    assert len(c['runtime_helpers'])==14 and c['source_native_acceptance'] is None
    assert c['calibration_phase']=='probability_probe_then_wait' and c['target_old_policy_kl'] is None
    assert c['startup_change_scope']==fixtures.contract.STARTUP_SCOPE


@pytest.mark.parametrize('mutation',['wrong_manifest','wrong_diff','wrong_cpu','wrong_scope','extra_file','gpu','source','threshold','not_pass'])
def test_factory_runtime_review_is_explicit_exact_static_not_gpu(mutation):
    proofs={role:dict(sha256=str(index)*64) for index,role in enumerate(contract.FACTORY_PROOF_ROLES)}
    r=dict(status='PASS',code_commit=contract.SOURCE,runtime_manifest_sha256=proofs['runtime_manifest']['sha256'],
        runtime_diff_sha256=proofs['runtime_diff']['sha256'],runtime_cpu_sha256=proofs['runtime_cpu']['sha256'],
        factory_change_scope=contract.FACTORY_SCOPE,runtime_changed_filenames=copy.deepcopy(contract.FACTORY_CHANGED),
        gpu_used=False,source_modified=False,production_thresholds_selected=False)
    if mutation.startswith('wrong_'):
        key={'wrong_manifest':'runtime_manifest_sha256','wrong_diff':'runtime_diff_sha256','wrong_cpu':'runtime_cpu_sha256',
             'wrong_scope':'factory_change_scope'}[mutation];r[key]='wrong'
    elif mutation=='extra_file':r['runtime_changed_filenames'].append('policy_safety.py')
    elif mutation=='gpu':r['gpu_used']=True
    elif mutation=='source':r['source_modified']=True
    elif mutation=='threshold':r['production_thresholds_selected']=True
    else:r['status']='NEEDS_FIX'
    with pytest.raises((ValueError,KeyError)):prepare.validate_runtime_review(proofs,r,contract.FACTORY_CHANGED)


def failure_inputs(tmp,mutation=None):
    out=prepare.OUT
    payload=json.loads((out/'CALIBRATION3276-FAILURE-PAYLOAD-v1.json').read_text())
    review=json.loads((out/'CALIBRATION3276-FAILURE-INDEPENDENT-REVIEW-v1.json').read_text())
    diagnostic=json.loads((out/'CALIBRATION3276-STARTUP-MRO-INDEPENDENT-DIAGNOSIS-v1.json').read_text())
    if mutation=='adam':review['actual_adam_writes']=1
    elif mutation=='bool_adam':review['actual_adam_writes']=False
    elif mutation=='restore_complete':review['actual_restore25_completed']=True
    elif mutation=='sampling':review['actual_sampling']=True
    elif mutation=='cgroup':review['resource']['cgroup_job_and_step_ENOENT_with_readable_slurm_ancestor']=False
    elif mutation=='residual':review['resource']['saved_final_compute']=[{'pid':1}]
    elif mutation=='wrong_job':payload['job_id']='3272'
    elif mutation=='corrupt_raw':payload['originals'][0]['text']+='changed'
    elif mutation=='missing_raw':payload['originals'].pop()
    elif mutation=='wrong_diagnosis':diagnostic['job_id']='3272'
    paths={role:tmp/(role+'.json') for role in ('failure_payload','failure_review','failure_mro_diagnostic')}
    paths['failure_payload'].write_text(json.dumps(payload));paths['failure_mro_diagnostic'].write_text(json.dumps(diagnostic))
    review['consumed_payload']={'path':str(paths['failure_payload']),'sha256':contract.sha(paths['failure_payload'])}
    review['diagnosis']['independent_diagnosis_path']=str(paths['failure_mro_diagnostic'])
    review['diagnosis']['independent_diagnosis_sha256']=contract.sha(paths['failure_mro_diagnostic'])
    paths['failure_review'].write_text(json.dumps(review))
    proofs={role:dict(path=str(path),bytes=path.stat().st_size,sha256=contract.sha(path)) for role,path in paths.items()}
    return proofs,paths


@pytest.mark.parametrize('mutation',['adam','bool_adam','restore_complete','sampling','cgroup','residual',
 'wrong_job','corrupt_raw','missing_raw','wrong_diagnosis'])
def test_actual_failure3276_zero_update_release_and_mro_evidence_closes(tmp_path,mutation):
    with pytest.raises((ValueError,KeyError)):prepare.validate_failure_inputs(*failure_inputs(tmp_path,mutation))


def test_actual_original_failure_3276_preserved_and_not_calibration_pass(tmp_path):
    originals=prepare.validate_failure_inputs(*failure_inputs(tmp_path))
    assert len(originals)==11


def test_actual_runtime14_original_hashes_and_static_review_only():
    roles=dict(runtime_manifest='CALIBRATION-RUNTIME-IMPLEMENTATION-v7.json',runtime_diff='CALIBRATION-RUNTIME-DIFF-v7.patch',
        runtime_cpu='CALIBRATION-FACTORY-CPU-v7.json',runtime_review='CALIBRATION-COLOCATION-INDEPENDENT-REVIEW-v7.json',
        failure_payload='CALIBRATION3276-FAILURE-PAYLOAD-v1.json',failure_review='CALIBRATION3276-FAILURE-INDEPENDENT-REVIEW-v1.json',
        failure_mro_diagnostic='CALIBRATION3276-STARTUP-MRO-INDEPENDENT-DIAGNOSIS-v1.json')
    proofs={role:dict(path=str(prepare.OUT/name),bytes=(prepare.OUT/name).stat().st_size,sha256=contract.sha(prepare.OUT/name))
        for role,name in roles.items()}
    runtime,paths=prepare.validate_runtime_inputs(proofs)
    assert len(runtime['files'])==14 and runtime['wrapper_sha256']==candidate7()['unchanged_runtime_wrapper_sha256']
