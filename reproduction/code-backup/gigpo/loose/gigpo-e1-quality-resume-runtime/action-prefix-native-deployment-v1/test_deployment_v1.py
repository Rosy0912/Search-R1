"""New native-only control tests; fixtures are explicit CPU evidence only."""
import copy
import concurrent.futures
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
import deployment_contract_v1 as contract
import submit_once_v1 as once


def record(path, letter='a'):
    return dict(path=str(path), bytes=1, sha256=letter * 64)


def candidate_fixture():
    files = [record(contract.REPO / ('tracked_' + str(n) + '.py')) for n in range(480)]
    files += [record(contract.HELPERS / n) for n in contract.HELPER_NAMES]
    files += [record(contract.CONTROLS / n) for n in contract.NAMES]
    for name, digest in [('resource_guard_v2.py', '9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'),
            ('resource_ownership_v2.py', '7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed')]:
        next(x for x in files if x['path'] == str(contract.HELPERS / name))['sha256'] = digest
    driver = dict(path=str(contract.RESTORE / 'training_state/driver.pt'), bytes=742936, sha256=contract.DRIVER_SHA)
    files.append(driver)
    clone = dict(path=str(contract.CLONE), bytes=128687, sha256=contract.CLONE_SHA)
    roles = {key:record(contract.ROOT / ('logs/' + key + '.json')) for key in
             ('source_implementation', 'source_review', 'native_helper_manifest', 'native_helper_review')}
    return dict(status='CANDIDATE_NOT_READY', code_commit=contract.SOURCE, repo=str(contract.REPO),
        helpers=str(contract.HELPERS), controls=str(contract.CONTROLS), native_scope=contract.SCOPE,
        validation_scope='GIGPO_POLICY_SAFETY_NATIVE_ONLY', native_initialization='OFFICIAL_MODEL_INITIALIZATION',
        native_actual_restore=False, formal_updates_contributed=0, official_revision=contract.REVISION,
        official_model_path=str(contract.MODEL), native_config_driver=driver, restore_step=25,
        restore_checkpoint=str(contract.RESTORE), restore_manifest_sha256='3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4',
        allowed_safety_modes=['native_fixture'], native_test_threshold_sets=[[100., 1.]],
        expected_optimizer_steps_per_outer=10, expected_gradient_micro_batches_per_rank=160,
        global_row_denominator=256, production_denominator_per_mini=32, gpu_count=8, node='h100-8',
        native_submitted=False, training_submitted=False, production_probability_tolerance=None,
        production_thresholds=None, source_native_acceptance=None, files=files,
        evidence=copy.deepcopy([clone] + list(roles.values())), source_file_count=480, source_actual_clone=clone,
        resource_guard=dict(path=str(contract.HELPERS / 'resource_guard_v2.py'),
            sha256='9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'),
        resource_ownership=dict(path=str(contract.HELPERS / 'resource_ownership_v2.py'),
            sha256='7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed'),
        proof_roles=roles, parent_candidate={'sha256':contract.BASE_CANDIDATE_SHA},
        inherited_physical={'sha256':contract.BASE_PHYSICAL_SHA})


def test_complete_explicit_synthetic_candidate_schema():
    assert contract.validate_candidate(candidate_fixture())


def source_review_fixture():
    return dict(status='PASS_SOURCE_AND_EXACT_CPU_EVIDENCE_SCOPE_ONLY',
        scope='RAW_SAMPLED_ACTION_IDENTITY_SOURCE_STATIC_REVIEW_NOT_GPU_OR_PROBABILITY_OR_DEPLOYMENT_READY',
        source_commit=contract.SOURCE, source_manifest_sha256='b'*64, gpu_used=False,
        deployment_ready=False, actual_gpu_path_accepted=False, probability_identity_accepted=False,
        production_thresholds_selected=False)


def test_exact_narrow_source_review_status_supported_without_relabel():
    assert contract.validate_source_review(source_review_fixture(),'b'*64)


@pytest.mark.parametrize('field,value',[('status','PASS'),('source_commit','5'*40),('source_manifest_sha256','a'*64),
    ('gpu_used',True),('deployment_ready',True),('actual_gpu_path_accepted',True),('probability_identity_accepted',True),
    ('production_thresholds_selected',True)])
def test_narrow_source_review_cannot_be_relabelled_or_inflated(field,value):
    review=source_review_fixture();review[field]=value
    with pytest.raises(ValueError):contract.validate_source_review(review,'b'*64)


@pytest.mark.parametrize('field,value', [('code_commit', '5'*40), ('repo', str(contract.ROOT / 'repo-policy-safety-52befe9')),
    ('helpers', str(contract.ROOT / 'tools/policy-safety-native-52befe9-v2')), ('native_actual_restore', True),
    ('native_initialization','HEALTHY25_RESTORE'), ('formal_updates_contributed',1), ('official_revision','main'),
    ('source_native_acceptance',{'job':3267}), ('production_probability_tolerance',.1), ('production_thresholds',{}),
    ('allowed_safety_modes',['production']), ('native_test_threshold_sets',[[.02,.001]]),
    ('expected_optimizer_steps_per_outer',20), ('expected_gradient_micro_batches_per_rank',80), ('gpu_count',4),
    ('node','a800'), ('source_file_count',439), ('native_submitted',True)])
def test_scope_and_identity_refusals(field,value):
    candidate=candidate_fixture();candidate[field]=value
    with pytest.raises(ValueError):contract.validate_candidate(candidate)


@pytest.mark.parametrize('kind',['source_missing','source_duplicate','helper_extra','control_missing','guard_changed',
    'evidence_duplicate','proof_role_missing','proof_role_unbound','clone_changed','parent_changed','driver_missing'])
def test_exact_closure_refusals(kind):
    c=candidate_fixture()
    if kind=='source_missing':c['files'].pop(0)
    elif kind=='source_duplicate':c['files'].append(copy.deepcopy(c['files'][0]))
    elif kind=='helper_extra':c['files'].append(record(contract.HELPERS / 'unknown.py'))
    elif kind=='control_missing':c['files']=[r for r in c['files'] if r['path']!=str(contract.CONTROLS/'submit_once_v1.py')]
    elif kind=='guard_changed':c['resource_guard']['sha256']='0'*64
    elif kind=='evidence_duplicate':c['evidence'].append(copy.deepcopy(c['evidence'][0]))
    elif kind=='proof_role_missing':del c['proof_roles']['native_helper_review']
    elif kind=='proof_role_unbound':c['proof_roles']['native_helper_review']['sha256']='b'*64
    elif kind=='clone_changed':c['source_actual_clone']['sha256']='1'*64
    elif kind=='parent_changed':c['parent_candidate']['sha256']='2'*64
    else:c['files']=[r for r in c['files'] if not r['path'].endswith('/driver.pt')]
    with pytest.raises(ValueError):contract.validate_candidate(c)


@pytest.mark.parametrize('kind',['extra','missing','duplicate','different_sha','different_bytes'])
def test_physical_closure_cannot_substitute_bytes(kind):
    expected=[record('/absolute/test')];actual=copy.deepcopy(expected)
    if kind=='extra':actual.append(record('/absolute/extra'))
    elif kind=='missing':actual=[]
    elif kind=='duplicate':actual.append(copy.deepcopy(actual[0]))
    elif kind=='different_sha':actual[0]['sha256']='b'*64
    else:actual[0]['bytes']=2
    with pytest.raises(ValueError):contract.match_closure(actual,expected,'CPUfixture')


def setup_once(monkeypatch,tmp_path):
    for field,name in [('INTENT','intent.json'),('RECEIPT','receipt.json'),('BINDING','binding.json'),('READY','ready.json')]:
        monkeypatch.setattr(once,field,tmp_path/name)
    monkeypatch.setattr(once,'CONTROLS',Path(once.__file__).resolve().parent)
    review=tmp_path/'review.json';review.write_text('explicit synthetic fixture')
    value=dict(files=[],evidence=[],candidate_record=dict(path='/fixture/candidate',bytes=1,sha256='c'*64),
        physical_record=dict(path='/fixture/physical',bytes=1,sha256='d'*64),
        deployment_review=dict(path=str(review),bytes=1,sha256='a'*64))
    monkeypatch.setattr(once,'read_bound',lambda r:copy.deepcopy(value))
    monkeypatch.setattr(once,'validate_ready',lambda v:dict(repo=str(contract.REPO)))
    monkeypatch.setattr(once,'sha',lambda p:'a'*64)
    return value


def response(out='',rc=0):return SimpleNamespace(stdout=out,stderr='',returncode=rc)


@pytest.mark.parametrize('kind',['queue_occupied','compute_occupied','queue_unreadable','compute_unreadable','wrong_node'])
def test_no_claim_on_occupied_unknown_or_wrong_node(monkeypatch,tmp_path,kind):
    setup_once(monkeypatch,tmp_path);calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[0]=='squeue':return response('123 other RUNNING' if kind=='queue_occupied' else '',1 if kind=='queue_unreadable' else 0)
        return response('321 foreign 1000' if kind=='compute_occupied' else '',1 if kind=='compute_unreadable' else 0)
    with pytest.raises(ValueError):once.submit('a'*64,run,lambda:'other' if kind=='wrong_node' else 'h100-8')
    assert not once.INTENT.exists() and not once.RECEIPT.exists() and not once.BINDING.exists()
    assert not any(x[0]=='sbatch' for x in calls)


@pytest.mark.parametrize('kind',['timeout','nonzero','unknown_job'])
def test_uncertain_response_persists_claim_and_never_retries(monkeypatch,tmp_path,kind):
    setup_once(monkeypatch,tmp_path);calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[0]!='sbatch':return response()
        if kind=='timeout':raise TimeoutError('synthetic uncertain submit')
        return response('not-a-job' if kind=='unknown_job' else '123',1 if kind=='nonzero' else 0)
    with pytest.raises((ValueError,TimeoutError)):once.submit('a'*64,run,lambda:'h100-8')
    assert once.INTENT.exists() and once.BINDING.exists() and once.RECEIPT.exists()
    assert json.loads(once.RECEIPT.read_text())['status']=='UNCERTAIN_NO_RESUBMISSION'
    with pytest.raises(ValueError):once.submit('a'*64,run,lambda:'h100-8')
    assert sum(x[0]=='sbatch' for x in calls)==1


def test_two_concurrent_claimers_submit_exactly_once(monkeypatch,tmp_path):
    setup_once(monkeypatch,tmp_path);barrier=threading.Barrier(2);calls=[];lock=threading.Lock()
    def run(args,**kwargs):
        if args[0]=='nvidia-smi':barrier.wait(timeout=5)
        if args[0]=='sbatch':
            with lock:calls.append(args)
            return response('999;cluster')
        return response()
    def attempt():
        try:return once.submit('a'*64,run,lambda:'h100-8')
        except (FileExistsError,ValueError) as error:return type(error).__name__
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:attempt(),range(2)))
    assert len(calls)==1 and sum(isinstance(x,dict) for x in results)==1
    assert len(calls[0])==7 and calls[0][0:2]==['sbatch','--parsable']
    assert json.loads(once.RECEIPT.read_text())['status']=='SUBMITTED_EXACTLY_ONCE'
    assert json.loads(once.RECEIPT.read_text())['job']=='999'


def proof_input_fixture():
    def entry(name):return dict(local_path='/explicit/local/fixture/'+name,
        path=str(contract.ROOT/'logs'/name),bytes=1,sha256='a'*64)
    return dict(roles={role:entry(role+'.json') for role in
        ('source_implementation','source_review','native_helper_manifest','native_helper_review')},
        additional_evidence=[entry('cpu.log')])


def test_new_proof_dependency_shape_accepts_explicit_cpu_fixture_only():
    assert contract.validate_proof_input(proof_input_fixture())


@pytest.mark.parametrize('kind',['missing_dependency','duplicate_path','outside_logs','traversal','missing_bytes'])
def test_new_proof_dependencies_refuse_unclosed_or_replaced_originals(kind):
    value=proof_input_fixture()
    if kind=='missing_dependency':value['additional_evidence']=[]
    elif kind=='duplicate_path':value['additional_evidence'].append(copy.deepcopy(next(iter(value['roles'].values()))))
    elif kind=='outside_logs':value['additional_evidence'][0]['path']=str(contract.ROOT/'tools/invalid-proof.log')
    elif kind=='traversal':value['additional_evidence'][0]['path']=str(contract.ROOT/'logs/../invalid-proof.log')
    else:del value['additional_evidence'][0]['bytes']
    with pytest.raises(ValueError):contract.validate_proof_input(value)
