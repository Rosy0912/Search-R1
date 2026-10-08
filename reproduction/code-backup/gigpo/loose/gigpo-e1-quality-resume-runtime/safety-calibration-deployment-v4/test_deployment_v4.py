"""Unit counterexamples only: no remote, Slurm, native or GPU acceptance."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import base64
import gzip
from pathlib import Path

import pytest

from deployment_contract_v4 import *
import deployment_contract_v4 as contract


@pytest.mark.parametrize('mutation', ['native_null', 'one_outer', 'probe_only', 'production_mode',
                                    'production_kl', 'invented_probability_tolerance'])
def test_real_deployment_draft_cannot_launch_or_escape_current_live_review(mutation):
    # Local generated candidate is a draft; these tests never read remote paths.
    output = Path(__file__).resolve().parents[3] / 'outputs/gigpo-e1-policy-safety'
    path = output / 'CALIBRATION-CANDIDATE-52BEFE9-v3.json'
    candidate = json.loads(path.read_text())
    candidate.update(helpers=str(HELPERS), controls=str(CONTROLS))
    implementation = json.loads((output/'CALIBRATION-IMPLEMENTATION-v1.json').read_text())
    candidate['runtime_helpers'] = [dict(path=str(HELPERS/Path(r['path']).name),bytes=r['bytes'],sha256=r['sha256'])
                                    for r in implementation['files']]
    candidate['files'] += candidate['runtime_helpers']
    if mutation == 'native_null':
        with pytest.raises(ValueError):validate_ready(candidate)
        with pytest.raises(ValueError, match='null'):read_native_acceptance(None, candidate)
        return
    if mutation in ('one_outer','probe_only'):candidate['calibration_phase']=mutation
    elif mutation == 'production_mode':candidate['allowed_safety_modes']=['production']
    elif mutation == 'production_kl':candidate['target_old_policy_kl']=.01
    else:candidate['actor_vllm_abs_tolerance']=.01
    with pytest.raises(ValueError):validate_controls(candidate)


def native_fixture(root=ROOT):
    guard = dict(resource_guard=dict(path=str(NATIVE_HELPERS/'resource_guard_v2.py'),sha256='a'*64),
                 resource_ownership=dict(path=str(NATIVE_HELPERS/'resource_ownership_v2.py'),sha256='b'*64))
    run=root/'runs/gigpo-e1-policy-safety-native-job9999'
    acceptance = dict(status='PASS', code_commit=SOURCE, job_id='9999', result_sha256='d'*64,
                      native_binding_sha256='e'*64, guard_identity=guard,
                      result_path=str(run/'result.json'),native_binding_path=str(root/'logs/native-binding.json'),
                      terminal_path=str(root/'logs/quality10-native-terminal-9999-raw-v1.json'),terminal_sha256='f'*64,
                      physical_payload_path=str(root/'logs/native-terminal-physical.json.gz'),physical_payload_sha256='c'*64,
                      terminal_state='COMPLETED',terminal_exit_code='0:0',terminal_requeue=0,terminal_restarts=0,cgroup_removed=True)
    result = dict(status='PASS',job='9999',validation_scope=NATIVE_SCOPE,code_commit=SOURCE,
        restore_manifest_sha256=MANIFEST,cpu_exit=0,gpu_exit=0,fixtures_passed=True,actual_outer_updates=0,
        cleanup_residual_states={},final_gpu=dict(processes=[]),guard_identity=guard,binding_sha256='e'*64,
        cgroup='/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_9999/step_batch',
        oracle_records=[dict(rank=i,status='PASS',sequence_pg_kl_dp8_micro2_equal=True,
            production_denominator_per_mini=32,global_row_denominator=256,
            single_invalid_rank_synchronous_refusal=True,no_extra_adam_after_refusal=True) for i in range(8)])
    result['phases'] = [[dict(phase=phase,rank=i,status='PASS',full_saved_state_equal=True,
        official_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',source_manifest_sha256=MANIFEST,
        optimizer_steps_before=500,optimizer_steps_after=510,scheduler=25,native_mini_optimizer_steps=10,
        gradient_micro_batches=160,monitor_guard_count=20,actual_outer_updates=0,
        restored_identity=dict(rank=i,pid=1000+i+(phase=='B')*100,start_ticks='1',model='unit-only'),
        next_mini_identity=dict(rank=i,pid=1000+i+(phase=='B')*100,start_ticks='1',model='unit-only'),
        python_numpy_rng_sha256='f'*64) for i in range(8)] for phase in ('A','B')]
    binding = dict(code_commit=SOURCE,helpers=str(NATIVE_HELPERS),**guard)
    terminal = dict(fields=dict(JobId='9999',NodeList='h100-8',JobState='COMPLETED',ExitCode='0:0',Requeue='0',Restarts='0'))
    def original(path, value):
        payload=json.dumps(value,sort_keys=True).encode()
        return dict(path=str(path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                    content_base64=base64.b64encode(payload).decode())
    b_record=original(acceptance['native_binding_path'],binding)
    acceptance['native_binding_sha256']=result['binding_sha256']=b_record['sha256']
    r_record=original(acceptance['result_path'],result);t_record=original(acceptance['terminal_path'],terminal)
    acceptance['result_sha256']=r_record['sha256'];acceptance['terminal_sha256']=t_record['sha256']
    review = dict(status='PASS',scope=NATIVE_SCOPE,code_commit=SOURCE,job_id='9999',restore_manifest_sha256=MANIFEST,
        result_sha256=acceptance['result_sha256'],terminal_path=acceptance['terminal_path'],terminal_sha256=acceptance['terminal_sha256'],
        terminal_state='COMPLETED',terminal_exit_code='0:0',terminal_requeue=0,terminal_restarts=0,cgroup_removed=True,
        physical_payload_path=acceptance['physical_payload_path'],physical_payload_sha256=acceptance['physical_payload_sha256'])
    physical=dict(status='ORIGINAL_NATIVE_TERMINAL_AND_PHYSICAL_EVIDENCE_UNREVIEWED',job='9999',run=str(run),
        result=copy.deepcopy(result),terminal=copy.deepcopy(terminal),own_cgroups=[],own_job_queue='',
        cgroup_release=dict(job_id='9999',path='/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_9999',
            result_cgroup=result['cgroup'],ancestor_path='/sys/fs/cgroup/freezer/slurm_h100-8',
            ancestor_readable=True,stat_outcome='ENOENT',scope_stat_outcome='ENOENT',removed=True),
        current_compute=dict(returncode=0,stdout='',stderr=''),
        current_node_queue='',originals=[r_record,t_record,b_record],
        physical=[{k:r_record[k] for k in ('path','bytes','sha256')}])
    return acceptance,result,review,binding,guard,terminal,physical


@pytest.mark.parametrize('mutation', ['null', 'failed3259', 'obsolete3243', 'tiny_only', 'missing_rank',
    'wrong_source', 'dirty_cleanup', 'gpu_remaining', 'wrong_guard', 'wrong_adam', 'old_revision', 'replay_diff', 'review_diff'])
def test_actual_native_gates_reject_partial_old_or_unaccepted_evidence(mutation):
    a,r,v,b,g,t,p = copy.deepcopy(native_fixture())
    if mutation == 'null': a=None
    elif mutation in ('failed3259','obsolete3243'):
        a['job_id']=r['job']=v['job_id']='3259' if mutation=='failed3259' else '3243'
    elif mutation=='tiny_only': r['phases']=[]
    elif mutation=='missing_rank': r['phases'][1].pop()
    elif mutation=='wrong_source': r['code_commit']='0'*40
    elif mutation=='dirty_cleanup': r['cleanup_residual_states']={'123':'S'}
    elif mutation=='gpu_remaining': r['final_gpu']['processes']=['foreign']
    elif mutation=='wrong_guard': r['guard_identity']['resource_guard']['sha256']='0'*64;g=copy.deepcopy(native_fixture()[4])
    elif mutation=='wrong_adam': r['phases'][0][0]['optimizer_steps_after']=520
    elif mutation=='old_revision': r['phases'][0][0]['official_revision']='base'
    elif mutation=='replay_diff': r['phases'][1][0]['next_mini_identity']['model']='changed'
    else: v['result_sha256']='0'*64
    with pytest.raises(ValueError): validate_native_payload(a,r,v,b,g,t,p)


def test_complete_native_fixture_only_passes_schema_not_real_gpu():
    validate_native_payload(*native_fixture())


@pytest.mark.parametrize('mutation', ['review_failed','review_exit','review_cgroup_false','review_cgroup_missing',
    'review_terminal_sha','review_physical_sha','review_terminal_requeue','review_terminal_restarts',
    'terminal_missing','terminal_failed','terminal_exit','terminal_job','terminal_node','terminal_requeue','terminal_restarts',
    'terminal_path','physical_missing','physical_other_job','physical_other_run','physical_result_diff','physical_terminal_diff',
    'physical_own_cgroups','physical_own_queue','release_missing','release_unreadable','release_stat_exists','release_scope_exists',
    'release_removed_false','release_other_job','release_other_scope','release_other_ancestor','physical_gpu_unreadable',
    'raw_terminal_missing','raw_terminal_corrupt','raw_terminal_bytes','raw_terminal_sha','raw_result_missing',
    'physical_result_sha','physical_duplicate_path','native_binding_guard_diff',
    'acceptance_terminal_missing','acceptance_failed','acceptance_exit','acceptance_requeue','acceptance_restarts','acceptance_cgroup_false'])
def test_cd1_requires_actual_terminal_and_cgroup_originals(mutation):
    a,r,v,b,g,t,p=copy.deepcopy(native_fixture())
    if mutation=='review_failed':v['terminal_state']='FAILED'
    elif mutation=='review_exit':v['terminal_exit_code']='1:0'
    elif mutation=='review_cgroup_false':v['cgroup_removed']=False
    elif mutation=='review_cgroup_missing':v.pop('cgroup_removed')
    elif mutation=='review_terminal_sha':v['terminal_sha256']='0'*64
    elif mutation=='review_physical_sha':v['physical_payload_sha256']='0'*64
    elif mutation=='review_terminal_requeue':v['terminal_requeue']=1
    elif mutation=='review_terminal_restarts':v['terminal_restarts']=1
    elif mutation=='terminal_missing':t={}
    elif mutation=='terminal_failed':t['fields']['JobState']='FAILED'
    elif mutation=='terminal_exit':t['fields']['ExitCode']='1:0'
    elif mutation=='terminal_job':t['fields']['JobId']='9998'
    elif mutation=='terminal_node':t['fields']['NodeList']='h100-11'
    elif mutation=='terminal_requeue':t['fields']['Requeue']='1'
    elif mutation=='terminal_restarts':t['fields']['Restarts']='1'
    elif mutation=='terminal_path':a['terminal_path']=a['terminal_path'].replace('9999','9998')
    elif mutation=='physical_missing':p={}
    elif mutation=='physical_other_job':p['job']='9998'
    elif mutation=='physical_other_run':p['run']=p['run'].replace('9999','9998')
    elif mutation=='physical_result_diff':p['result']['job']='9998'
    elif mutation=='physical_terminal_diff':p['terminal']['fields']['JobId']='9998'
    elif mutation=='physical_own_cgroups':p['own_cgroups']=[r['cgroup']]
    elif mutation=='physical_own_queue':p['own_job_queue']='9999 COMPLETING'
    elif mutation=='release_missing':p.pop('cgroup_release')
    elif mutation=='release_unreadable':p['cgroup_release']['ancestor_readable']=False
    elif mutation=='release_stat_exists':p['cgroup_release']['stat_outcome']='EXISTS'
    elif mutation=='release_scope_exists':p['cgroup_release']['scope_stat_outcome']='EXISTS'
    elif mutation=='release_removed_false':p['cgroup_release']['removed']=False
    elif mutation=='release_other_job':p['cgroup_release']['job_id']='9998'
    elif mutation=='release_other_scope':p['cgroup_release']['result_cgroup']=r['cgroup'].replace('9999','9998')
    elif mutation=='release_other_ancestor':p['cgroup_release']['ancestor_path']='/sys/fs/cgroup/absent'
    elif mutation=='physical_gpu_unreadable':p['current_compute']['returncode']=1
    elif mutation=='raw_terminal_missing':p['originals'].pop(1)
    elif mutation=='raw_terminal_corrupt':p['originals'][1]['content_base64']='%%invalid'
    elif mutation=='raw_terminal_bytes':p['originals'][1]['bytes']+=1
    elif mutation=='raw_terminal_sha':p['originals'][1]['sha256']='0'*64
    elif mutation=='raw_result_missing':p['originals'].pop(0)
    elif mutation=='physical_result_sha':p['physical'][0]['sha256']='0'*64
    elif mutation=='physical_duplicate_path':p['physical']+=p['physical']
    elif mutation=='native_binding_guard_diff':b['resource_guard']['sha256']='0'*64
    elif mutation=='acceptance_terminal_missing':a.pop('terminal_state')
    elif mutation=='acceptance_failed':a['terminal_state']='FAILED'
    elif mutation=='acceptance_exit':a['terminal_exit_code']='1:0'
    elif mutation=='acceptance_requeue':a['terminal_requeue']=1
    elif mutation=='acceptance_restarts':a['terminal_restarts']=1
    elif mutation=='acceptance_cgroup_false':a['cgroup_removed']=False
    with pytest.raises((ValueError,KeyError,TypeError)):validate_native_payload(a,r,v,b,g,t,p)


def actual_bound_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(contract,'ROOT',tmp_path)
    a,r,v,b,g,t,p=copy.deepcopy(native_fixture(tmp_path))
    for record in p['originals']:
        path=Path(record['path']);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(base64.b64decode(record['content_base64']))
    physical=Path(a['physical_payload_path']);physical.write_bytes(gzip.compress(json.dumps(p).encode(),mtime=0))
    a['physical_payload_sha256']=v['physical_payload_sha256']=sha(physical)
    review=tmp_path/'logs/independent-native-review.json';review.write_text(json.dumps(v))
    a.update(review_path=str(review),review_sha256=sha(review))
    return a,r,v,b,g,t,p


def test_reads_all_native_actual_bound_json_and_gzip_proofs(tmp_path,monkeypatch):
    a,r,v,b,g,t,p=actual_bound_fixture(tmp_path,monkeypatch)
    assert read_native_acceptance(a,g)==r
    # Unrelated current activity remains recorded, never misreported as residue.
    p['current_compute']['stdout']='7777 unrelated-process 2048 MiB'
    p['current_node_queue']='7778 other-task RUNNING'
    validate_native_payload(a,r,v,b,g,t,p)


@pytest.mark.parametrize('proof', ['result','terminal','physical_payload','review','native_binding'])
def test_changed_or_missing_bound_proof_refuses_before_launch(tmp_path,monkeypatch,proof):
    a,r,v,b,g,t,p=actual_bound_fixture(tmp_path,monkeypatch)
    target=Path(a[proof+'_path'])
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Missing/changed proof'):read_native_acceptance(a,g)
    target.unlink()
    with pytest.raises(ValueError,match='Missing/changed proof'):read_native_acceptance(a,g)


def test_exact_cd1_failed_review_with_other_pass_fields_is_rejected(tmp_path,monkeypatch):
    a,r,v,b,g,t,p=actual_bound_fixture(tmp_path,monkeypatch)
    v.update(terminal_state='FAILED',terminal_exit_code='1:0',cgroup_removed=False)
    path=Path(a['review_path']);path.write_text(json.dumps(v));a['review_sha256']=sha(path)
    with pytest.raises(ValueError,match='independent terminal'):read_native_acceptance(a,g)


def isolated_candidate(g):
    runtime=[dict(path=str(HELPERS/('fixture-'+str(i)+'.py')),bytes=i,sha256='0'*64) for i in range(13)]
    return dict(status='CANDIDATE_NOT_READY',source_native_acceptance=None,code_commit=SOURCE,helpers=str(HELPERS),
        controls=str(CONTROLS),runtime_helpers=runtime,node='h100-8',repo=str(contract.ROOT/'repo-policy-safety-52befe9'),
        validation_scope=NATIVE_SCOPE,calibration_scope=SCOPE,allowed_safety_modes=['calibration'],
        calibration_phase='probability_probe_then_wait',allowed_calibration_phases=['probability_probe_then_wait'],
        restore_checkpoint=CHECKPOINT,restore_manifest_sha256=MANIFEST,restore_step=25,common_manifest_sha256=DATA,
        calibration_actual_outer_budget=1,optimizer_steps_per_outer=10,retained_training_updates_contributed=0,
        target_old_policy_kl=None,negative_tail_limit=None,probability_tolerance_evidence=[],
        actor_recompute_abs_tolerance=None,actor_vllm_abs_tolerance=None,actor_recompute_joint_abs_tolerance=None,
        actor_vllm_joint_abs_tolerance=None,prefix_mismatch_limit=0,action_token_mismatch_limit=0,
        training_submitted=False,calibration_submitted=False,files=runtime+[dict(r,bytes=1) for r in g.values()],evidence=[],**g)


def test_frozen_ready_retains_actual_terminal_originals_and_independent_acceptance(tmp_path,monkeypatch):
    from freeze_ready_v4 import freeze
    import freeze_ready_v4 as freezer
    a,r,v,b,g,t,p=actual_bound_fixture(tmp_path,monkeypatch)
    candidate=isolated_candidate(g);validate_candidate(candidate)
    candidate_path=tmp_path/'candidate.json';candidate_path.write_text(json.dumps(candidate))
    physical=dict(candidate_sha256=sha(candidate_path),code_commit=SOURCE,
        status='PHYSICAL_CODE_PROOF_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU',
        metadata=[dict(path=x['path'],bytes=x['bytes']) for x in candidate['files']],evidence=[],pad_token_id=151643)
    physical_path=tmp_path/'calibration-physical.json';physical_path.write_text(json.dumps(physical))
    review=dict(status='PASS',scope=SCOPE,candidate_sha256=sha(candidate_path),physical_sha256=sha(physical_path),
        freeze_helper_sha256=sha(freezer.__file__),native_result_sha256=a['result_sha256'],native_review_sha256=a['review_sha256'],
        native_terminal_sha256=a['terminal_sha256'],native_physical_payload_sha256=a['physical_payload_sha256'])
    review_path=tmp_path/'deployment-review.json';review_path.write_text(json.dumps(review))
    spec=tmp_path/'native-acceptance-spec.json';spec.write_text(json.dumps(a))
    ready_path=tmp_path/'ready.json'
    freeze(str(candidate_path),sha(candidate_path),str(physical_path),sha(physical_path),str(review_path),sha(review_path),str(spec),str(ready_path))
    ready=json.loads(ready_path.read_text());validate_ready(ready)
    records={e['path']:e for e in ready['evidence']}
    for name in ('result','review','native_binding','terminal','physical_payload'):
        assert records[a[name+'_path']]['sha256']==a[name+'_sha256']
        assert records[a[name+'_path']]['bytes']==Path(a[name+'_path']).stat().st_size
    # This is a CPU-only synthetic fixture, not a native or deployment receipt.
    assert ready['source_native_acceptance']['job_id']=='9999'


@pytest.mark.parametrize('mutation',['native_helper_directory','native_runtime_copy_path','wrong_control_directory','missing_runtime_copy'])
def test_calibration_code_cannot_be_added_to_frozen_native_runtime(mutation):
    g=native_fixture()[4];c=isolated_candidate(g)
    if mutation=='native_helper_directory':c['helpers']=str(NATIVE_HELPERS)
    elif mutation=='native_runtime_copy_path':c['runtime_helpers'][0]['path']=str(NATIVE_HELPERS/'calibration_driver_v1.py')
    elif mutation=='wrong_control_directory':c['controls']=str(NATIVE_HELPERS)
    else:c['files'].pop(0)
    with pytest.raises(ValueError):validate_controls(c)


def submit_fixture(tmp_path,monkeypatch,answer='9998\n',returncode=0,queue=''):
    p=Path(__file__).with_name('submit_once_v4.py')
    spec=importlib.util.spec_from_file_location('submit_unit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'INTENT',tmp_path/'intent.json');monkeypatch.setattr(m,'RECEIPT',tmp_path/'receipt.json')
    # Submission control is isolated here; actual acceptance schema is tested above.
    monkeypatch.setattr(m,'validate_ready',lambda _:None)
    binding=tmp_path/'binding.json';binding.write_text(json.dumps(dict(files=[],evidence=[],repo='/unit-only',helpers=str(HELPERS),
        source_native_acceptance=dict(job_id='9999'))))
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(status='PASS',scope=SCOPE,binding_sha256=sha(binding),submit_helper_sha256=sha(p))))
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:queue if args[0]=='squeue' else '')
    calls=[]
    def once(args,**kwargs):calls.append(args);return subprocess.CompletedProcess(args,returncode,answer,'')
    monkeypatch.setattr(m.subprocess,'run',once)
    return m,calls,(str(binding),sha(binding),str(review),sha(review))


def test_submit_claim_success_blocks_every_repeat(tmp_path,monkeypatch):
    m,calls,args=submit_fixture(tmp_path,monkeypatch);m.main(*args)
    assert json.loads(m.RECEIPT.read_text())['job_id']=='9998'
    with pytest.raises(ValueError):m.main(*args)
    assert len(calls)==1


def test_uncertain_submit_retains_claim_and_rejects_retry(tmp_path,monkeypatch):
    m,calls,args=submit_fixture(tmp_path,monkeypatch,answer='transport timeout',returncode=1)
    with pytest.raises(ValueError):m.main(*args)
    assert m.INTENT.is_file() and m.RECEIPT.is_file()
    with pytest.raises(ValueError):m.main(*args)
    assert len(calls)==1


def test_sbatch_exception_preserves_intent_and_failure_receipt(tmp_path,monkeypatch):
    m,calls,args=submit_fixture(tmp_path,monkeypatch)
    def failing(*args,**kwargs):calls.append(args);raise subprocess.TimeoutExpired('unit',60)
    monkeypatch.setattr(m.subprocess,'run',failing)
    with pytest.raises(subprocess.TimeoutExpired):m.main(*args)
    assert m.INTENT.is_file() and json.loads(m.RECEIPT.read_text())['status']=='UNCERTAIN_OR_FAILED_NO_RETRY'
    with pytest.raises(ValueError):m.main(*args)
    assert len(calls)==1


def test_occupied_node_does_not_claim_or_submit(tmp_path,monkeypatch):
    m,calls,args=submit_fixture(tmp_path,monkeypatch,queue='other RUNNING')
    with pytest.raises(ValueError):m.main(*args)
    assert not m.INTENT.exists() and not m.RECEIPT.exists() and not calls
