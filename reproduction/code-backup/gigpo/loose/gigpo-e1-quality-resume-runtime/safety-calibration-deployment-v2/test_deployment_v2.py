"""Unit counterexamples only: no remote, Slurm, native or GPU acceptance."""
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from deployment_contract_v2 import *


@pytest.mark.parametrize('mutation', ['native_null', 'one_outer', 'probe_only', 'production_mode',
                                    'production_kl', 'invented_probability_tolerance'])
def test_real_deployment_draft_cannot_launch_or_escape_current_live_review(mutation):
    # Local generated candidate is a draft; these tests never read remote paths.
    path = Path(__file__).resolve().parents[3] / 'outputs/gigpo-e1-policy-safety/CALIBRATION-CANDIDATE-52BEFE9-v2.json'
    candidate = json.loads(path.read_text())
    if mutation == 'native_null':
        with pytest.raises(ValueError):validate_ready(candidate)
        with pytest.raises(ValueError, match='null'):read_native_acceptance(None, candidate)
        return
    if mutation in ('one_outer','probe_only'):candidate['calibration_phase']=mutation
    elif mutation == 'production_mode':candidate['allowed_safety_modes']=['production']
    elif mutation == 'production_kl':candidate['target_old_policy_kl']=.01
    else:candidate['actor_vllm_abs_tolerance']=.01
    with pytest.raises(ValueError):validate_controls(candidate)


def native_fixture():
    guard = dict(resource_guard=dict(path=str(HELPERS/'resource_guard_v2.py'),sha256='a'*64),
                 resource_ownership=dict(path=str(HELPERS/'resource_ownership_v2.py'),sha256='b'*64))
    acceptance = dict(status='PASS', code_commit=SOURCE, job_id='9999', result_sha256='d'*64,
                      native_binding_sha256='e'*64, guard_identity=guard)
    result = dict(status='PASS',job='9999',validation_scope=NATIVE_SCOPE,code_commit=SOURCE,
        restore_manifest_sha256=MANIFEST,cpu_exit=0,gpu_exit=0,fixtures_passed=True,actual_outer_updates=0,
        cleanup_residual_states={},final_gpu=dict(processes=[]),guard_identity=guard,binding_sha256='e'*64,
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
    review = dict(status='PASS',scope=NATIVE_SCOPE,code_commit=SOURCE,job_id='9999',restore_manifest_sha256=MANIFEST,result_sha256='d'*64)
    binding = dict(code_commit=SOURCE,helpers=str(HELPERS))
    return acceptance,result,review,binding,guard


@pytest.mark.parametrize('mutation', ['null', 'failed3259', 'obsolete3243', 'tiny_only', 'missing_rank',
    'wrong_source', 'dirty_cleanup', 'gpu_remaining', 'wrong_guard', 'wrong_adam', 'old_revision', 'replay_diff', 'review_diff'])
def test_actual_native_gates_reject_partial_old_or_unaccepted_evidence(mutation):
    a,r,v,b,g = copy.deepcopy(native_fixture())
    if mutation == 'null': a=None
    elif mutation in ('failed3259','obsolete3243'):
        a['job_id']=r['job']=v['job_id']='3259' if mutation=='failed3259' else '3243'
    elif mutation=='tiny_only': r['phases']=[]
    elif mutation=='missing_rank': r['phases'][1].pop()
    elif mutation=='wrong_source': r['code_commit']='0'*40
    elif mutation=='dirty_cleanup': r['cleanup_residual_states']={'123':'S'}
    elif mutation=='gpu_remaining': r['final_gpu']['processes']=['foreign']
    elif mutation=='wrong_guard': r['guard_identity']['resource_guard']['sha256']='0'*64;g=copy.deepcopy(native_fixture()[-1])
    elif mutation=='wrong_adam': r['phases'][0][0]['optimizer_steps_after']=520
    elif mutation=='old_revision': r['phases'][0][0]['official_revision']='base'
    elif mutation=='replay_diff': r['phases'][1][0]['next_mini_identity']['model']='changed'
    else: v['result_sha256']='0'*64
    with pytest.raises(ValueError): validate_native_payload(a,r,v,b,g)


def test_complete_native_fixture_only_passes_schema_not_real_gpu():
    validate_native_payload(*native_fixture())


def submit_fixture(tmp_path,monkeypatch,answer='9998\n',returncode=0,queue=''):
    p=Path(__file__).with_name('submit_once_v2.py')
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
