"""CPU boundary fixtures; never an actual first5 acceptance or release."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import publish_first5_release as tool


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return tool.cpu.binding(path)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, 'ROOT', tmp_path)
    monkeypatch.setattr(tool, 'HERE', tmp_path/'tools/deployment')
    job, token = '9999', 'a' * 32
    run = tmp_path / 'runs' / ('formal-aligned-em200-' + token + '-' + job)
    run.mkdir(parents=True)
    (tmp_path / 'evidence').mkdir()
    (run / 'logs').mkdir()
    source = write(tmp_path / 'tools/deployment/formal.py', {'CPU_FIXTURE': True})
    submit = dict(source=tool.SOURCE, root=str(tmp_path), job=job, token=token, run=str(run),
        comment='BAPO_ALIGNED_EM200:formal:' + tool.SOURCE + ':' + token,
        scope='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION', files={'formal.py': source['sha256']})
    submission = write(tmp_path / 'scheduler-head/formal-submit-receipt.json', submit)
    config = write(run / 'config/resolved-config.json', {'trainer': {'fresh200_gate_receipt': str(run / 'logs/fresh200-step5-independent-release.json')},
        'actor_rollout_ref': {'model': {'path': str(tmp_path / 'official')}}})
    manifest = write(run / 'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json',
        {'files': [config], 'rank_roundtrips': [{'pid': 1000 + rank} for rank in range(8)]})
    process = lambda pid: dict(pid=pid,start_ticks=pid+100,command='CPU_FIXTURE_NOT_TARGET',uids=[0]*4,
        executable='CPU_FIXTURE',selected_env={'BAPO_RUN_ID':str(run),'SLURM_JOB_ID':job},state='S')
    processes = [process(123)] + [process(1000+r) for r in range(8)]
    live = run / 'train.log'
    live.write_text(''.join('FRESH200_ATTEMPT_JSON={}\n' for _ in range(5)) + 'FRESH200_STEP5_PAUSED_JSON=' + json.dumps(
        {'pid':123,'manifest_path':manifest['path'],'manifest_sha256':manifest['sha256']}) + '\n')
    prefix = {'CPU_FIXTURE_ONLY': True}
    hf_files = [write(run / 'checkpoints/actor/global_step_5' / ('fixture' + str(i)), {'CPU_FIXTURE': i}) for i in range(12)]
    official_rows = [{'file':'fixture'+str(i),'sha256':'0'*64,'bytes':1} for i in range(12)]
    model_manifest = write(tmp_path / 'official/VERIFIED-MODEL-MANIFEST.json', {'files':official_rows})
    monkeypatch.setattr(tool.cpu, 'MODEL_SHA', model_manifest['sha256'])
    monkeypatch.setattr(tool.cpu, 'official_model_path', lambda p: tmp_path/'official')
    monkeypatch.setattr(tool.cpu, 'process_identity', lambda pid,run,job: process(pid))
    first = dict(manifest=manifest,source_commit=tool.SOURCE,run_root=str(run),job=job,token=token,
        train_log_prefix=prefix,pause_nonce='b'*64,config=json.loads(Path(config['path']).read_text()),
        original_member_bindings=[config],native_roundtrip_reports=[{'pid':1000+r} for r in range(8)],processes=processes,driver_pid=123)
    proof = dict(status='CPU_EVIDENCE_COLLECTED_NOT_INDEPENDENTLY_REVIEWED_NOT_RELEASE',acceptance=False,release_receipt=False,
        source_commit=tool.SOURCE,root=str(tmp_path),issues=[],actual_submission=submission,
        runtime=dict(platform='linux',cuda_visible_devices='',cuda_available=False,cuda_initialized=False),first5=first,
        cpu_hf=dict(status='ACTUAL_CPU_HF_CHECKED_NOT_RELEASE',scope='official',parameters=3085938688,tied_input_output_storage=True,
            all_parameters_finite=True,forward=dict(device='cpu',finite=True),parameter_delta=dict(all_official_and_export_tensors_finite=True,changed_elements=1),loading_info={}),
        hf_files=hf_files,official_model=dict(manifest=model_manifest,files=[dict(path=str(tmp_path/'official'/r['file']),sha256=r['sha256'],bytes=r['bytes'])for r in official_rows]),
        formal_tool_bindings=[source],helper_files=[write(tmp_path/'tools/deployment'/name,{'CPU_FIXTURE':name}) for name in ('aligned_first5_cpu_evidence.py','aligned_first5_math.py','aligned-first5-math-provenance.json')])
    proof['helper_sha256']=proof['helper_files'][0]['sha256']
    proof_path = tmp_path / 'evidence/cpu-evidence.json'
    cpu_binding = write(proof_path, proof)
    review = write(tmp_path / 'evidence/independent-review.json', {'scope':'CPU_FIXTURE_NOT_PRODUCTION_REVIEW'})
    receipt = dict(publication_schema=tool.SCHEMA,source_commit=tool.SOURCE,baseline_aligned=True,run_root=str(run),
        formal_submission=submission,independent_review=review,cpu_evidence=cpu_binding,train_log_prefix=prefix,
        pause_nonce='b'*64,driver_birth=tool.stable_process(processes[0]))
    receipt_path = tmp_path / 'evidence/external-release.json'
    write(receipt_path, receipt)
    calls=[]
    def validate(path, manifest_path, repo, expected_driver_pid):
        calls.append((path,manifest_path,repo,expected_driver_pid))
        return json.loads(Path(path).read_text())
    return SimpleNamespace(run=run,proof=proof,proof_path=proof_path,receipt=receipt,receipt_path=receipt_path,
        report_sha=review['sha256'],cpu_sha=cpu_binding['sha256'],gate=SimpleNamespace(validate_release=validate),calls=calls,
        manifest=Path(manifest['path']),submit=submit,process=process)


def check(f):
    return tool.validate_candidate(f.receipt_path,tool.cpu.digest(f.receipt_path),f.report_sha,f.cpu_sha,f.gate)


def test_reviewable_prepare_publish_exact_bytes_exclusive(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    plan=tool.prepare(f.receipt_path,tool.cpu.digest(f.receipt_path),f.report_sha,f.cpu_sha,tmp_path/'evidence/plan',f.gate)
    target=Path(plan['identity']['target'])
    assert not target.exists()
    plan_path=tmp_path/'evidence/plan/release-plan.json'
    result=tool.publish(plan_path,tool.cpu.digest(plan_path),plan['payload']['sha256'],f.gate)
    assert result['status']=='EXTERNAL_REVIEWED_RECEIPT_PUBLISHED_NOT_TRAINING_COMPLETION'
    assert target.read_bytes()==f.receipt_path.read_bytes()
    assert f.calls[-1]==(str(f.receipt_path),f.manifest,tmp_path/'repo',123)
    assert len(f.calls)==2
    assert tool.publish(plan_path,tool.cpu.digest(plan_path),plan['payload']['sha256'],f.gate)['status']=='IDENTICAL_RECEIPT_ALREADY_PRESENT_NO_WRITE'
    target.write_text('existing independent different receipt')
    with pytest.raises(ValueError,match='never overwrite'):tool.publish(plan_path,tool.cpu.digest(plan_path),plan['payload']['sha256'],f.gate)
    assert target.read_text()=='existing independent different receipt'


@pytest.mark.parametrize('job',['3315','3345'])
def test_historical_or_hardware_job_cannot_publish(tmp_path,monkeypatch,job):
    f=fixture(tmp_path,monkeypatch)
    f.submit['job']=job
    write(tmp_path/'scheduler-head/formal-submit-receipt.json',f.submit)
    with pytest.raises(ValueError,match='new actual formal job'):check(f)
    assert not f.calls


@pytest.mark.parametrize('mutation',['hf','report','cpu','source','birth','incomplete'])
def test_changed_originals_and_missing_actual_cpu_fail_before_gate(tmp_path,monkeypatch,mutation):
    f=fixture(tmp_path,monkeypatch)
    if mutation=='hf':Path(f.proof['hf_files'][0]['path']).write_text('tampered')
    elif mutation=='report':Path(f.receipt['independent_review']['path']).write_text('tampered')
    elif mutation=='cpu':f.proof_path.write_text('tampered')
    elif mutation=='source':f.receipt['source_commit']='60f566';write(f.receipt_path,f.receipt)
    elif mutation=='birth':monkeypatch.setattr(tool.cpu,'process_identity',lambda pid,run,job:dict(f.process(pid),start_ticks=999))
    else:
        f.proof['cpu_hf']['scope']='CPU_FIXTURE_ONLY'
        f.receipt['cpu_evidence']=write(f.proof_path,f.proof);f.cpu_sha=f.receipt['cpu_evidence']['sha256'];write(f.receipt_path,f.receipt)
    with pytest.raises(ValueError):check(f)
    assert not f.calls


def test_actual_source080_gate_rejects_incomplete_gpu_fixture_without_publication(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    from search_r1.baselines import fresh200_gate
    assert Path(fresh200_gate.__file__).resolve().name=='fresh200_gate.py'
    f.gate=fresh200_gate
    with pytest.raises(ValueError,match='profile-specific'):check(f)
    assert not (f.run/'logs/fresh200-step5-independent-release.json').exists()


def test_driver_rechecked_after_actual_gate_work(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    original=f.gate.validate_release
    def expires(*args,**kwargs):
        result=original(*args,**kwargs)
        monkeypatch.setattr(tool.cpu,'process_identity',lambda pid,run,job:dict(f.process(pid),start_ticks=999))
        return result
    f.gate.validate_release=expires
    with pytest.raises(ValueError,match='changed before publication'):check(f)


def test_atomic_no_partial_overwrite_and_symlink_refusal(tmp_path):
    target=tmp_path/'gate.json'
    raw=b'{"CPU_FIXTURE_NOT_RELEASE":true}\n'
    tool.atomic_publish(target,raw)
    assert target.read_bytes()==raw
    with pytest.raises(FileExistsError):tool.atomic_publish(target,b'different')
    assert target.read_bytes()==raw
    link=tmp_path/'link.json';link.symlink_to(target)
    with pytest.raises(ValueError,match='canonical'):tool.atomic_publish(link,b'different')
    assert target.read_bytes()==raw
    assert not list(tmp_path.glob('.first5-release-*'))


def test_hf_change_during_gate_and_plan_payload_tamper_are_rejected(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    original=f.gate.validate_release
    def changes(*args,**kwargs):
        result=original(*args,**kwargs)
        Path(f.proof['hf_files'][0]['path']).write_text('changed while native gate hashed originals')
        return result
    f.gate.validate_release=changes
    with pytest.raises(ValueError,match='changed during gate'):check(f)
    assert not (f.run/'logs/fresh200-step5-independent-release.json').exists()


def test_reviewed_payload_and_plan_sha_are_mandatory(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    plan=tool.prepare(f.receipt_path,tool.cpu.digest(f.receipt_path),f.report_sha,f.cpu_sha,tmp_path/'evidence/plan',f.gate)
    plan_path=tmp_path/'evidence/plan/release-plan.json'
    expected=tool.cpu.digest(plan_path)
    with pytest.raises(ValueError,match='reviewed bytes changed'):tool.publish(plan_path,'0'*64,plan['payload']['sha256'],f.gate)
    Path(plan['payload']['path']).write_text('different payload')
    with pytest.raises(ValueError,match='reviewed bytes changed'):tool.publish(plan_path,expected,plan['payload']['sha256'],f.gate)
    assert not Path(plan['identity']['target']).exists()


@pytest.mark.parametrize('change',['manifest','live_released','live_sixth'])
def test_gate_cannot_mutate_original_manifest_or_resume_driver_during_work(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch)
    original=f.gate.validate_release
    def changes(*args,**kwargs):
        result=original(*args,**kwargs)
        if change=='manifest':f.manifest.write_text('changed original manifest after gate returned')
        else:
            with (f.run/'train.log').open('a') as handle:
                handle.write(('FRESH200_STEP5_RELEASED_JSON=' if change=='live_released' else 'FRESH200_ATTEMPT_JSON=')+'{}\n')
        return result
    f.gate.validate_release=changes
    with pytest.raises(ValueError):check(f)
    assert not (f.run/'logs/fresh200-step5-independent-release.json').exists()
