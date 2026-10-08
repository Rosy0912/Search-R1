"""CPU-only publisher checks. No network, RSA signing, approval or GPU is executed."""
import copy
import contextlib
import importlib.util
import io
import os
import re
import shlex
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('author200_gpu_init_draft',HERE/'publish_gpu_init.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def ready_review():
    run=m.ROOT/'runs/formal-123'
    ready={'source_commit':m.SOURCE,'job_id':'123','uid':0,'pid':345,
           'process_start_ticks':'123456','run_root':str(run),'nonce':'a'*64,
           'binding_sha256':'b'*64,'public_key_sha256':m.PUBLIC_SHA,
           'nvml_uuids':['GPU-'+str(i) for i in range(8)]}
    review={'status':m.REVIEW_STATUS,'source':m.SOURCE,'job_id':'123','wrapper_pid':345,
            'process_start_ticks':'123456','nonce':'a'*64,'ready_sha256':'c'*64,
            'binding_sha256':'b'*64,'public_key_sha256':m.PUBLIC_SHA,
            'actual_profile_sha256':'e'*64,'actual_import_proof_sha256':'f'*64,
            'native_closure_sha256':'d'*64,'uuids':ready['nvml_uuids'],
            'all_seven_binding_hashes':{r:'d'*64 for r in m.ROLES}}
    return ready,review,run


def test_actual_review_identity_not_markdown_pass():
    ready,review,run=ready_review()
    m.validate_review(review,ready,'c'*64,'123',run)
    for key,value in [('status','PASS'),('job_id','124'),('source','0'*40),
                      ('wrapper_pid',True),('process_start_ticks','stale'),
                      ('nonce','0'*64),('ready_sha256','0'*64),
                      ('native_closure_sha256','0'*64),('uuids',ready['nvml_uuids'][:-1])]:
        bad=copy.deepcopy(review);bad[key]=value
        with pytest.raises(ValueError):m.validate_review(bad,ready,'c'*64,'123',run)
    for key in ('actual_profile_sha256','actual_import_proof_sha256'):
        for value in (None,True,'not-a-pin','0'*63):
            bad=copy.deepcopy(review);bad[key]=value
            with pytest.raises(ValueError):m.validate_review(bad,ready,'c'*64,'123',run)


def test_duplicate_nonfinite_review_and_key_material_rejected():
    for raw in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}']:
        with pytest.raises(ValueError):m.unique(raw)
    assert "input=canonical(payload)" in (HERE/'publish_gpu_init.py').read_text()
    assert 'LOCAL_KEY.read_bytes' not in (HERE/'publish_gpu_init.py').read_text()


def test_remote_program_regex_and_byte_arguments_compile_correctly():
    code=m.live_code({'scope':'CPU_ONLY'},'print("fixture")')
    compile(code,'CPU_PROGRAM_FIXTURE','exec')
    assert re.search(r'^Uid:\s+(.*)','Uid:\t0\t0\t0\t0',re.M) is not None
    assert 'split(b\'\\0\')' in code
    assert 'query-compute-apps=pid,gpu_uuid,used_gpu_memory' in code
    assert 'query-compute-apps' in code and 'not apps.strip()' in code
    assert 'gpu-init-approval.sha256' in code and 'pre-gpu-import-check.json' in code
    assert "hashfile(profilepath)==D['actual_profile_sha256']" in code
    assert "hashfile(proofpath)==D['actual_import_proof_sha256']" in code
    for forbidden in ('pkill','ray stop','scancel','sbatch','CUDA_VISIBLE_DEVICES=0'):
        assert forbidden not in code


def atomic_fixture_namespace():
    # Only the fixture UID predicate differs: execution user is the local test
    # owner, while production remains fixed UID0. Link/write/fsync/no-follow
    # publication semantics are identical, and no actual task file is touched.
    namespace={}
    fixture=m.ATOMIC_PUBLISH.replace('parent.stat().st_uid==0','parent.stat().st_uid==os.getuid()')
    exec('def need(ok,msg):\n if not ok:raise ValueError(msg)\n'+fixture,namespace)
    return namespace


def test_atomic_noclobber_and_symlink_cannot_replace_original(tmp_path):
    ns=atomic_fixture_namespace();path=tmp_path/'approval.json'
    ns['exclusive'](path,b'CPU_UNIT_FIXTURE_NOT_APPROVAL')
    assert path.read_bytes()==b'CPU_UNIT_FIXTURE_NOT_APPROVAL'
    with pytest.raises(FileExistsError):ns['exclusive'](path,b'overwrite')
    assert path.read_bytes()==b'CPU_UNIT_FIXTURE_NOT_APPROVAL'
    target=tmp_path/'target';target.write_bytes(b'preserve')
    link=tmp_path/'alias';link.symlink_to(target)
    with pytest.raises(FileExistsError):ns['exclusive'](link,b'overwrite')
    assert target.read_bytes()==b'preserve' and link.is_symlink()
    assert not list(tmp_path.glob('.root-approval-*.tmp'))


def test_symlink_parent_rejected(tmp_path):
    ns=atomic_fixture_namespace();real=tmp_path/'real';real.mkdir();alias=tmp_path/'alias';alias.symlink_to(real)
    with pytest.raises(ValueError):ns['exclusive'](alias/'approval',b'fixture')
    assert list(real.iterdir())==[]


def test_review_publish_program_only_transmits_public_review_bytes(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr(m,'remote',lambda command,data:captured.append((command,data)))
    m.publish_review(tmp_path/'review.json',b'{"scope":"CPU_ONLY"}')
    command,data=captured[0];tokens=shlex.split(command)
    assert tokens[:2]==['python3','-c']
    compile(tokens[2],'REMOTE_REVIEW_PROGRAM_FIXTURE','exec')
    assert data==b'{"scope":"CPU_ONLY"}' and str(m.LOCAL_KEY) not in command
    assert 'os.link(temp,path,follow_symlinks=False)' in tokens[2]


def test_final_generated_program_emits_real_newline_and_pin_last(tmp_path,monkeypatch):
    envelope=b'CPU_UNIT_FIXTURE_NOT_APPROVAL'
    bindings=tmp_path/'bindings';bindings.mkdir()
    code=m.final_publication_code({'run_root':str(tmp_path)},envelope)
    compile(code,'REMOTE_FINAL_PROGRAM_FIXTURE','exec')
    start=code.index('\nstate=validate_live()')
    namespace=atomic_fixture_namespace()
    namespace['D']={'run_root':str(tmp_path)}
    namespace['validate_live']=lambda:{'scope':'CPU_UNIT_FIXTURE'}
    monkeypatch.setattr(namespace['sys'],'stdin',io.TextIOWrapper(io.BytesIO(envelope)))
    events=[];real_exclusive=namespace['exclusive']
    def observe(path,raw):
        events.append(path.name);real_exclusive(path,raw)
    namespace['exclusive']=observe
    with contextlib.redirect_stdout(io.StringIO()):exec(code[start:],namespace)
    assert events==['gpu-init-approval.json','gpu-init-approval.sha256']
    assert (bindings/'gpu-init-approval.json').read_bytes()==envelope
    assert (bindings/'gpu-init-approval.sha256').read_bytes()==m.sha(envelope).encode()+b'\n'


def test_actual_observed_ready_review_joins_without_network_or_signing():
    review=m.unique((HERE/'ACTUAL3261_PRE_GPU_INDEPENDENT_REVIEW.json').read_bytes())
    candidates=list((HERE/'actual-snapshots').glob('3261-*/original-files/approval-ready.json'))
    assert len(candidates)==1
    ready_raw=candidates[0].read_bytes();ready=m.unique(ready_raw)
    m.validate_review(review,ready,m.sha(ready_raw),'3261',m.ROOT/'runs/formal-3261')
    for key,filename in [('actual_profile_sha256','evidence/pre-gpu-profile.json'),
                         ('actual_import_proof_sha256','evidence/pre-gpu-import-check.json')]:
        assert review[key]==m.sha((candidates[0].parent/filename).read_bytes())


def test_approval_only_scope_and_no_private_key_transport():
    text=(HERE/'publish_gpu_init.py').read_text()
    assert "EXPLICIT_OPSD_E1_GPU_INITIALIZATION_ONLY" in text
    assert "PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW" not in text
    assert 'inspect actual remote original artifacts' not in text.lower() or 'retry' in text.lower()
    assert "exclusive(r/'bindings/gpu-init-approval.json',raw)" in text
    assert "exclusive(r/'bindings/gpu-init-approval.sha256'" in text
