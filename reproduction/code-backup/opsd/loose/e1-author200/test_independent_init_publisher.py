"""Independent CPU counterexamples; never use a signer or a remote transport."""
import ast, copy, hashlib, importlib.util, json, os, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('independent_init',HERE/'publish_gpu_init.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
BASE=HERE/'actual-snapshots/3261-20261005T194802469223Z/original-files'
def identity():
    ready=json.loads((BASE/'approval-ready.json').read_text())
    review=json.loads((HERE/'ACTUAL3261_PRE_GPU_INDEPENDENT_REVIEW.json').read_text())
    return ready,review,hashlib.sha256((BASE/'approval-ready.json').read_bytes()).hexdigest()
def test_actual_observed_identity_and_stale_same_pid_rejected():
    ready,review,pin=identity();run=m.ROOT/'runs/formal-3261'
    m.validate_review(review,ready,pin,'3261',run)
    stale=copy.deepcopy(review);stale['process_start_ticks']=str(int(stale['process_start_ticks'])-1)
    with pytest.raises(ValueError):m.validate_review(stale,ready,pin,'3261',run)
    stale=copy.deepcopy(review);stale['job_id']='3210'
    with pytest.raises(ValueError):m.validate_review(stale,ready,pin,'3261',run)
def test_review_cannot_substitute_first5_pass_or_omit_actual_profile():
    ready,review,pin=identity();run=m.ROOT/'runs/formal-3261'
    bad=copy.deepcopy(review);bad['status']='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'
    with pytest.raises(ValueError):m.validate_review(bad,ready,pin,'3261',run)
    for name in ('actual_profile_sha256','actual_import_proof_sha256'):
        bad=copy.deepcopy(review);bad.pop(name)
        with pytest.raises(ValueError):m.validate_review(bad,ready,pin,'3261',run)
def fixture():
    ns={};src=m.ATOMIC_PUBLISH.replace('parent.stat().st_uid==0','parent.stat().st_uid==os.getuid()')
    exec('def need(ok,msg):\n if not ok:raise ValueError(msg)\n'+src,ns)
    return ns
def test_atomic_two_concurrent_writers_exactly_one_wins(tmp_path):
    ns=fixture();barrier=threading.Barrier(2);p=tmp_path/'fixture.json'
    def writer(data):
        barrier.wait()
        try:ns['exclusive'](p,data);return ('win',data)
        except FileExistsError:return ('blocked',data)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(writer,b'CPU_A_NOT_APPROVAL');b=pool.submit(writer,b'CPU_B_NOT_APPROVAL');results=[a.result(),b.result()]
    assert sorted(x[0] for x in results)==['blocked','win']
    assert p.read_bytes()==next(data for state,data in results if state=='win')
    assert not list(tmp_path.glob('.root-approval-*.tmp'))
    assert (p.stat().st_mode & 0o777)==0o600
def test_same_final_symlink_and_dangling_symlink_fail_closed(tmp_path):
    ns=fixture();dst=tmp_path/'dangling';dst.symlink_to(tmp_path/'missing')
    with pytest.raises(FileExistsError):ns['exclusive'](dst,b'CPU_FIXTURE_NOT_APPROVAL')
    assert dst.is_symlink() and not (tmp_path/'missing').exists()
def test_only_local_signer_and_fresh_live_before_final_publication():
    text=(HERE/'publish_gpu_init.py').read_text();tree=ast.parse(text)
    local_key_uses=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='remote':
            assert not any(isinstance(x,ast.Name) and x.id=='LOCAL_KEY' for x in ast.walk(node))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['read_bytes','read_text'] and isinstance(node.func.value,ast.Name):
            assert node.func.value.id!='LOCAL_KEY'
    suffix=m.final_publication_code({'fixture':'CPU_ONLY'},b'CPU_ONLY')
    assert suffix.index('state=validate_live()')<suffix.index("exclusive(r/'bindings/gpu-init-approval.json',raw)")<suffix.index("exclusive(r/'bindings/gpu-init-approval.sha256'")
