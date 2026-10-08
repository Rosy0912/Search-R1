import importlib.util,json,os,sys,types
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import context_policy as p
CALLER=Path(__file__).parents[1]/'topd-32b-native-score-caller-v1-readonly-review/caller_contracts.py'

def require(x,msg):
    if not x:raise RuntimeError(msg)

def policy():return p.policies(CALLER,types.SimpleNamespace(require=require,PARAMETERS=32763876352))

def test_exact_native_policy_reused_no_alternate_schema():
    fn=policy()['authorization'];ident=dict(pid=30,starttime_ticks=91)
    receipt=dict(phase='CONTEXT_ONLY',pid=30,starttime_ticks=91,job='33',provenance_sha256='real-bound')
    fn(receipt,'CONTEXT_ONLY',ident,'real-bound',33)
    receipt['phase']='SCORING_ONLY'
    with pytest.raises(RuntimeError):fn(receipt,'CONTEXT_ONLY',ident,'real-bound',33)

@pytest.mark.parametrize('field',['pid','starttime_ticks','job','provenance_sha256'])
def test_permission_cannot_cross_owner_job_or_teacher(tmp_path,field):
    receipt=dict(phase='CONTEXT_ONLY',pid=30,starttime_ticks=91,job='33',provenance_sha256='real-bound')
    receipt[field]='other';path=tmp_path/'permit';path.write_text(json.dumps(receipt));path.chmod(0o600)
    with pytest.raises(RuntimeError):p.wait_permission(path,policy(),'CONTEXT_ONLY',dict(pid=30,starttime_ticks=91),'real-bound',33)

def test_world_readable_permission_refused(tmp_path):
    path=tmp_path/'permit';path.write_text('{}');path.chmod(0o644)
    with pytest.raises(RuntimeError,match='Unsafe'):p.wait_permission(path,policy(),'CONTEXT_ONLY',{},'real',33)

def test_same_owner_eight_actual_gpu_contexts_required():
    uuids=[str(i) for i in range(8)];r=dict(pid=30,allocators=[dict(device=i,uuid=str(i),allocated_bytes=0,reserved_bytes=0) for i in range(8)],gpu_pid_uuid=[dict(pid=30,uuid=str(i)) for i in range(8)])
    policy()['context_receipt'](r,dict(pid=30),uuids)
    r['gpu_pid_uuid'].pop()
    with pytest.raises(RuntimeError):policy()['context_receipt'](r,dict(pid=30),uuids)

def test_native_policy_source_drift_refused(tmp_path):
    f=tmp_path/'caller.py';f.write_text(CALLER.read_text()+'\n# changed\n')
    with pytest.raises(RuntimeError,match='source drift'):p.policies(f,types.SimpleNamespace(require=require))
