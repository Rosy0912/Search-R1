import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest
sys.path.insert(0, str(Path(__file__).parent))
import startup_guard as guard

SEALED = Path(__file__).parents[1] / 'qwen25-32b-eval64/runtime_safety.py'


def snapshot(extra=()):
    ids = {'10': dict(pid=10, starttime_ticks=1, cgroup=['0::/foreign']),
           '20': dict(pid=20, starttime_ticks=2, cgroup=['0::/job'])}
    ids.update({str(p):dict(pid=p,starttime_ticks=p,cgroup=['0::/job']) for p in extra})
    return dict(gpus=[dict(index=str(i),uuid=str(i),used_mib=200,total_mib=10000,sm=0) for i in range(8)],
                applications=[dict(uuid=str(i),pid=p,used_mib=100) for i in range(8) for p in (10,20)+tuple(extra)],
                identities=ids)


def setup(tmp_path, monkeypatch, snapshots, errors=None, **kwargs):
    monkeypatch.setitem(sys.modules, 'eval64', types.SimpleNamespace(require=guard.require))
    spec=importlib.util.spec_from_file_location('test_sealed_runtime', SEALED)
    sealed=importlib.util.module_from_spec(spec); spec.loader.exec_module(sealed)
    baseline=snapshot(); baseline['applications']=[p for p in baseline['applications'] if p['pid']==10]
    baseline['identities'].pop('20')
    runtime=dict(baseline=baseline,retriever_identity=snapshot()['identities']['20'],cgroup='/job')
    iterator=iter(snapshots); current={}
    def read_snapshot():
        current['snapshot']=next(iterator)
        return current['snapshot']
    sealed.nvml_snapshot=read_snapshot
    sealed.pid_identity=lambda pid:current['snapshot']['identities'][str(pid)]
    calls=[]
    def gate(runtime,pre_model=False):
        calls.append(pre_model)
        initial=sealed.nvml_snapshot(); final=sealed.nvml_snapshot()
        if errors:
            raise RuntimeError(errors)
        guard.require(initial['identities']==final['identities'],guard.IDENTITY_CHANGED)
        # Contract fixture only; production invokes actual sealed full SM validator.
        return dict(full_gate_fixture=True)
    sealed.active_gate=gate
    wrapped=guard.StartupGuard(sealed,expected_path=SEALED,trace_path=tmp_path/'trace.jsonl',retry_pause=0,startup_seconds=kwargs.pop('startup_seconds',20),**kwargs)
    wrapped.lease_reader=lambda:set(str(i) for i in range(8))
    wrapped.uid_reader=lambda p:wrapped.owner_uid
    return wrapped,runtime,calls


def test_own_churn_fresh_validation_and_raw_identity(tmp_path,monkeypatch):
    first=snapshot(); added=snapshot((30,)); stable=snapshot((30,))
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[first,added,stable,stable])
    assert wrapped.active_gate(runtime)['full_gate_fixture'] and len(calls)==2
    rows=[json.loads(x) for x in wrapped.trace_path.read_text().splitlines()]
    assert rows[0]['accepted'] is False and rows[0]['retry_authorized'] is True
    assert rows[1]['accepted'] is True
    assert rows[0]['captures'][0]['raw_snapshot']==first
    assert rows[0]['captures'][1]['raw_snapshot']==added


@pytest.mark.parametrize('kind',['foreign','reuse','uid','missingbaseline','unknownuid','memory','badlease'])
def test_dangerous_window_not_retried(tmp_path,monkeypatch,kind):
    first=snapshot(); final=snapshot((30,))
    if kind=='foreign': final['identities']['30']['cgroup']=['0::/other']
    if kind=='reuse': final['identities']['20']['starttime_ticks']=999
    if kind=='missingbaseline':
        final['identities'].pop('10'); final['applications']=[p for p in final['applications'] if p['pid']!=10]
    if kind=='memory':
        for p in final['applications']:
            if p['pid']==10:p['used_mib']=2000
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[first,final])
    if kind=='uid':wrapped.uid_reader=lambda p:wrapped.owner_uid+1 if p==30 else wrapped.owner_uid
    if kind=='unknownuid':wrapped.uid_reader=lambda p:(_ for _ in ()).throw(FileNotFoundError()) if p==30 else wrapped.owner_uid
    if kind=='badlease':wrapped.lease_reader=lambda:(_ for _ in ()).throw(RuntimeError('FD absent'))
    with pytest.raises(RuntimeError,match=guard.IDENTITY_CHANGED):wrapped.active_gate(runtime)
    assert len(calls)==1
    assert not json.loads(wrapped.trace_path.read_text())['retry_authorized']


@pytest.mark.parametrize('condition',['pre_model','expired','closed','other_error','bounded'])
def test_scope_and_budget(tmp_path,monkeypatch,condition):
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[snapshot(),snapshot((30,))],
        errors='Foreign aggregate SM exceeds15% or unknown' if condition=='other_error' else None,
        max_attempts=1 if condition=='bounded' else 6)
    if condition=='expired':wrapped.deadline=0
    if condition=='closed':wrapped.close_startup()
    with pytest.raises(RuntimeError):wrapped.active_gate(runtime,pre_model=condition=='pre_model')
    assert len(calls)==1


def test_spy_same_return_object(tmp_path,monkeypatch):
    original=snapshot(); wrapped,_,_=setup(tmp_path,monkeypatch,[original])
    wrapped.local.captures=[]
    assert wrapped.sealed.nvml_snapshot() is original
    assert wrapped.local.captures[0]['raw_snapshot'] is not original
    del wrapped.local.captures


def test_wrong_path_rejected(tmp_path,monkeypatch):
    wrapped,_,_=setup(tmp_path,monkeypatch,[])
    with pytest.raises(RuntimeError,match='path'):
        guard.StartupGuard(wrapped.sealed,expected_path=tmp_path/'shadow',trace_path=tmp_path/'x')


def test_fresh_success_still_requires_uid(tmp_path,monkeypatch):
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[snapshot(),snapshot((30,)),snapshot((30,)),snapshot((30,))])
    wrapped.uid_reader=lambda p:wrapped.owner_uid+1 if p==30 and len(calls)==2 else wrapped.owner_uid
    with pytest.raises(RuntimeError,match='foreign UID'):wrapped.active_gate(runtime)
    rows=[json.loads(x) for x in wrapped.trace_path.read_text().splitlines()]
    assert len(calls)==2 and all(not row['accepted'] for row in rows)


def test_cross_attempt_same_pid_reuse(tmp_path,monkeypatch):
    stable=snapshot((30,)); stable['identities']['30']['starttime_ticks']=999
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[snapshot(),snapshot((30,)),stable,stable])
    with pytest.raises(RuntimeError,match='Cross-attempt'):wrapped.active_gate(runtime)
    assert len(calls)==2


def test_child_hashing_does_not_consume_cuda_startup_window(tmp_path,monkeypatch):
    clock=[0.]
    monkeypatch.setattr(guard.time,'monotonic',lambda:clock[0])
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[snapshot(),snapshot(),snapshot(),snapshot((30,)),snapshot((30,)),snapshot((30,))])
    clock[0]=123.  # Native child asset rehash may take >122s before any GPU PID.
    assert wrapped.active_gate(runtime)['full_gate_fixture']
    assert wrapped.deadline is None
    clock[0]=180.
    assert wrapped.active_gate(runtime)['full_gate_fixture']
    assert wrapped.deadline==200. and len(calls)==3


def test_unsafe_event_never_starts_clock(tmp_path,monkeypatch):
    final=snapshot((30,));final['identities']['30']['cgroup']=['0::/foreign-new']
    wrapped,runtime,_=setup(tmp_path,monkeypatch,[snapshot(),final])
    with pytest.raises(RuntimeError):wrapped.active_gate(runtime)
    assert wrapped.deadline is None


def test_ready_appears_inside_failed_window(tmp_path,monkeypatch):
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[snapshot(),snapshot((30,))])
    wrapped.ready_check=lambda:bool(calls)
    with pytest.raises(RuntimeError):wrapped.active_gate(runtime)
    assert wrapped.closed and len(calls)==1


def test_success_after_active_deadline_rejected(tmp_path,monkeypatch):
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[snapshot(),snapshot((30,)),snapshot((30,)),snapshot((30,))])
    clock=[0.];monkeypatch.setattr(guard.time,'monotonic',lambda:clock[0])
    original=wrapped.original_gate
    def slow(*args,**kwargs):
        result=original(*args,**kwargs)
        if len(calls)==2:clock[0]=31.
        return result
    wrapped.original_gate=slow
    with pytest.raises(RuntimeError,match='after bounded deadline'):wrapped.active_gate(runtime)
    assert all(not json.loads(line)['accepted'] for line in wrapped.trace_path.read_text().splitlines())


@pytest.mark.parametrize('fail',['none','supervisor_error','source_drift'])
def test_actual_launch_parent_binding_popen_and_restoration(tmp_path,monkeypatch,fail):
    import hashlib
    import launch
    root=tmp_path/'root'; tools=root/'tools/qwen25-32b-terminal-fix-refreeze'
    own=root/'tools/qwen25-32b-startup-guard-candidate-v5'
    tools.mkdir(parents=True);own.mkdir();(root/'logs').mkdir()
    for name in ('supervisor.py','eval64.py','eval64.sbatch'):(tools/name).write_text(name)
    (tools/'runtime_safety.py').write_bytes(SEALED.read_bytes())
    (own/'launch.py').write_text('fixture explicit entry path')
    (own/'files.json').write_text(json.dumps({'launch.py':hashlib.sha256((own/'launch.py').read_bytes()).hexdigest()}))
    monkeypatch.setattr(launch,'ROOT',root);monkeypatch.setattr(launch,'TOOLS',tools)
    monkeypatch.setattr(launch,'__file__',str(own/'launch.py'))
    monkeypatch.setattr(launch,'PINS',{name:hashlib.sha256((tools/name).read_bytes()).hexdigest() for name in launch.PINS})
    monkeypatch.setenv('AGENT_ROOT',str(root));monkeypatch.setenv('SLURM_JOB_ID','777')
    monkeypatch.setenv('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    monkeypatch.setattr(sys,'argv',['launch.py','--binding',str(tmp_path/'binding.json')])
    wrapped,runtime,_=setup(tmp_path,monkeypatch,[snapshot(),snapshot()])
    sealed=wrapped.sealed
    # A fresh sealed module identity, not a previously installed spy.
    del sealed._startup_guard_spy
    sealed.nvml_snapshot=wrapped.original_snapshot
    baseline_gate=sealed.active_gate;baseline_snapshot=sealed.nvml_snapshot
    sealed.__file__=str(tools/'runtime_safety.py')
    monkeypatch.setattr(guard,'held_leases',lambda:set(str(i) for i in range(8)))
    monkeypatch.setattr(guard,'process_uid',lambda pid:os_uid)
    os_uid=wrapped.owner_uid
    calls=[];sentinel=object(); fds=(7,8)
    def original_popen(command,*args,**kwargs):calls.append((command,args,kwargs));return sentinel
    monkeypatch.setattr(launch.subprocess,'Popen',original_popen)
    seen=[]
    def supervisor_main():
        assert sealed.active_gate is not baseline_gate
        command=['harmless-control','--x'];options={'pass_fds':fds,'env':{'only':'test'}}
        assert launch.subprocess.Popen(command,**options) is sentinel
        assert calls[-1][0] is command and calls[-1][2]['pass_fds'] is fds
        output=root/'logs/evaluation';output.mkdir()
        assert launch.subprocess.Popen(['python',str(tools/'eval64.py'),'--output',str(output)],pass_fds=fds) is sentinel
        assert sealed.active_gate(runtime)['full_gate_fixture']
        (output/'readiness.json').write_text('{}')
        # A second evaluator cannot reopen/renew the startup window.
        with pytest.raises(RuntimeError,match='spawn repeated'):
            launch.subprocess.Popen(['python',str(tools/'eval64.py'),'--output',str(output)])
        seen.append(True)
        if fail=='supervisor_error':raise RuntimeError('original supervisor failed')
    supervisor=types.SimpleNamespace(main=supervisor_main)
    def loader(name,path):
        assert Path(path)==tools/({'eval64':'eval64.py','runtime_safety':'runtime_safety.py','explicit_sealed_supervisor':'supervisor.py'}[name])
        return sealed if name=='runtime_safety' else supervisor if name=='explicit_sealed_supervisor' else types.SimpleNamespace()
    monkeypatch.setattr(launch,'load_file',loader)
    if fail=='source_drift':(tools/'supervisor.py').write_text('changed after freeze')
    if fail=='none':launch.main()
    else:
        with pytest.raises(RuntimeError):launch.main()
    assert launch.subprocess.Popen is original_popen
    assert sealed.active_gate is baseline_gate and sealed.nvml_snapshot is baseline_snapshot
    if fail=='source_drift':assert not calls and not seen
    else:
        assert seen and len(calls)==2
        records=[json.loads(x) for x in (root/'logs/qwen25-32b-startup-guard-777.jsonl').read_text().splitlines()]
        assert records[1]['event']=='ACTUAL_EVALUATOR_SPAWN_ELIGIBILITY_ARM'
        assert records[1]['deadline_monotonic'] is None
        assert records[-1]['event']=='WRAPPER_FINISHED'


def test_stable_monitor_after_window_and_expired_churn(tmp_path,monkeypatch):
    clock=[0.];monkeypatch.setattr(guard.time,'monotonic',lambda:clock[0])
    a=snapshot((30,));b=snapshot((30,31))
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[snapshot(),a,a,a,a,a,a,b])
    assert wrapped.active_gate(runtime)['full_gate_fixture']
    assert wrapped.deadline==20.
    clock[0]=40.
    assert wrapped.active_gate(runtime)['full_gate_fixture']  # Still loading, unchanged validator stable.
    with pytest.raises(RuntimeError,match=guard.IDENTITY_CHANGED):wrapped.active_gate(runtime)
    assert wrapped.deadline==20. and len(calls)==4


def test_actual_multiple_safe_cuda_startup_phases_bounded120(tmp_path,monkeypatch):
    clock=[0.];monkeypatch.setattr(guard.time,'monotonic',lambda:clock[0])
    a=snapshot((30,));b=snapshot((30,31));c=snapshot((30,31,32))
    wrapped,runtime,calls=setup(tmp_path,monkeypatch,[snapshot(),a,a,a,a,b,b,b,b,c,c,c,c,c,c,snapshot((30,31,32,33))],startup_seconds=120)
    assert wrapped.active_gate(runtime)['full_gate_fixture'] and wrapped.deadline==120.
    clock[0]=7.;assert wrapped.active_gate(runtime)['full_gate_fixture']
    clock[0]=18.;assert wrapped.active_gate(runtime)['full_gate_fixture']
    clock[0]=23.;assert wrapped.active_gate(runtime)['full_gate_fixture']  # Beyond old20sec, original full check stable.
    clock[0]=121.
    with pytest.raises(RuntimeError):wrapped.active_gate(runtime)
    assert wrapped.deadline==120. and len(calls)==8


def test_explicit_cache_environment_only_ownroot(tmp_path,monkeypatch):
    import launch
    for key in ('VLLM_CACHE_ROOT','XDG_CACHE_HOME','HF_HOME','TORCH_HOME','TRITON_CACHE_DIR'):
        monkeypatch.setenv(key,'/foreign/cache')
    values=launch.configure_own_cache(tmp_path,123)
    assert len(values)==5 and all(Path(v).is_dir() and tmp_path in Path(v).parents for v in values.values())
    assert all(launch.os.environ[k]==v for k,v in values.items())


def test_cache_symlink_rejected_before_external_write(tmp_path):
    import launch
    root=tmp_path/'own';root.mkdir();outside=tmp_path/'foreign';outside.mkdir()
    (root/'cache').symlink_to(outside,target_is_directory=True)
    with pytest.raises(RuntimeError,match='symlink'):launch.configure_own_cache(root,123)
    assert not list(outside.iterdir())
