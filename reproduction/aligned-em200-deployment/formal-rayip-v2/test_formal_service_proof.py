"""Pure ownership/address fixtures; never start Ray, inspect proc or call target tools."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def proof(monkeypatch):
    def require(value, message):
        if not value:
            raise ValueError(message)
    admission=SimpleNamespace(ROOT=Path('/fixed/root'),SOURCE='source080',require=require,raw_file=lambda p:b'{"path":"/owned/runtime"}')
    monkeypatch.setitem(sys.modules,'formal_admission',admission)
    spec=importlib.util.spec_from_file_location('rayproof_fixture',Path(__file__).with_name('formal_service_proof.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def interface(ip='10.33.0.5',flags=None,state='UP'):
    return dict(ifname='fixture',flags=flags if flags is not None else ['UP'],operstate=state,addr_info=[dict(family='inet',local=ip)])


def test_real_interface_query_keeps_only_active_numeric_addresses(proof,monkeypatch):
    rows=[interface(),interface('10.33.0.99',state='DOWN'),interface('127.0.0.1',['UP','LOOPBACK'],'UNKNOWN')]
    def query(command,**kwargs):
        assert command==['ip','-j','-4','address','show','up']
        assert kwargs['check'] and kwargs['timeout']==10
        return SimpleNamespace(stdout=json.dumps(rows))
    monkeypatch.setattr(proof.subprocess,'run',query)
    assert proof.active_local_ipv4()['addresses']==['10.33.0.5','127.0.0.1']


@pytest.mark.parametrize('rows',[[],[interface(state='DOWN')],[interface('0.0.0.0')],[interface('224.0.0.1')],[interface('host.example')]])
def test_bad_or_inactive_interface_data_refused(proof,monkeypatch,rows):
    monkeypatch.setattr(proof.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=json.dumps(rows)))
    with pytest.raises(ValueError):proof.active_local_ipv4()


@pytest.fixture
def connected(proof,monkeypatch):
    nodes=[dict(Alive=True,NodeManagerAddress='10.33.0.5',NodeManagerPort=12345,Resources={'GPU':8})]
    events=[];owner=dict(pid=30,ppid=20,start_ticks=100,uid=5,executable={'path':'/fixture/raylet'})
    fake_ray=SimpleNamespace(is_initialized=lambda:False,init=lambda **k:events.append(('init',k)),nodes=lambda:nodes,shutdown=lambda:events.append(('shutdown',)),_private=SimpleNamespace(worker=SimpleNamespace(global_worker=SimpleNamespace(node=SimpleNamespace(address_info={'session_dir':'/owned/runtime/ray/session_1'})))))
    monkeypatch.setitem(sys.modules,'ray',fake_ray)
    monkeypatch.setenv('BAPO_RAY_CLI_PID','10')
    monkeypatch.setattr(proof,'validate_services',lambda:events.append(('births_checked',)))
    monkeypatch.setattr(proof,'active_local_ipv4',lambda:dict(addresses=['10.33.0.5','127.0.0.1'],interfaces=['CPU_FIXTURE']))
    monkeypatch.setattr(proof,'port_owner',lambda p:dict(owner))
    def own(pid):
        if pid==30:return dict(owner)
        if pid==20:return dict(pid=20,ppid=10,start_ticks=200)
        if pid==10:return dict(pid=10,ppid=2,start_ticks=300)
        raise ValueError('foreign ancestry')
    monkeypatch.setattr(proof,'own_identity',own)
    monkeypatch.setattr(proof,'path',lambda p:Path('/fixture')/p)
    monkeypatch.setattr(proof,'freeze',lambda n,v:events.append(('freeze',n,v)))
    return proof,nodes,owner,events,fake_ray


def test_normalized_local_ip_retains_ownership_session_and_freezes_evidence(connected):
    proof,nodes,owner,events,ray=connected;proof.verify_ray_connection()
    frozen=next(e[2]for e in events if e[0]=='freeze')
    assert frozen['advertised_ip']=='10.33.0.5' and frozen['raylet']==owner
    assert frozen['session_dir']=='/owned/runtime/ray/session_1'
    assert frozen['local_interface_evidence']['interfaces']==['CPU_FIXTURE']
    assert sum(e[0]=='births_checked'for e in events)==2
    assert events[-1]==('shutdown',)


@pytest.mark.parametrize('change',[{'NodeManagerAddress':'10.33.0.99'},{'Resources':{'GPU':7}},{'NodeManagerPort':True},{'NodeManagerPort':0},{'NodeManagerPort':65536}])
def test_foreign_ip_gpu_count_and_invalid_port_never_freeze(connected,change):
    proof,nodes,owner,events,ray=connected;nodes[0].update(change)
    with pytest.raises(ValueError):proof.verify_ray_connection()
    assert not any(e[0]=='freeze'for e in events) and events[-1]==('shutdown',)


def test_two_alive_nodes_refused(connected):
    proof,nodes,owner,events,ray=connected;nodes.append(dict(nodes[0]))
    with pytest.raises(ValueError):proof.verify_ray_connection()
    assert not any(e[0]=='freeze'for e in events)


def test_foreign_raylet_executable_and_ancestry_refused(connected):
    proof,nodes,owner,events,ray=connected;owner['executable']['path']='/fixture/foreign'
    with pytest.raises(ValueError,match='not own raylet'):proof.verify_ray_connection()
    owner['executable']['path']='/fixture/raylet';owner['ppid']=1
    with pytest.raises(ValueError,match='ancestry'):proof.verify_ray_connection()
    assert not any(e[0]=='freeze'for e in events)


def test_foreign_job_identity_failure_propagates(connected,monkeypatch):
    proof,nodes,owner,events,ray=connected
    def reject(pid):raise ValueError('foreign service run/job')
    monkeypatch.setattr(proof,'own_identity',reject)
    with pytest.raises(ValueError,match='run/job'):proof.verify_ray_connection()
    assert not any(e[0]=='freeze'for e in events)


def test_changed_raylet_epoch_refused(connected,monkeypatch):
    proof,nodes,owner,events,ray=connected;calls=[]
    def port(p):
        calls.append(p);return dict(owner,start_ticks=100 if len(calls)==1 else 101)
    monkeypatch.setattr(proof,'port_owner',port)
    with pytest.raises(ValueError,match='raylet changed'):proof.verify_ray_connection()
    assert not any(e[0]=='freeze'for e in events)


def test_local_interface_disappears_refused(connected,monkeypatch):
    proof,nodes,owner,events,ray=connected;calls=[]
    def local():
        calls.append(1);return dict(addresses=['10.33.0.5'] if len(calls)==1 else ['127.0.0.1'])
    monkeypatch.setattr(proof,'active_local_ipv4',local)
    with pytest.raises(ValueError,match='disappeared'):proof.verify_ray_connection()
    assert not any(e[0]=='freeze'for e in events)


def test_foreign_session_refused(connected):
    proof,nodes,owner,events,ray=connected
    ray._private.worker.global_worker.node.address_info['session_dir']='/other/runtime/ray/session'
    with pytest.raises(ValueError,match='foreign Ray session'):proof.verify_ray_connection()
    assert not any(e[0]=='freeze'for e in events)
