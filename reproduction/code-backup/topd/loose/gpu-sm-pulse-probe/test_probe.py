"""CPU adversarial identity contracts; no CUDA hardware claim."""
import importlib.util
import os
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['AGENT_ROOT']) if os.environ.get('AGENT_ROOT') else HERE.parent
sys.path.insert(0,str(ROOT/'repo' if os.environ.get('AGENT_ROOT') else ROOT/'topd'))
sys.path.insert(0,str(ROOT/'tools/qwen25-32b-eval64' if os.environ.get('AGENT_ROOT') else ROOT/'qwen25-32b-eval64'))
spec=importlib.util.spec_from_file_location('pulse_probe',HERE/'probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def pair():
    base=dict(gpus=[dict(uuid='u',used_mib=1000,total_mib=80000,sm=0)],applications=[dict(uuid='u',pid=1,used_mib=1000)],identities={'1':dict(pid=1,starttime_ticks=1,cgroup=['0::/foreign'])})
    current=dict(base,applications=base['applications']+[dict(uuid='u',pid=2,used_mib=10)],identities=dict(base['identities'],**{'2':dict(pid=2,starttime_ticks=2,cgroup=['0::/own'])}))
    return base,current

def test_baseline_and_single_uuid_own_are_classified():
    b,c=pair();assert p.validate_snapshot(c,b,{('u',2)},{'u'},'/own')=={('u',1)}

@pytest.mark.parametrize('fault',['reused','disappeared','outsider','wrongscope','memory','nonfinite'])
def test_real_identity_or_measurement_violations_reject(fault):
    b,c=pair()
    if fault=='reused':c['identities']['1']=dict(c['identities']['1'],starttime_ticks=3)
    if fault=='disappeared':c['applications']=c['applications'][1:]
    if fault=='outsider':c['applications'][-1]['pid']=3;c['identities']['3']=c['identities']['2']
    if fault=='wrongscope':c['identities']['2']['cgroup']=['0::/another']
    if fault=='memory':c['applications'][0]['used_mib']=20000
    if fault=='nonfinite':c['gpus'][0]['sm']=float('nan')
    with pytest.raises(RuntimeError):p.validate_snapshot(c,b,{('u',2)},{'u'},'/own')

class Child:
    def poll(self):return None

def test_busy_receipts_do_not_skip_rolling_warmup(monkeypatch):
    base,one=pair();uids={'u'};own={('u',2)}
    clock=[0.]
    monkeypatch.setattr(p.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(p.time,'sleep',lambda delay:clock.__setitem__(0,clock[0]+delay))
    queue=iter([dict(one,gpus=[dict(one['gpus'][0],sm=0)]),dict(one,gpus=[dict(one['gpus'][0],sm=14)]),dict(one,gpus=[dict(one['gpus'][0],sm=90)])])
    monkeypatch.setattr(p.c,'nvml_snapshot',lambda:next(queue))
    attempts=[];checks=[]
    snapshot,foreign=p.wait_active_snapshot(base,own,uids,'/own',[Child()],attempts,lambda:checks.append(True),warmup_seconds=1)
    assert [r['snapshot']['gpus'][0]['sm'] for r in attempts]==[0,14,90]
    assert len(checks)==3 and foreign=={('u',1)}
    assert snapshot['gpus'][0]['sm']==90

def test_never_high_timeout_retains_every_actual_attempt(monkeypatch):
    base,one=pair();clock=[0.]
    monkeypatch.setattr(p.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(p.time,'sleep',lambda delay:clock.__setitem__(0,clock[0]+delay))
    monkeypatch.setattr(p.c,'nvml_snapshot',lambda:one)
    attempts=[]
    with pytest.raises(RuntimeError,match='never reached'):
        p.wait_active_snapshot(base,{('u',2)},{'u'},'/own',[Child()],attempts,lambda:None,timeout_seconds=1)
    assert len(attempts)==2 and all(r['snapshot']['gpus'][0]['sm']==0 for r in attempts)

def test_warmup_does_not_retry_unknown_owner(monkeypatch):
    base,one=pair();one['applications'][-1]['pid']=3;one['identities']['3']=one['identities']['2']
    monkeypatch.setattr(p.c,'nvml_snapshot',lambda:one)
    attempts=[]
    with pytest.raises(RuntimeError,match='Unknown GPU owner'):
        p.wait_active_snapshot(base,{('u',2)},{'u'},'/own',[Child()],attempts,lambda:None)
    assert len(attempts)==1
