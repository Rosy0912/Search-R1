"""CPU gather/ACK barrier contracts; fake scorer is not native GPU evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import threading
import time
import pytest
import torch
from safetensors.torch import save
from topd32b_shared_pipeline import backend as b


class FakeScorer:
    def __init__(self,fail=None):self.calls=0;self.fail=fail
    def score_round(self,tensors,workers):
        self.calls+=1
        if self.fail=='score':raise RuntimeError('CPU fixture row error')
        output=tensors['responses'].float()*tensors['attention']
        frozen=dict(parameter_sha256='p',buffer_sha256='b',parameter_count=3,parameter_bytes=12,
                    parameter_devices=['cpu'],buffer_devices=['cpu'],training_modules=0,requires_grad_parameters=0,grad_parameters=0)
        record=dict(status='RELEASED',all_threads_joined=True,input_sha256=b.digest(save(tensors)),
                    initial_cpu=frozen,released_cpu=frozen,output_sha256=b.digest(save(dict(teacher_log_probs=output,row_indices=tensors['row_indices']))),allocators_released=[dict(device=i,allocated_bytes=0,reserved_bytes=int(self.fail=='release' and i==7)) for i in range(8)])
        return output,record


@pytest.fixture
def setup(tmp_path):
    tmp_path.chmod(0o700)
    owner=dict(pid=123,starttime=1,uid=0)
    m=dict({k:k+'-fixture' for k in b.IDENTITY},protocol=b.PROTOCOL,kind='CPU_FIXTURE_NO_ALGORITHM',
           gpu_owner_process_count=1,rank_endpoint_count=8,owner=owner,driver=owner,parent_cgroup='/slurm/fixture',
           leased_uuids=['GPU-'+str(i) for i in range(8)],rows_per_rank=2,workers=8,temperature=1.,chunk=16,
           outer_updates=2,timeout_seconds=.5,capability='a'*64,pad_token_id=0,
           endpoints=[dict(rank=i,pid=123,starttime=1,socket=str(tmp_path/('r'+str(i)+'.sock'))) for i in range(8)])
    scorer=FakeScorer();c=b.Coordinator(scorer,m,tmp_path)
    return c,scorer,m,tmp_path


def request(c,rank,nonce='b'*64):
    start=rank*2
    tensors=dict(prompts=torch.ones((2,3),dtype=torch.long),responses=torch.arange(start,start+2)[:,None].expand(2,3).contiguous(),
                 attention=torch.tensor([[1,0,1]]*2),row_indices=torch.arange(start,start+2))
    body=save(tensors)
    meta=dict(b.identity(c.m),protocol=b.PROTOCOL,command='SCORE_RANK',capability=c.m['capability'],
              outer_step=c.step,nonce=nonce,rank=rank,input_sha256=b.digest(body))
    return meta,body


def phase(c,omit=None,mutate=None):
    outputs=[];errors=[]
    def worker(rank):
        try:
            meta,body=request(c,rank)
            if mutate:meta,body=mutate(rank,meta,body)
            result,payload=c.submit(rank,meta,body,dict(cpu_fixture_peer=True))
            outputs.append((rank,result,payload))
            ack=dict(meta,command='ACK_RANK',input_sha256=result['receipt']['sha256'])
            c.acknowledge(rank,ack,b'')
        except Exception as exc:errors.append(str(exc))
    threads=[threading.Thread(target=worker,args=(i,)) for i in range(8) if i!=omit]
    for t in threads:t.start()
    for t in threads:t.join(3)
    assert not any(t.is_alive() for t in threads)
    return outputs,errors


def test_eight_ranks_stage_once_globalrelease_then_ack_actor_barrier(setup):
    c,scorer,m,d=setup
    out,err=phase(c)
    assert not err and len(out)==8 and scorer.calls==1
    assert c.state=='ACKED' and c.acks==set(range(8))
    assert (d/'global-release-step1.json').exists() and (d/'global-ack-step1.json').exists()
    c.advance();assert c.step==2 and c.state=='READY'
    # New nonce mandatory across phases, no request replay.
    old=request(c,0)[0]
    with pytest.raises(RuntimeError):c.submit(0,old,request(c,0)[1],{})


@pytest.mark.parametrize('failure',['score','release'])
def test_any_score_or_allocator_failure_publishes_no_rank_result(setup,failure):
    c,s,_,d=setup;s.fail=failure
    out,errors=phase(c)
    assert errors and not out and c.state=='FAILED' and not (d/'global-release-step1.json').exists()


def test_missing_rank_times_out_without_staging_or_actor_ack(setup):
    c,s,_,d=setup
    out,errors=phase(c,omit=7)
    assert errors and not out and s.calls==0 and c.state=='FAILED'
    assert not (d/'global-ack-step1.json').exists()


@pytest.mark.parametrize('fault',['gold','reorder','sha','config'])
def test_one_bad_rank_poison_entire_global_phase(setup,fault):
    c,s,_,_=setup
    def mutate(rank,meta,body):
        if rank!=4:return meta,body
        if fault=='config':meta['config_sha256']='wrong'
        elif fault=='sha':meta['input_sha256']='wrong'
        else:
            tensors=b.load(body)
            if fault=='gold':tensors['gold']=torch.tensor([1])
            else:tensors['row_indices']=tensors['row_indices'].flip(0)
            body=save(tensors);meta['input_sha256']=b.digest(body)
        return meta,body
    out,errors=phase(c,mutate=mutate)
    assert errors and not out and s.calls==0 and c.state=='FAILED'


def test_no_actor_result_until_allrank_ack(setup):
    c,s,_,d=setup;results=[]
    def submit(rank):results.append(c.submit(rank,*request(c,rank),{}))
    ts=[threading.Thread(target=submit,args=(i,)) for i in range(8)]
    for t in ts:t.start()
    for t in ts:t.join()
    assert c.state=='RESULT_READY' and len(results)==8
    meta,_=request(c,0);meta.update(command='ACK_RANK',input_sha256=c.receipt['sha256'])
    with pytest.raises(RuntimeError,match='timeout'):c.acknowledge(0,meta,b'')
    assert c.state=='FAILED' and not (d/'global-ack-step1.json').exists()


def test_manifest_cannot_masquerade_as_eight_model_processes_or_training_ready(setup):
    _,_,m,_=setup
    for bad in [dict(m,gpu_owner_process_count=8),dict(m,kind='TRAINED_TEACHER'),dict(m,workers=1)]:
        with pytest.raises(RuntimeError):b.validate_manifest(bad)


def test_hardware_kind_rejects_cpu_stub_even_if_manifest_claimed_native(setup):
    _,scorer,m,d=setup
    with pytest.raises(RuntimeError,match='Actual native'):
        b.Coordinator(scorer,dict(m,kind='NATIVE_HARDWARE_PROBE_NO_TRAINING'),d)


def test_rank_failure_during_score_discards_even_released_results(setup):
    c,scorer,_,d=setup
    real=scorer.score_round
    def fail_midway(*args,**kwargs):
        c.fail('peer disconnected while GPU scoring')
        return real(*args,**kwargs)
    scorer.score_round=fail_midway
    out,errors=phase(c)
    assert errors and not out and c.state=='FAILED'
    assert not (d/'global-release-step1.json').exists()


def test_before_actor_requires_fresh_global_ack_and_refuses_tamper(setup):
    from types import SimpleNamespace
    import json
    c,_,m,d=setup;out,err=phase(c)
    assert not err
    receipt=json.loads((d/'global-release-step1.json').read_text())
    client=SimpleNamespace(failed=False,step=2,last_receipts=[receipt],m=m,directory=d)
    assert b.require_global_ack_before_actor(client,1)['ranks']==list(range(8))
    ack=json.loads((d/'global-ack-step1.json').read_text());ack['ranks']=[0]
    (d/'global-ack-step1.json').write_text(json.dumps(ack))
    with pytest.raises(RuntimeError):b.require_global_ack_before_actor(client,1)
    client.failed=True
    with pytest.raises(RuntimeError):b.require_global_ack_before_actor(client,1)


def test_real_linux_eight_sockets_two_rounds_actual_peer_global_release(tmp_path):
    import os,socket
    if not hasattr(socket,'SO_PEERCRED') or not Path('/proc/self/stat').exists():
        pytest.skip('Actual Linux SO_PEERCRED requires native CPU gate')
    actual=b.process_identity(os.getpid())
    scope=actual['cgroups'][0]
    if scope in ('/','/system.slice'):
        pytest.skip('Requires real job-specific own cgroup for authenticated Unix test')
    tmp_path.chmod(0o700)
    owner=dict(actual,uid=os.getuid())
    m=dict({k:k+'-fixture' for k in b.IDENTITY},protocol=b.PROTOCOL,kind='CPU_FIXTURE_NO_ALGORITHM',
           gpu_owner_process_count=1,rank_endpoint_count=8,owner=owner,driver=owner,parent_cgroup=scope,
           leased_uuids=['GPU-'+str(i) for i in range(8)],rows_per_rank=2,workers=8,temperature=1.,chunk=16,
           outer_updates=2,timeout_seconds=3,capability='a'*64,pad_token_id=0,
           endpoints=[dict(rank=i,pid=os.getpid(),starttime=owner['starttime'],socket=str(tmp_path/('r'+str(i)))) for i in range(8)])
    scorer=FakeScorer();c=b.Coordinator(scorer,m,tmp_path);server=b.RankUnixServer(c);server.start()
    errors=[]
    def serve():
        try:server.serve()
        except BaseException as exc:errors.append(str(exc))
    t=threading.Thread(target=serve);t.start()
    client=b.RankTeacherClient(m,tmp_path)
    try:
        p=torch.ones((16,3),dtype=torch.long);r=torch.arange(16)[:,None].expand(16,3).contiguous();a=torch.tensor([[1,0,1]]*16)
        for step in (1,2):
            out=client.score(p,r,a,temperature=1.,chunk=16,step=step)
            assert torch.equal(out,r.float()*a)
            b.require_global_ack_before_actor(client,step)
        t.join(5)
        assert not t.is_alive() and not errors and scorer.calls==2
        assert not any(Path(e['socket']).exists() for e in m['endpoints'])
    finally:
        if t.is_alive():c.fail('test termination');t.join(5)


def test_training_kind_missing_new_readiness_never_constructs(setup):
    _,scorer,m,d=setup
    with pytest.raises((KeyError,RuntimeError)):
        b.Coordinator(scorer,dict(m,kind='NATIVE_32B_TOPD_SHARED_PIPELINE'),d)


@pytest.mark.parametrize('scope',['CPU_FIXTURE_ONLY','OLD_3B_HARDWARE_PROOF','TOPD_SHARED_FIXTURE'])
def test_legacy_or_cpu_review_cannot_unlock_new_training_kind(setup,monkeypatch,scope):
    from topd32b_shared_pipeline import readiness
    import json
    _,_,m,d=setup
    monkeypatch.setattr(readiness,'ROOT',d)
    (d/'logs').mkdir()
    f=d/'logs/old.json';f.write_text(json.dumps(dict(status='REVIEWED',scope=scope)))
    m=dict(m,kind='NATIVE_32B_TOPD_SHARED_PIPELINE',production_readiness=dict(path=str(f),sha256=b.digest(f.read_bytes())))
    with pytest.raises(RuntimeError,match='New shared32B'):b.validate_manifest(m)
