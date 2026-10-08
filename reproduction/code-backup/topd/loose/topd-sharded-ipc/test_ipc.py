"""External shared IPC CPU fixtures; mocks explicitly not GPU evidence."""
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save,load
from transformers import Qwen2Config,Qwen2ForCausalLM

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import protocol,session,client,server
ROOT=HERE.parent.parent
source=ROOT/'work/topd'
if not source.is_dir():source=Path(os.environ['AGENT_ROOT'])/'repo'
sys.path.insert(0,str(source))
from search_r1.baselines.distillation import FrozenTeacher
scorer_path=HERE.parent/'topd-sharded-scorer/scorer.py'
assert hashlib.sha256(scorer_path.read_bytes()).hexdigest()=='2d5da3fdd94b637c1625677b302df4df21395ef138f9de2ce484035eab7298ea'
spec=importlib.util.spec_from_file_location('frozen_external_scorer',scorer_path);scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)


class CpuRuntime:
    kind='CPU_ONLY_FIXTURE_NOT_HARDWARE_EVIDENCE';device='cpu'
    def __init__(self):self.stages=0;self.releases=0
    def verify(self):pass
    def flags(self):return {'scope':'CPU_FIXTURE_NO_CUDA_FLAGS'}
    def reset_peaks(self):pass
    def dispatch(self,model,mapping):self.stages+=1;return model
    def synchronize(self):pass
    def peaks(self):return [{'scope':'CPU_FIXTURE_NOT_NVML'} for _ in range(8)]
    def verify_partition(self,model,inventory):pass
    def unhook(self,model):pass
    def gpu_identity(self):return [{'scope':'CPU_FIXTURE_NOT_NVML'}]
    def release_allocators(self):self.releases+=1;return [dict(device=i,allocated_bytes=0,reserved_bytes=0) for i in range(8)]


def build_fixture(directory):
    directory=directory.resolve()
    torch.manual_seed(17);torch.set_num_threads(1)
    config=Qwen2Config(vocab_size=128,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=8192,pad_token_id=0);config._attn_implementation='eager'
    teacher=FrozenTeacher.__new__(FrozenTeacher);teacher.tokenizer=SimpleNamespace(pad_token_id=0);teacher.device=torch.device('cpu');teacher.scored_tokens=0
    teacher.model=Qwen2ForCausalLM(config).float().eval().requires_grad_(False);runtime=CpuRuntime()
    mapping={name:'cuda:0' for name in ['model.embed_tokens','model.rotary_emb','model.norm','lm_head','model.layers.0','model.layers.1']}
    shared=scorer.SharedScorer(teacher,runtime,mapping,range(128),expected_parameters=sum(p.numel() for p in teacher.model.parameters()))
    manifest={key:str(index)*64 for index,key in enumerate(protocol.IDENTITY)}
    identity=dict(pid=os.getpid(),starttime=123,uid=os.getuid())
    manifest.update(original_wrapper_file_sha256=protocol.file_sha(inspect.getfile(shared.original)),frozen_source_binding=session.BINDING,protocol=protocol.PROTOCOL,backend='local_gpu_fp32_sharded_shared',kind='CPU_FIXTURE_NO_ALGORITHM',rows_per_rollout=5,outer_updates=2,temperature=1.,chunk=16,scoring_batch_rows=4,max_prompt=2048,max_response=4096,leased_uuids=[f'GPU-FIXTURE-{i}' for i in range(8)],capability=secrets.token_hex(32),service=dict(identity,socket=str(directory/'s.sock')),driver=identity,parent_cgroup='/fixture/job_1',pad_token_id=0,timeout_seconds=5)
    sess=session.SharedSession(shared,manifest,directory)
    sess.peer_identity=dict(identity,cgroups=['/fixture/job_1/driver']);sess.actual_service_identity=dict(identity,cgroups=['/fixture/job_1/scorer'])
    tensors=dict(prompts=torch.tensor([[1,2,3]]*5),responses=torch.tensor([[4,5,0]]*5),attention=torch.tensor([[1,1,0]]*5),row_indices=torch.arange(5))
    return sess,tensors,manifest,directory,runtime


@pytest.fixture
def fixture():
    with tempfile.TemporaryDirectory(prefix='topd-ipc-',dir='/tmp') as tmp:
        directory=Path(tmp);directory.chmod(0o700)
        yield build_fixture(directory)


def requests(fixture,nonce='f'*64,transport_rows=2):
    sess,tensors,manifest,_,_=fixture;cli=client.SharedTeacherClient(manifest,sess.directory);step=sess.next_step
    begin=cli.base('BEGIN_SCORE',nonce,step);begin.update(rows=5,prompt_width=3,response_width=3,temperature=1.,chunk=16,chunk_count=3,scoring_batch_rows=4,input_sha256=session.tensor_sha(tensors),expected_scored_tokens=25)
    chunks=[];ledger=[]
    for index,start in enumerate(range(0,5,transport_rows)):
        stop=min(start+transport_rows,5);raw=save({key:value[start:stop].contiguous() for key,value in tensors.items()})
        meta=cli.base('INPUT_CHUNK',nonce,step);meta.update(chunk_index=index,row_start=start,row_stop=stop,input_sha256=protocol.digest(raw))
        chunks.append((meta,raw));ledger.append(dict(chunk_index=index,row_start=start,row_stop=stop,input_sha256=protocol.digest(raw)))
    commit=cli.base('COMMIT_SCORE',nonce,step);commit.update(input_sha256=session.tensor_sha(tensors),ledger_sha256=protocol.digest(protocol.canonical(ledger)))
    return cli,begin,chunks,commit


def assembled(fixture):
    sess,*_=fixture;cli,begin,chunks,commit=requests(fixture);sess.begin(begin,b'')
    for meta,raw in chunks:sess.accept_chunk(meta,raw)
    return cli,begin,commit


def test_two_rounds_one_stage_and_result_after_release_exact_source(fixture):
    sess,tensors,manifest,directory,runtime=fixture
    reference=FrozenTeacher.log_probs(sess.scorer.teacher,tensors['prompts'],tensors['responses'],tensors['attention'],temperature=1.,targets=None,chunk=16)
    for step in [1,2]:
        cli,begin,chunks,commit=requests(fixture,nonce=str(step)*64)
        sess.begin(begin,b'')
        for meta,raw in chunks:sess.accept_chunk(meta,raw)
        assert runtime.stages==step-1
        metadata,payload=sess.commit(commit,b'');values,record,tokens=cli.validate_result(metadata,payload,tensors,begin['nonce'],step)
        assert torch.allclose(reference,values,atol=rtol(),rtol=rtol()) and tokens==25
        assert runtime.stages==runtime.releases==step and sess.scorer.state=='RELEASED'
        assert record['publication_epoch']>=record['original_scorer_receipt']['released_epoch']
        assert manifest['capability'] not in ''.join(path.read_text() for path in directory.glob('*.json'))
        ack=cli.base('ACK_RESULT',begin['nonce'],step);ack.update(output_sha256=metadata['output_sha256'],receipt_sha256=metadata['receipt']['sha256'])
        assert sess.acknowledge(ack,b'')['status']=='ACKED'
    assert len(list(directory.glob('receipt-*.json')))==2


def rtol():return 1e-4


@pytest.mark.parametrize('change',['gold','identity','capability','nonce','rows','context','temperature','chunk','batch','step'])
def test_begin_drift_never_stages(fixture,change):
    sess,_,_,_,runtime=fixture;_,begin,_,_=requests(fixture)
    if change=='gold':begin['targets']='secret'
    elif change=='identity':begin['data_identity']='0'*64
    elif change=='capability':begin['capability']='0'*64
    elif change=='nonce':begin['nonce']='bad'
    elif change=='rows':begin['rows']=2561
    elif change=='context':begin['response_width']=4097
    elif change=='temperature':begin['temperature']=.7
    elif change=='chunk':begin['chunk']=32
    elif change=='batch':begin['scoring_batch_rows']=8
    else:begin['outer_step']=2
    with pytest.raises(RuntimeError):sess.begin(begin,b'')
    assert runtime.stages==0


@pytest.mark.parametrize('change',['duplicate','gap','rows','mask','dtype','gold','wire','token'])
def test_chunk_drift_never_stages(fixture,change):
    sess,_,_,_,runtime=fixture;_,begin,chunks,_=requests(fixture);sess.begin(begin,b'');meta,raw=copy.deepcopy(chunks[0])
    if change=='duplicate':sess.accept_chunk(meta,raw)
    elif change=='gap':meta['chunk_index']=1
    elif change in ('rows','mask','dtype','gold','token'):
        tensors=load(raw)
        if change=='rows':tensors['row_indices']=torch.tensor([1,0])
        elif change=='mask':tensors['attention'][0,0]=2
        elif change=='dtype':tensors['responses']=tensors['responses'].float()
        elif change=='gold':tensors['targets']=torch.tensor([99])
        else:tensors['responses'][0,0]=128
        raw=save(tensors);meta['input_sha256']=protocol.digest(raw)
    else:meta['input_sha256']='0'*64
    with pytest.raises(RuntimeError):sess.accept_chunk(meta,raw)
    assert runtime.stages==0


def test_incomplete_or_wrong_commit_rejects_before_one_stage(fixture):
    sess,_,_,_,runtime=fixture;_,begin,chunks,commit=requests(fixture);sess.begin(begin,b'');sess.accept_chunk(*chunks[0])
    with pytest.raises(RuntimeError,match='Incomplete'):sess.commit(commit,b'')
    for meta,raw in chunks[1:]:sess.accept_chunk(meta,raw)
    commit['ledger_sha256']='0'*64
    with pytest.raises(RuntimeError,match='ledger'):sess.commit(commit,b'')
    assert runtime.stages==0


def test_eighth_release_nonzero_cannot_publish(fixture):
    sess,_,_,directory,runtime=fixture;_,_,commit=assembled(fixture)
    runtime.release_allocators=lambda:[dict(device=i,allocated_bytes=0,reserved_bytes=int(i==7)*32) for i in range(8)]
    with pytest.raises(RuntimeError):sess.commit(commit,b'')
    assert not list(directory.glob('receipt*')) and runtime.stages==1


@pytest.mark.parametrize('change',['status','hash','row_order','mask','release','peer','sidecar'])
def test_resigned_result_cannot_bypass_actual_evidence(fixture,change):
    sess,tensors,_,directory,_=fixture;cli,begin,commit=assembled(fixture);metadata,payload=sess.commit(commit,b'')
    if change=='status':metadata['status']='STAGED'
    elif change=='hash':metadata['output_sha256']='0'*64
    elif change in ('row_order','mask'):
        output=load(payload)
        if change=='row_order':output['row_indices']=torch.arange(4,-1,-1)
        else:output['teacher_log_probs'][0,2]=1.
        payload=save(output);metadata['output_sha256']=protocol.digest(payload)
    else:
        path=directory/metadata['receipt']['name'];record=json.loads(path.read_text())
        if change=='release':record['original_scorer_receipt']['allocators_released'][7]['reserved_bytes']=32
        elif change=='peer':record['actual_service_identity']['starttime']+=1
        else:record['sidecar']['sha256']='0'*64
        path.write_bytes(protocol.canonical(record)+b'\n');metadata['receipt']['sha256']=protocol.file_sha(path)
    with pytest.raises(RuntimeError):cli.validate_result(metadata,payload,tensors,begin['nonce'],1)


def test_unknown_ack_no_hidden_second_stage(fixture):
    sess,_,_,_,runtime=fixture;cli,begin,commit=assembled(fixture);metadata,_=sess.commit(commit,b'')
    ack=cli.base('ACK_RESULT',begin['nonce'],1);ack.update(output_sha256='0'*64,receipt_sha256=metadata['receipt']['sha256'])
    with pytest.raises(RuntimeError,match='ACK'):sess.acknowledge(ack,b'')
    assert sess.next_step==1 and runtime.stages==runtime.releases==1


def test_max2560row_wire_split_arithmetic_not_algorithm_reduction():
    assert 2560*(2048+4096+4096)*8>protocol.MAX_FRAME
    assert 320*(2048+4096+4096)*8<protocol.MAX_FRAME
    assert 2560*4096*4<protocol.MAX_FRAME


@pytest.mark.parametrize('change',['meta','payload','nonfinite','duplicate','incomplete'])
def test_bounded_frames_reject_before_untrusted_allocation(change):
    left,right=socket.socketpair()
    try:
        if change=='meta':left.sendall(struct.pack('!QQ',protocol.MAX_META+1,0))
        elif change=='payload':left.sendall(struct.pack('!QQ',2,protocol.MAX_FRAME+1))
        elif change=='nonfinite':raw=b'{"x":NaN}';left.sendall(struct.pack('!QQ',len(raw),0)+raw)
        elif change=='duplicate':raw=b'{"x":1,"x":2}';left.sendall(struct.pack('!QQ',len(raw),0)+raw)
        else:left.sendall(struct.pack('!QQ',10,0)+b'{}');left.shutdown(socket.SHUT_WR)
        with pytest.raises(RuntimeError):protocol.receive(right)
    finally:left.close();right.close()


def test_private_artifact_symlink_and_foreign_mode_rejected(fixture):
    _,_,_,directory,_=fixture
    outside=directory/'outside';outside.write_text('{}');outside.chmod(0o600);(directory/'alias').symlink_to(outside)
    with pytest.raises(RuntimeError):protocol.read_private_json(directory,dict(name='alias',sha256=protocol.file_sha(outside)))
    item=protocol.write_private_json(directory,'private.json',dict(x=1));(directory/item['name']).chmod(0o644)
    with pytest.raises(RuntimeError):protocol.read_private_json(directory,item)


def test_nonfixture_manifest_never_treats_cpu_pass_as_native(fixture):
    sess,_,manifest,_,_=fixture;manifest=copy.deepcopy(manifest);manifest['kind']='QUALITY_ACCEPTED_FROZEN_32B_BASE'
    with pytest.raises(RuntimeError,match='CPU fixture only'):session.SharedSession(sess.scorer,manifest,sess.directory)


def fake_peer(connection,expected,parent):return dict(expected,cgroups=[parent+'/CPU_FIXTURE'])


def fake_self_identity(pid):return dict(pid=pid,starttime=123,cgroups=['/fixture/job_1/CPU_FIXTURE'])


def test_cpu_socket_two_rounds_stays_single_stage_per_round(fixture,monkeypatch):
    sess,tensors,manifest,directory,runtime=fixture
    monkeypatch.setattr(client,'peer',fake_peer);monkeypatch.setattr(server,'peer',fake_peer);monkeypatch.setattr(server,'process_identity',fake_self_identity)
    daemon=server.SharedUnixServer(sess);daemon.start();errors=[]
    def run():
        try:daemon.serve()
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=run);thread.start()
    try:
        cli=client.SharedTeacherClient(manifest,directory)
        for step in (1,2):values=cli.score(tensors['prompts'],tensors['responses'],tensors['attention'],1.,16,step,transport_rows=2)
        assert cli.scored_tokens==50 and len(cli.last_receipts)==1 and runtime.stages==runtime.releases==2
        assert bool((values[tensors['attention']==0]==0).all())
    finally:thread.join(timeout=6);daemon.close()
    assert not thread.is_alive() and not errors and not (directory/'s.sock').exists()


def test_unknown_ack_permanently_poisoned_client_no_training_retry(fixture,monkeypatch):
    sess,tensors,manifest,directory,runtime=fixture;manifest['outer_updates']=1
    sess.manifest_sha=protocol.digest(protocol.canonical(manifest))
    monkeypatch.setattr(client,'peer',fake_peer);monkeypatch.setattr(server,'peer',fake_peer);monkeypatch.setattr(server,'process_identity',fake_self_identity)
    actual_receive=client.receive
    def corrupt_ack(connection):
        metadata,payload=actual_receive(connection)
        if metadata.get('status')=='ACKED':metadata['outer_step']=999
        return metadata,payload
    monkeypatch.setattr(client,'receive',corrupt_ack)
    daemon=server.SharedUnixServer(sess);daemon.start();thread=threading.Thread(target=daemon.serve);thread.start();cli=client.SharedTeacherClient(manifest,directory)
    try:
        with pytest.raises(RuntimeError,match='Unknown result ACK'):cli.score(tensors['prompts'],tensors['responses'],tensors['attention'],1.,16,1,transport_rows=2)
        assert cli.scored_tokens==0 and cli.failed and runtime.stages==runtime.releases==1
        with pytest.raises(RuntimeError,match='permanently failed'):cli.score(tensors['prompts'],tensors['responses'],tensors['attention'],1.,16,1)
    finally:thread.join(timeout=6);daemon.close()
    assert not thread.is_alive() and runtime.stages==1


def test_concurrent_begin_rejected_and_disconnect_assembler_never_stages(fixture):
    sess,_,_,_,runtime=fixture;_,begin,_,_=requests(fixture);sess.begin(begin,b'')
    with pytest.raises(RuntimeError):sess.begin(begin,b'')
    sess.abort();assert sess.state=='FAILED' and runtime.stages==0 and not sess.chunks


@pytest.mark.parametrize('binding',['scorer_sha256','original_method_sha256','original_file_sha256'])
def test_frozen_source_binding_drift_rejected_before_stage(fixture,binding):
    sess,_,manifest,directory,runtime=fixture
    changed=copy.deepcopy(manifest);changed['frozen_source_binding'][binding]='0'*64
    with pytest.raises(RuntimeError):session.SharedSession(sess.scorer,changed,directory)
    assert runtime.stages==0


@pytest.mark.skipif(sys.platform!='linux' or not hasattr(socket,'SO_PEERCRED'),reason='Native Linux actual peer/cgroup only')
@pytest.mark.parametrize('drift',['none','starttime','parent_scope'])
def test_native_linux_actual_two_process_peer_identity(fixture,drift):
    import subprocess,time
    sess,tensors,_,directory,_=fixture
    reference=FrozenTeacher.log_probs(sess.scorer.teacher,tensors['prompts'],tensors['responses'],tensors['attention'],temperature=1.,targets=None,chunk=16)
    identity=protocol.process_identity(os.getpid());identity['uid']=os.getuid()
    paths=[p for p in identity['cgroups'] if p not in ('/','/system.slice')]
    assert paths,'Native gate requires actual specific Slurm job cgroup'
    parent=paths[0]
    # A Slurm driver and child inherit the precise current step/job scope.
    expected=copy.deepcopy(identity)
    if drift=='starttime':expected['starttime']+=1
    control=dict(driver=expected,parent_cgroup=parent if drift!='parent_scope' else '/foreign-topd/job_1')
    protocol.write_private_json(directory,'control.json',control)
    with (directory/'child.log').open('w') as output:
        child=subprocess.Popen([sys.executable,str(HERE/'linux-peer-fixture.py'),str(directory)],stdout=output,stderr=output,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        try:
            deadline=time.monotonic()+40
            ready=directory/'ready-private.json'
            while not ready.exists() and time.monotonic()<deadline and child.poll() is None:time.sleep(.05)
            assert ready.exists(),(directory/'child.log').read_text()
            manifest=json.loads(ready.read_text());cli=client.SharedTeacherClient(manifest,directory)
            if drift=='none':
                for step in [1,2]:
                    values=cli.score(tensors['prompts'],tensors['responses'],tensors['attention'],1.,16,step,transport_rows=2)
                    assert values.shape==tensors['responses'].shape and torch.allclose(values,reference,atol=1e-4,rtol=1e-4)
                    actual=cli.last_actual_server_peer
                    assert actual['pid']==child.pid and actual['pid']!=os.getpid()
                    assert actual['starttime']==manifest['service']['starttime']
                    assert cli.last_receipts[0]['actual_driver_peer']['pid']==os.getpid()
                    assert cli.last_receipts[0]['actual_service_identity']['pid']==child.pid
                assert cli.scored_tokens==50
            else:
                with pytest.raises((RuntimeError,ConnectionError,OSError)):
                    cli.score(tensors['prompts'],tensors['responses'],tensors['attention'],1.,16,1,transport_rows=2)
                assert cli.failed and cli.scored_tokens==0
            child.wait(timeout=20)
            evidence=json.loads((directory/'child-evidence.json').read_text())
            assert evidence['cuda_initialized'] is False
            assert evidence['stages']==evidence['releases']==(2 if drift=='none' else 0)
            assert child.returncode==(0 if drift=='none' else 1)
            assert not (Path('/proc')/str(child.pid)).exists()
            protocol.write_private_json(directory,'parent-observation.json',dict(scope='CPU_ONLY_NATIVE_PROCESS_EXIT_NOT_HARDWARE',pid=child.pid,returncode=child.returncode,actual_proc_remaining=False,driver_pid=os.getpid(),numeric_reference='DIRECT_ORIGINAL_FROZENTEACHER_CPU',atol=1e-4,rtol=1e-4))
            assert not (directory/'s.sock').exists()
            evidence_root=os.environ.get('TOPD_IPC_NATIVE_EVIDENCE_DIR')
            if evidence_root:
                import shutil
                target=Path(evidence_root)/('actual-peer-'+drift);target.mkdir(mode=0o700)
                # Private capability/control artifacts intentionally never published.
                for item in directory.iterdir():
                    if item.name in ('child-evidence.json','parent-observation.json','child.log') or item.name.startswith(('receipt-','partition-')):
                        shutil.copyfile(item,target/item.name);(target/item.name).chmod(0o600)
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=20)


def test_original_model_failure_releases_owned_round_and_publishes_nothing(fixture,monkeypatch):
    sess,_,_,directory,runtime=fixture;_,_,commit=assembled(fixture)
    def fail_model(*args,**kwargs):raise RuntimeError('OWN_CPU_FIXTURE_MODEL_FAILURE')
    monkeypatch.setattr(sess.scorer.teacher.model.model,'forward',fail_model)
    with pytest.raises(RuntimeError) as failure:sess.commit(commit,b'')
    raw=failure.value.args[0]
    assert raw['status']=='FAILED' and 'OWN_CPU_FIXTURE_MODEL_FAILURE' in raw['error']
    assert 'release_error' not in raw and raw['released_epoch']>=raw['staged_epoch']
    assert runtime.stages==runtime.releases==1
    assert sess.scorer.state=='RELEASED'
    assert not list(directory.glob('receipt-*.json'))
    sess.abort();assert sess.state=='FAILED' and not sess.chunks


def test_cpu_socket_disconnect_during_assembly_poisoned_no_stage(fixture,monkeypatch):
    sess,_,manifest,directory,runtime=fixture
    monkeypatch.setattr(server,'peer',fake_peer);monkeypatch.setattr(server,'process_identity',fake_self_identity)
    daemon=server.SharedUnixServer(sess);daemon.start();errors=[]
    def run():
        try:daemon.serve()
        except Exception as exc:errors.append(type(exc).__name__)
    thread=threading.Thread(target=run);thread.start()
    _,begin,_,_=requests(fixture)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as conn:
        conn.connect(manifest['service']['socket']);protocol.send(conn,begin)
        assert protocol.receive(conn)[0]['status']=='CPU_BEGIN_ACCEPTED'
    thread.join(timeout=5)
    assert not thread.is_alive() and errors and sess.state=='FAILED'
    assert runtime.stages==runtime.releases==0 and not sess.chunks and not (directory/'s.sock').exists()
