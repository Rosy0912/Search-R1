"""Real TinyQwen eight local Unix sockets; CPU only, no production acceptance."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import threading
import sys
from unittest.mock import patch
import pytest
import torch

repo=(Path(os.environ['AGENT_ROOT'])/'repo') if os.environ.get('AGENT_ROOT') else Path(__file__).parent/'topd'
sys.path.insert(0,str(repo))
from search_r1.baselines import topd_local_gpu as ipc
from search_r1.baselines.distillation import FrozenTeacher


@pytest.fixture
def tiny(tmp_path):
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    config=Qwen2Config(vocab_size=3,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,bos_token_id=0,eos_token_id=1,tie_word_embeddings=True)
    model=Qwen2ForCausalLM(config)
    raw=Tokenizer(WordLevel({'<bos>':0,'<eos>':1,'<unk>':2},unk_token='<unk>'));raw.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,bos_token='<bos>',eos_token='<eos>',unk_token='<unk>',pad_token='<unk>')
    model.save_pretrained(tmp_path);tokenizer.save_pretrained(tmp_path)
    return tmp_path,tokenizer


def manifest(root,teacher,rows=2):
    return dict(kind='CPU_FIXTURE_NO_ALGORITHM',backend='local_gpu_fp32_parallel',execution_device='cpu',teacher_path=str(teacher),run_id='fixture',code_commit='a'*40,config_sha256='b'*64,data_identity='c'*64,teacher_receipt_sha256='d'*64,source_sha256='e'*64,cgroup='/fixture',outer_updates=1,rows_per_service=rows,temperature=1.,chunk=16,timeout_seconds=20,
                services=[dict(uuid=f'GPU-fixture-{rank}',socket=str(root/f's{rank}'),capability=str(rank),pid=os.getpid()) for rank in range(8)])


def cpu_release(teacher):
    teacher.model.to('cpu');teacher.device=torch.device('cpu')
    return dict(allocated_bytes=0,reserved_bytes=0)


def test_real_eight_socket_parallel_scores_preserve_every_row_and_cpu_math(tiny):
    teacher_path,tokenizer=tiny
    with tempfile.TemporaryDirectory(prefix='t8-',dir='/tmp') as directory:
        root=Path(directory);record=manifest(root,teacher_path);record['outer_updates']=2;record['student_tokenizer']=ipc.tokenizer_identity(tokenizer);path=root/'manifest';path.write_text(json.dumps(record))
        errors=[];threads=[];sockets_ready=[]
        for rank in range(8):
            teacher=FrozenTeacher(str(teacher_path),tokenizer,'cpu')
            session=ipc.ScorerSession(teacher,record,rank,record['services'][rank]['uuid'],lambda:None,device='cpu',release=cpu_release)
            def run(session=session,rank=rank):
                try:ipc.serve(session,root/f's{rank}',lambda connection:None,root/f'r{rank}',persistent=False)
                except Exception as error:errors.append(error)
            thread=threading.Thread(target=run,daemon=True);thread.start();threads.append(thread)
        import time
        deadline=time.time()+10
        while not all((root/f's{rank}').exists() for rank in range(8)):
            assert time.time()<deadline;time.sleep(.01)
        prompts=torch.tensor([[2,0,2]]*16);responses=torch.tensor([[1,0] if row%2 else [0,1] for row in range(16)]);attention=torch.ones_like(responses);attention[:,1]=(torch.arange(16)%3!=0).long()
        client=ipc.LocalTeacherClient(path,teacher_path,required_kind='CPU_FIXTURE_NO_ALGORITHM')
        result=client.score(prompts,responses,attention,1.,16,1)
        second=client.score(prompts.flip(0),responses.flip(0),attention.flip(0),1.,16,2)
        reference=FrozenTeacher(str(teacher_path),tokenizer,'cpu')
        expected=torch.cat([reference.log_probs(prompts[row:row+1],responses[row:row+1],attention[row:row+1],1.,None,16) for row in range(16)])
        assert torch.equal(result,expected) and torch.equal(second,expected.flip(0)) and client.scored_tokens==2*reference.scored_tokens
        from search_r1.baselines.advantages import topd
        old=torch.linspace(-2.,-.1,32).reshape(16,2)
        uids=[row%4 for row in range(16)]  # Every UID crosses scorer partitions.
        actual_adv=topd(result,old,attention,uids,.1)
        expected_adv=topd(expected,old,attention,uids,.1)
        assert all(torch.equal(actual,expected) for actual,expected in zip(actual_adv,expected_adv))
        assert len(client.last_receipts)==8 and all(r['status']=='RELEASED' for r in client.last_receipts)
        for thread in threads:thread.join(10);assert not thread.is_alive()
        assert not errors


def request(session):
    rows=session.manifest['rows_per_service'];rank=session.rank
    raw=ipc.tensor_save(dict(prompts=torch.tensor([[0]]*rows),responses=torch.tensor([[1,0]]*rows),attention=torch.ones(rows,2,dtype=torch.int64),row_indices=torch.arange(rank*rows,(rank+1)*rows)))
    metadata={key:session.manifest[key] for key in ipc.IDENTITY}
    metadata.update(capability=session.manifest['services'][rank]['capability'],outer_step=1,temperature=1.,chunk=16,input_sha256=ipc.digest(raw))
    return metadata,raw


@pytest.mark.parametrize('change',['gold_field','gold_tensor','identity','hash','order','release_workspace'])
def test_privacy_identity_order_and_release_fail_closed(tiny,tmp_path,change):
    path,tokenizer=tiny;record=manifest(tmp_path,path)
    teacher=FrozenTeacher(str(path),tokenizer,'cpu')
    release=cpu_release if change!='release_workspace' else lambda teacher:dict(allocated_bytes=33554432,reserved_bytes=33554432)
    session=ipc.ScorerSession(teacher,record,0,record['services'][0]['uuid'],lambda:None,'cpu',release)
    metadata,raw=request(session)
    if change=='gold_field':metadata['accepted_answers']=['never transmit']
    elif change=='gold_tensor':
        values=ipc.tensor_load(raw);values['reward_model']=torch.tensor([1]);raw=ipc.tensor_save(values);metadata['input_sha256']=ipc.digest(raw)
    elif change=='identity':metadata['run_id']='foreign'
    elif change=='hash':metadata['input_sha256']='0'*64
    elif change=='order':
        values=ipc.tensor_load(raw);values['row_indices']=torch.tensor([1,0]);raw=ipc.tensor_save(values);metadata['input_sha256']=ipc.digest(raw)
    update_actor=[]
    with pytest.raises(RuntimeError):
        session.score(metadata,raw)
        update_actor.append('must never execute')
    assert not update_actor and session.next_step==1


def test_production_client_rejects_fixture_kind(tiny,tmp_path):
    path,_=tiny;file=tmp_path/'manifest';file.write_text(json.dumps(manifest(tmp_path,path)))
    with pytest.raises(RuntimeError,match='Wrong teacher backend/kind'):ipc.LocalTeacherClient(file,path)


def test_oversized_frames_reject_before_reading():
    with pytest.raises(RuntimeError,match='frame exceeds'):ipc.receive_exact(None,ipc.MAX_FRAME+1)


def test_controller_gpu_failure_stops_before_advantages_and_actor(tiny):
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from search_r1.baselines.controller import Controller
    from search_r1.baselines import advantages
    controller=Controller.__new__(Controller);controller.method='topd'
    controller.config=OmegaConf.create(dict(topd=dict(alpha=.1,teacher_backend='local_gpu_fp32_parallel'),distillation=dict(chunk_size=16,privileged_answers=False)))
    controller.trainer=SimpleNamespace(config=OmegaConf.create(dict(actor_rollout_ref=dict(rollout=dict(temperature=1.)))))
    controller.teacher=SimpleNamespace(scored_tokens=0,score=lambda *args:(_ for _ in ()).throw(RuntimeError('no RELEASED')))
    data=SimpleNamespace(batch=dict(prompts=torch.zeros(16,1,dtype=torch.long),responses=torch.ones(16,2,dtype=torch.long),attention_mask=torch.ones(16,3),info_mask=torch.ones(16,3),token_level_rewards=torch.ones(16,2),old_log_probs=torch.zeros(16,2)),non_tensor_batch=dict(uid=list(range(16))),meta_info=dict(topd_outer_step=1))
    with patch.object(advantages,'topd') as objective,patch.dict(os.environ,{},clear=True):
        with pytest.raises(RuntimeError,match='no RELEASED'):controller.compute_advantages(data,{})
        objective.assert_not_called()


def test_partial_model_move_failure_still_attempts_release(tiny,tmp_path):
    path,tokenizer=tiny;record=manifest(tmp_path,path)
    teacher=FrozenTeacher(str(path),tokenizer,'cpu');released=[]
    def release(teacher):
        released.append(True)
        return dict(allocated_bytes=0,reserved_bytes=0)
    session=ipc.ScorerSession(teacher,record,0,record['services'][0]['uuid'],lambda:None,'cpu',release)
    metadata,raw=request(session)
    with patch.object(teacher.model,'to',side_effect=RuntimeError('partial H2D')):
        with pytest.raises(RuntimeError,match='partial H2D'):session.score(metadata,raw)
    assert released==[True] and session.next_step==1


def test_lease_checks_actual_fd_open_description_and_inode(tmp_path):
    import fcntl
    directory=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'
    directory.mkdir(exist_ok=True)
    uuid='GPU-fixture-'+os.urandom(8).hex();path=directory/(uuid+'.lock')
    owner=os.open(path,os.O_RDWR|os.O_CREAT,0o600)
    foreign=os.open(path,os.O_RDWR)
    try:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ipc.lease_ownership(dict(uuid=uuid,fd=owner))
        with pytest.raises(RuntimeError,match='descriptor is foreign'):ipc.lease_ownership(dict(uuid=uuid,fd=foreign))
        fcntl.flock(owner,fcntl.LOCK_UN)
        with pytest.raises(RuntimeError,match='not held'):ipc.lease_ownership(dict(uuid=uuid,fd=owner))
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with tempfile.TemporaryFile() as other:
            with pytest.raises(RuntimeError,match='file mismatch'):ipc.lease_ownership(dict(uuid=uuid,fd=other.fileno()))
    finally:
        os.close(foreign);os.close(owner);path.unlink()


@pytest.mark.skipif(not hasattr(__import__('socket'),'SO_PEERCRED'),reason='Linux actual Unix peer credentials required')
def test_linux_actual_socket_peer_pid_cgroup_and_foreign_scope_rejection():
    import socket
    scopes=[row.split(':',2)[-1] for row in Path('/proc/self/cgroup').read_text().splitlines()]
    first,second=socket.socketpair()
    try:
        assert ipc.own_peer(first,scopes[0])==os.getpid()
        with pytest.raises(RuntimeError,match='outside own Slurm cgroup'):ipc.own_peer(first,'/impossible/foreign/job')
    finally:first.close();second.close()


def test_cuda_context_lifecycle_allocates_syncs_deletes_then_resets_before_snapshot():
    events=[]
    class ContextToken:
        def __del__(self):events.append('delete')
    def allocate(*args,**kwargs):
        assert args==((),) and kwargs==dict(device=torch.device('cuda:0'),dtype=torch.float32)
        events.append('create-visible-context');return ContextToken()
    with patch.object(ipc.torch,'empty',side_effect=allocate),patch.object(ipc.torch.cuda,'synchronize',side_effect=lambda:events.append('sync')),patch.object(ipc.torch.cuda,'empty_cache',side_effect=lambda:events.append('empty')),patch.object(ipc.torch.cuda,'reset_peak_memory_stats',side_effect=lambda:events.append('reset')):
        ipc.initialize_cuda_context(torch.device('cuda:0'))
    assert events==['create-visible-context','sync','delete','empty','reset']


def test_context_setup_or_pre_h2d_snapshot_failure_still_releases_before_return(tiny,tmp_path):
    path,tokenizer=tiny;record=manifest(tmp_path,path)
    teacher=FrozenTeacher(str(path),tokenizer,'cpu');released=[]
    def release(teacher):
        released.append(True);return dict(allocated_bytes=0,reserved_bytes=0)
    session=ipc.ScorerSession(teacher,record,0,record['services'][0]['uuid'],lambda:None,'cuda:0',release)
    metadata,raw=request(session)
    for setup_error in (True,False):
        released.clear()
        with patch.object(ipc,'initialize_cuda_context',side_effect=RuntimeError('setup failed') if setup_error else None),patch.object(ipc,'cuda_process_snapshot',side_effect=RuntimeError('NVML absent')),patch.object(teacher.model,'to') as move:
            with pytest.raises(RuntimeError,match='setup failed|NVML absent'):session.score(metadata,raw)
            move.assert_not_called()
        assert released==[True] and session.next_step==1 and teacher.device.type=='cpu'


def test_actual_nvml_snapshot_still_rejects_wrong_uuid_after_context():
    import subprocess
    from unittest.mock import patch
    with patch.object(subprocess,'check_output',return_value=f'GPU-foreign, {os.getpid()}, 300\n'):
        with pytest.raises(RuntimeError,match='foreign UUID'):ipc.cuda_process_snapshot('GPU-leased')
