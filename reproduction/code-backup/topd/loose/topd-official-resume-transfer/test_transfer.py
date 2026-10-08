"""CPU-only receiver tests: no SSH/downloads/running-transfer modifications."""
import hashlib
import importlib.util
import io
from pathlib import Path
import concurrent.futures
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resume_candidate',HERE/'transfer.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
namespace={};exec(candidate.RECEIVER,namespace);receive=namespace['execute']


def sha(raw):return hashlib.sha256(raw).hexdigest()
def command(root,data,op,offset=0,raw=b'',name='model-00001-of-00017.safetensors',**kw):
    return receive(root,name,len(data),sha(data),op,offset,len(raw),sha(raw),io.BytesIO(raw),**kw)


def test_two_append_chunks_reprobe_and_full_official_finalize(tmp_path):
    data=b'official'*1000;name='model-00001-of-00017.safetensors'
    assert command(tmp_path,data,'probe')['size']==0
    assert command(tmp_path,data,'append',raw=data[:321])['size']==321
    probe=command(tmp_path,data,'probe');assert probe['size']==321 and probe['sha256']==sha(data[:321])
    command(tmp_path,data,'append',offset=321,raw=data[321:])
    assert command(tmp_path,data,'finalize',offset=len(data))['status']=='VERIFIED_FINAL'
    assert (tmp_path/name).read_bytes()==data
    assert not (tmp_path/(name+'.resume-v1.transfer')).exists()
    assert command(tmp_path,data,'probe')['status']=='VERIFIED_FINAL'


def test_interrupted_append_resumes_exact_prefix_not_zero(tmp_path):
    data=b'abcdefg'*100;name='model-00001-of-00017.safetensors'
    with pytest.raises(RuntimeError,match='Interrupted'):
        receive(tmp_path,name,len(data),sha(data),'append',0,100,sha(data[:100]),io.BytesIO(data[:37]))
    probe=command(tmp_path,data,'probe');assert probe['size']==37 and probe['sha256']==sha(data[:37])
    command(tmp_path,data,'append',offset=37,raw=data[37:]);command(tmp_path,data,'finalize',offset=len(data))
    assert (tmp_path/name).read_bytes()==data


def test_bad_chunk_rolls_back_only_owned_new_bytes(tmp_path):
    data=b'abcdefg'*100;name='model-00001-of-00017.safetensors'
    command(tmp_path,data,'append',raw=data[:10])
    with pytest.raises(RuntimeError,match='Chunk SHA'):
        receive(tmp_path,name,len(data),sha(data),'append',10,20,sha(data[10:30]),io.BytesIO(b'X'*20))
    assert (tmp_path/(name+'.resume-v1.transfer')).read_bytes()==data[:10]


@pytest.mark.parametrize('reason',['offset','oversize','finalsha','symlink'])
def test_fail_closed_preserves_own_partial(tmp_path,reason):
    data=b'123456789';name='model-00001-of-00017.safetensors';p=tmp_path/(name+'.resume-v1.transfer')
    p.write_bytes(data[:3]);before=p.read_bytes()
    if reason=='symlink':p.unlink();p.symlink_to(tmp_path/'foreign')
    elif reason=='oversize':p.write_bytes(data*2)
    with pytest.raises(RuntimeError):
        if reason=='offset':command(tmp_path,data,'append',offset=0,raw=data)
        elif reason=='finalsha':command(tmp_path,data,'finalize',offset=3)
        else:command(tmp_path,data,'probe')
    if reason in ('offset','finalsha'):assert p.read_bytes()==before
    assert not (tmp_path/name).exists()


@pytest.mark.parametrize('reason',['nonexclusive','wrongprefix','alreadyresume','pass'])
def test_exclusive_legacy_adoption_never_overwrites_live_candidate(tmp_path,reason):
    data=b'officialweights'*100;name='model-00001-of-00017.safetensors'
    old=tmp_path/(name+'.transfer');old.write_bytes(data[:35])
    if reason=='alreadyresume':(tmp_path/(name+'.resume-v1.transfer')).write_bytes(b'owned')
    probe=command(tmp_path,data,'probe',adopt_legacy=True)
    if reason!='alreadyresume':assert probe['status']=='LEGACY_PREFIX_REQUIRES_EXCLUSIVE_ADOPT'
    kwargs=dict(adopt_legacy=True,exclusive=reason!='nonexclusive',prefix_sha=sha(data[:35]) if reason!='wrongprefix' else '0'*64)
    if reason=='pass':
        result=command(tmp_path,data,'adopt',offset=35,**kwargs)
        assert result['size']==35 and not old.exists()
        command(tmp_path,data,'append',offset=35,raw=data[35:]);command(tmp_path,data,'finalize',offset=len(data))
        assert (tmp_path/name).read_bytes()==data
    else:
        with pytest.raises(RuntimeError):command(tmp_path,data,'adopt',offset=35,**kwargs)
        assert old.read_bytes()==data[:35]


def test_three_independent_files_threaded_cpu_roundtrip(tmp_path):
    def one(rank):
        data=bytes([rank])*10000;name=f'model-{rank:05d}-of-00017.safetensors'
        command(tmp_path,data,'append',raw=data[:100],name=name)
        command(tmp_path,data,'append',offset=100,raw=data[100:],name=name)
        command(tmp_path,data,'finalize',offset=len(data),name=name)
        return (tmp_path/name).read_bytes()==data
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:assert all(pool.map(one,[1,2,3]))


def test_main_exclusive_live_legacy_is_not_started(monkeypatch,tmp_path):
    proof=tmp_path/'handoff.json';proof.write_text(__import__('json').dumps(dict(scope='LEGACY_TRANSFER_TREE_STOPPED_EXCLUSIVE_HANDOFF',legacy_local_identities=[dict(pid=p,ppid=1,starttime='ACTUAL_FIXTURE_ONLY') for p in [23765,53280,53396]],legacy_remote_identities=[])))
    monkeypatch.setattr(candidate.sys,'argv',['transfer.py','--exclusive-handoff',str(proof),'model-00001-of-00017.safetensors'])
    monkeypatch.setattr(candidate.os,'kill',lambda *args:None)
    with pytest.raises(RuntimeError,match='still exists'):candidate.main()


def test_prefix_digest_is_equal_length_and_short_local_fails(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'abcdef')
    assert candidate.digest_file(p,3)==sha(b'abc')
    with pytest.raises(RuntimeError,match='shorter'):candidate.digest_file(p,7)


def test_relay_remote_wrong_prefix_never_appends_or_deletes_local(monkeypatch,tmp_path):
    data=b'official';name='model-00001-of-00017.safetensors';source=tmp_path/name;source.write_bytes(data)
    monkeypatch.setattr(candidate,'STAGING',tmp_path);monkeypatch.setattr(candidate,'legacy_guard',lambda _:None)
    calls=[]
    def probe(*args,**kwargs):
        calls.append(args[2]);return dict(status='PARTIAL',size=3,sha256=sha(b'bad'))
    monkeypatch.setattr(candidate,'remote_call',probe)
    with pytest.raises(RuntimeError,match='prefix mismatch'):candidate.relay(name,dict(size=len(data),lfs=dict(sha256=sha(data))))
    assert calls==['probe'] and source.read_bytes()==data


def test_existing_final_wrong_official_sha_never_overwritten(tmp_path):
    data=b'official';name='model-00001-of-00017.safetensors';p=tmp_path/name;p.write_bytes(b'XXXXXXX')
    with pytest.raises(RuntimeError,match='final official'):command(tmp_path,data,'probe')
    assert p.read_bytes()==b'XXXXXXX' and not (tmp_path/(name+'.resume-v1.transfer')).exists()
