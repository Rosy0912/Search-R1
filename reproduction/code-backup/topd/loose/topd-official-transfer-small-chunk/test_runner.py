"""CPU-only real chunk-boundary/prefix preservation fixtures; no SSH or downloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('small_chunk_adapter',HERE/'runner.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


def digest(raw):return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('mib',[1,2,4,8])
def test_exact_chunk_choice_and_unchanged_receiver_64mib_bound(mib):
    transfer=SimpleNamespace(CHUNK=64*1024**2)
    config=adapter.set_small_chunk(transfer,mib)
    assert transfer.CHUNK==mib*1024**2 and config['timeout_seconds']==600
    assert config['frozen_receiver_max_bytes']==64*1024**2
    assert config['sustained_raw_rate_to_complete_within_timeout_mib_s']==mib/600.


@pytest.mark.parametrize('value',[0,16,64,1.0,True])
def test_illegal_chunk_no_mutation(value):
    transfer=SimpleNamespace(CHUNK=64*1024**2)
    with pytest.raises(RuntimeError):adapter.set_small_chunk(transfer,value)
    assert transfer.CHUNK==64*1024**2


def test_real_8mib_boundary_plus_one_byte_relay_complete_official_sha(tmp_path):
    import sys
    transfer,v6=adapter.frozen_modules()
    original_chunk=transfer.CHUNK;adapter.set_small_chunk(transfer,8)
    size=8*1024**2+1;data=(b'FROZEN_OFFICIAL_TRANSPORT_CPU_ONLY'*300000)[:size];assert len(data)==size
    namespace={};exec(transfer.RECEIVER,namespace);execute=namespace['execute'];name='model-00001-of-00017.safetensors';(tmp_path/name).write_bytes(data)
    target=tmp_path/'remote-fixture';target.mkdir();calls=[]
    def local_receive(name,item,operation,offset=0,raw=b'',prefix_sha=None):
        calls.append((operation,offset,len(raw)))
        return execute(target,name,item['size'],item['lfs']['sha256'],operation,offset,len(raw),digest(raw),io.BytesIO(raw))
    monkey=pytest.MonkeyPatch()
    try:
        monkey.setattr(transfer,'STAGING',tmp_path);monkey.setattr(transfer,'remote_call',local_receive);monkey.setattr(transfer,'legacy_guard',lambda _:None)
        result=transfer.relay(name,dict(size=size,lfs=dict(sha256=digest(data))))
        assert result['status']=='VERIFIED_FINAL'
        assert [(o,n) for op,o,n in calls if op=='append']==[(0,8*1024**2),(8*1024**2,1)]
        assert (target/name).read_bytes()==data and (tmp_path/name).read_bytes()==data
    finally:monkey.undo();transfer.CHUNK=original_chunk


def test_interrupted8mib_preserves_prefix_then_next_smaller_tail(tmp_path):
    transfer,_=adapter.frozen_modules();namespace={};exec(transfer.RECEIVER,namespace);execute=namespace['execute']
    size=8*1024**2+7;data=b'X'*size;name='model-00002-of-00017.safetensors';written=2*1024**2+19
    with pytest.raises(RuntimeError,match='Interrupted'):
        execute(tmp_path,name,size,digest(data),'append',0,8*1024**2,digest(data[:8*1024**2]),io.BytesIO(data[:written]))
    probe=execute(tmp_path,name,size,digest(data),'probe',0,0,digest(b''),io.BytesIO())
    assert probe['size']==written and probe['sha256']==digest(data[:written])
    tail=data[written:];ack=execute(tmp_path,name,size,digest(data),'append',written,len(tail),digest(tail),io.BytesIO(tail))
    assert ack['size']==size
    result=execute(tmp_path,name,size,digest(data),'finalize',size,0,digest(b''),io.BytesIO())
    assert result['status']=='VERIFIED_FINAL' and (tmp_path/name).read_bytes()==data


def test_current_v6_live_or_uncaptured_parent_no_adapter_start(tmp_path,monkeypatch):
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(pid=41001)))
    proof=tmp_path/'handoff.json';proof.write_text(json.dumps(dict(legacy_local_identities=[])))
    with pytest.raises(RuntimeError,match='missing'):adapter.require_current_handoff(SimpleNamespace(STATE=state),proof)
    proof.write_text(json.dumps(dict(legacy_local_identities=[dict(pid=41001,ppid=1,starttime='CPU_FIXTURE_ONLY')])))
    monkeypatch.setattr(adapter.os,'kill',lambda *args:None)
    with pytest.raises(RuntimeError,match='still alive'):adapter.require_current_handoff(SimpleNamespace(STATE=state),proof)
