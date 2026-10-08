"""PureCPU lifecycle/failure fixtures; no curl, SSH or current staging access."""
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import copy

import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import downloader_v6 as v6
import transfer


def fixture_entries(count=17):
    values={f'model-{i:05d}-of-00017.safetensors':bytes([i])*31 for i in range(1,count+1)}
    entries={n:dict(rfilename=n,size=len(raw),lfs=dict(sha256=hashlib.sha256(raw).hexdigest())) for n,raw in values.items()}
    return values,entries


class FakeBackend:
    def __init__(self,root,values,store,fail=None,point=None,free=10**12):
        self.root=root;self.values=values;self.store=store;self.fail=fail;self.point=point;self.free=free
        self.events=[];self.lock=threading.Lock();self.active=0;self.peak=0;self.peak_files=0;self.targets={}
    def record(self,name,phase):
        with self.lock:self.events.append((name,phase))
        if self.fail==name and self.point==phase:raise RuntimeError('CPU_FIXTURE_FAILURE_'+phase)
    def free_bytes(self):return self.free
    def download(self,name,item):
        with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
        self.record(name,'download');p=self.root/name
        if p.exists():assert self.values[name].startswith(p.read_bytes())
        p.write_bytes(self.values[name]);time.sleep(.002)
        with self.lock:self.peak_files=max(self.peak_files,len(list(self.root.glob('*.safetensors'))))
        proof=v6.source_proof(p,item);self.record(name,'sourceverified');return proof
    def relay(self,name,item,proof):
        self.record(name,'relay');self.targets[name]=self.values[name]
        return dict(status='VERIFIED_FINAL',sha256=proof['sha256'])
    def verify_target(self,name,item,expected_sha=None):
        self.record(name,'targetverified');assert self.targets[name]==self.values[name]
        return dict(status='TARGET_FULL_SHA_VERIFIED',size=len(self.values[name]),sha256=hashlib.sha256(self.values[name]).hexdigest())


def pipeline(tmp_path,count=17,fail=None,point=None,slots=3,free=10**12):
    root=tmp_path/'stage';root.mkdir()
    values,entries=fixture_entries(count);store=v6.StateStore(tmp_path/'state.json',dict(files={},complete=False))
    backend=FakeBackend(root,values,store,fail,point,free)
    p=v6.Pipeline(entries,list(entries),root,store,backend,max_slots=slots,reserve=2)
    original=p.unlink_verified
    def unlink(name,proof):
        assert store.path.exists() and json.loads(store.path.read_text())['files'][name]['remote_sha_verified']
        backend.record(name,'statecommitted');original(name,proof);backend.record(name,'unlinked')
        with backend.lock:backend.active-=1
    p.unlink_verified=unlink
    return p,backend,store,values,entries


def test_all17_three_slots_complete_verify_commit_unlink_before_next(tmp_path):
    p,b,store,values,_=pipeline(tmp_path);p.run()
    assert len(store.value['files'])==17 and b.peak<=3 and b.peak_files<=3
    assert not list(p.staging.glob('*.safetensors')) and p.active=={}
    events=b.events
    for name in values:
        assert [phase for n,phase in events if n==name]==['download','sourceverified','relay','targetverified','statecommitted','unlinked']
    occupancy=0
    for _,phase in events:
        if phase=='download':occupancy+=1
        if phase=='unlinked':occupancy-=1
        assert 0<=occupancy<=3
    assert occupancy==0


@pytest.mark.parametrize('slots',[1,2])
def test_configurable_smaller_transfer_pool_still_full17(tmp_path,slots):
    p,b,store,_,_=pipeline(tmp_path,slots=slots);p.run()
    assert len(store.value['files'])==17 and b.peak<=slots


@pytest.mark.parametrize('point',['download','sourceverified','relay','targetverified','statecommitted'])
def test_phase_failure_preserves_failed_source_no_new_slot_admission(tmp_path,point):
    fail='model-00001-of-00017.safetensors';p,b,store,values,_=pipeline(tmp_path,fail=fail,point=point)
    with pytest.raises(RuntimeError,match='CPU_FIXTURE_FAILURE'):p.run()
    assert p.failed.is_set() and store.value['failed_file']==fail
    started=[n for n,phase in b.events if phase=='download']
    assert len(started)<=3 and all(n in list(values)[:3] for n in started)
    if point!='download':assert (p.staging/fail).read_bytes()==values[fail]
    assert not store.value.get('complete') and ('unlinked' not in [phase for n,phase in b.events if n==fail])


def test_atomic_commit_failure_keeps_full_local_verified_file(tmp_path,monkeypatch):
    p,b,store,values,_=pipeline(tmp_path,count=1)
    def fail_commit(*_):raise RuntimeError('COMMIT_DURABILITY_FAILED')
    monkeypatch.setattr(store,'commit',fail_commit)
    with pytest.raises(RuntimeError,match='COMMIT_DURABILITY'):p.run()
    assert (p.staging/next(iter(values))).exists() and not store.value['files']
    assert not any(phase=='unlinked' for _,phase in b.events)


def test_capacity_all_three_reserved_before_admit(tmp_path):
    p,b,store,values,_=pipeline(tmp_path,count=3,free=1)
    with pytest.raises(RuntimeError,match='reserve'):p.run()
    assert not b.events and not list(p.staging.glob('*.safetensors'))


def test_three_existing_partials_have_priority_no_fourth_file(tmp_path):
    p,b,store,values,_=pipeline(tmp_path)
    for name in list(values)[10:13]:(p.staging/name).write_bytes(values[name][:5])
    p.run()
    started=[n for n,phase in b.events if phase=='download']
    assert set(started[:3])==set(list(values)[10:13]) and b.peak_files<=3
    assert len(store.value['files'])==17


def test_unknown_fourth_weight_source_rejected_without_delete(tmp_path):
    p,b,store,values,_=pipeline(tmp_path)
    unknown=p.staging/'other.safetensors';unknown.write_bytes(b'preserve')
    with pytest.raises(RuntimeError,match='Unknown'):p.run()
    assert unknown.read_bytes()==b'preserve' and not b.events


def test_source_lfs_sha_failure_preserves_copy(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'correct')
    with pytest.raises(RuntimeError,match='LFS'):v6.source_proof(p,dict(size=7,lfs=dict(sha256='0'*64)))
    assert p.read_bytes()==b'correct'


def test_small_official_git_blob_hash_is_verified(tmp_path):
    p=tmp_path/'config.json';raw=b'{"valid":true}';p.write_bytes(raw)
    git=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    proof=v6.source_proof(p,dict(size=len(raw),blobId=git))
    assert proof['official_lfs_sha256'] is None and proof['official_git_blob_sha1']==git
    with pytest.raises(RuntimeError,match='Git'):v6.source_proof(p,dict(size=len(raw),blobId='0'*40))


def test_unlink_refuses_replaced_source_identity(tmp_path):
    p,b,store,values,entries=pipeline(tmp_path,count=1);name=next(iter(values));path=p.staging/name;path.write_bytes(values[name]);proof=v6.source_proof(path,entries[name]);replacement=path.with_suffix('.replacement');replacement.write_bytes(values[name]);replacement.replace(path)
    with pytest.raises(RuntimeError,match='identity changed'):v6.Pipeline.unlink_verified(p,name,proof)
    assert path.exists()


def test_state_atomic_save_failure_preserves_previous_bytes(tmp_path,monkeypatch):
    store=v6.StateStore(tmp_path/'state.json',dict(files={},complete=False));store.save();before=store.path.read_bytes()
    def no_replace(*args):raise OSError('OWN_TEMP_RENAME_FAIL')
    monkeypatch.setattr(v6.os,'replace',no_replace)
    store.value['complete']=True
    with pytest.raises(OSError):store.save()
    assert store.path.read_bytes()==before and json.loads(before)['complete'] is False


def test_complete_metadata_bytes_changed_same_revision_failclosed(monkeypatch,tmp_path):
    root=tmp_path;out=root/'outputs';out.mkdir();metadata=dict(sha=transfer.REVISION,siblings=[dict(rfilename='model-00001-of-00017.safetensors',size=1,lfs=dict(sha256='0'*64))]);path=out/'qwen2.5-32b-official-metadata.json';path.write_text(json.dumps(metadata))
    monkeypatch.setattr(transfer,'BASE',root)
    with pytest.raises(RuntimeError,match='complete metadata bytes'):transfer.official_metadata()


@pytest.mark.parametrize('change',['ppid','starttime','booleanonly','livepid'])
def test_handoff_identity_and_actual_absence_not_boolean(monkeypatch,tmp_path,change):
    proof=dict(scope='LEGACY_TRANSFER_TREE_STOPPED_EXCLUSIVE_HANDOFF',all_legacy_subtree_exited=True,legacy_local_identities=[dict(pid=i,ppid=1,starttime='CPU_ONLY_NOT_ACTUAL') for i in [23765,53280,53396]],legacy_remote_identities=[])
    if change in ('ppid','starttime'):proof['legacy_local_identities'][0].pop(change)
    elif change=='booleanonly':proof.pop('legacy_local_identities')
    p=tmp_path/'handoff.json';p.write_text(json.dumps(proof))
    if change=='livepid':monkeypatch.setattr(transfer.os,'kill',lambda *a:None)
    else:
        def absent(*a):raise ProcessLookupError()
        monkeypatch.setattr(transfer.os,'kill',absent)
    with pytest.raises(RuntimeError):transfer.exclusive_handoff(p)


@pytest.mark.parametrize('tamper_prior',[False,True])
def test_complete_cli17_plus7_pipeline_single_parent_lock_and_idempotent_restart(tmp_path,monkeypatch,capsys,tamper_prior):
    values,entries=fixture_entries()
    for index,name in enumerate(v6.SMALL):
        raw=('metadata-'+str(index)).encode();values[name]=raw;entries[name]=dict(rfilename=name,size=len(raw),blobId=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    root=tmp_path/'root';out=root/'outputs';out.mkdir(parents=True);stage=tmp_path/'stage';stage.mkdir();handoff=tmp_path/'handoff.json';handoff.write_text('{}')
    monkeypatch.setattr(transfer,'BASE',root);monkeypatch.setattr(transfer,'STAGING',stage);monkeypatch.setattr(v6,'STATE',out/'v6.json')
    monkeypatch.setattr(transfer,'exclusive_handoff',lambda _:dict(scope='CPU_FIXTURE_SIMULATED_HANDOFF'))
    monkeypatch.setattr(transfer,'official_metadata',lambda:dict(sha=transfer.REVISION,siblings=list(entries.values())))
    monkeypatch.setattr(v6.os,'getpid',lambda:999999)
    manifests=[];downloaded=[];targets={};lock_checks=[]
    class CompleteBackend:
        def __init__(self,staging):self.staging=staging
        def free_bytes(self):return 10**12
        def download(self,name,item):
            # Parent retains the exact original flock throughout; no child CLI
            # or a second self-lock acquisition is part of this path.
            with (stage/'download.lock').open('a') as lock:
                try:v6.fcntl.flock(lock,v6.fcntl.LOCK_EX|v6.fcntl.LOCK_NB)
                except BlockingIOError:lock_checks.append(True)
                else:raise AssertionError('Parent orchestration lock not held')
            downloaded.append(name);p=stage/name;p.write_bytes(values[name]);return v6.source_proof(p,item)
        def relay(self,name,item,proof):targets[name]=values[name];return dict(status='VERIFIED_FINAL')
        def verify_target(self,name,item,expected_sha=None):
            assert name in targets
            raw=targets[name];actual=hashlib.sha256(raw).hexdigest()
            assert not expected_sha or expected_sha==actual
            return dict(status='TARGET_FULL_SHA_VERIFIED',size=len(raw),sha256=actual)
        def publish_manifest(self,manifest):
            manifests.append(copy.deepcopy(manifest));return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    monkeypatch.setattr(v6,'OfficialBackend',CompleteBackend)
    monkeypatch.setattr(sys,'argv',['downloader_v6.py','--exclusive-handoff',str(handoff)])
    v6.main();state=json.loads(v6.STATE.read_text())
    assert state['complete'] and len(state['files'])==len(downloaded)==24
    assert len(lock_checks)==24 and not list(stage.glob('*.safetensors'))
    assert all(not (stage/name).exists() for name in entries)
    before=list(downloaded)
    if tamper_prior:
        bad_name=next(iter(entries));state['files'][bad_name]['sha256']='0'*64
        # Equal-size nonofficial remote + falsely verified prior, no local source.
        targets[bad_name]=b'X'*len(values[bad_name]);v6.STATE.write_text(json.dumps(state))
        with pytest.raises(RuntimeError,match='trusted official LFS'):v6.main()
        assert downloaded==before and len(manifests)==1 and not (stage/bad_name).exists()
    else:
        v6.main()
        assert downloaded==before and manifests[0]==manifests[1]
    assert 'NOT_EVALUATED' in capsys.readouterr().out


def test_handoff_old_remote_pid_presence_failsclosed(monkeypatch,tmp_path):
    proof=dict(scope='LEGACY_TRANSFER_TREE_STOPPED_EXCLUSIVE_HANDOFF',legacy_local_identities=[dict(pid=i,ppid=1,starttime='CPU_FIXTURE_IDENTITY_ONLY') for i in [23765,53280,53396]],legacy_remote_identities=[dict(host='h100-11',pid=1234,ppid=1,starttime='CPU_FIXTURE_IDENTITY_ONLY')])
    p=tmp_path/'handoff.json';p.write_text(json.dumps(proof))
    def absent(*a):raise ProcessLookupError()
    monkeypatch.setattr(transfer.os,'kill',absent);calls=[]
    def fail_remote(*args,**kwargs):calls.append(args);return type('R',(),dict(returncode=1,stdout=''))()
    monkeypatch.setattr(transfer.subprocess,'run',fail_remote)
    with pytest.raises(RuntimeError,match='remote process absence'):transfer.exclusive_handoff(p)
    assert len(calls)==1


def test_expected_real_complete_official_metadata_bytes_match():
    metadata=transfer.official_metadata()
    assert metadata['sha']==transfer.REVISION
    shards=[x for x in metadata['siblings'] if x['rfilename'].endswith('.safetensors')]
    assert len(shards)==17 and all(len(x['lfs']['sha256'])==64 for x in shards)


def test_official_verify_target_never_uses_prior_in_place_of_official(monkeypatch,tmp_path):
    called=[]
    monkeypatch.setattr(v6.subprocess,'run',lambda *a,**kw:called.append(a))
    with pytest.raises(RuntimeError,match='official LFS'):
        v6.OfficialBackend(tmp_path).verify_target('model-00001-of-00017.safetensors',dict(size=9,lfs=dict(sha256='a'*64)),'b'*64)
    assert not called


@pytest.mark.parametrize('operation',['git_blob_verify','manifest_publish_resume'])
def test_actual_generated_remote_python_scripts_cpu_only(tmp_path,monkeypatch,operation):
    import contextlib,io,shlex,socket
    from types import SimpleNamespace
    monkeypatch.setattr(transfer,'REMOTE',str(tmp_path))
    monkeypatch.setattr(transfer,'route',lambda command:['CPU_FIXTURE_ONLY',command])
    def execute_generated(argv,**kwargs):
        parts=shlex.split(argv[1]);stream=io.StringIO();code=0
        with monkeypatch.context() as scoped:
            scoped.setattr(sys,'argv',[parts[0]]+parts[3:]);scoped.setattr(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(kwargs.get('input',b''))))
            scoped.setattr(socket,'gethostname',lambda:'h100-11')
            try:
                with contextlib.redirect_stdout(stream):exec(parts[2],{'__name__':'__main__'})
            except AssertionError:code=1
        output=stream.getvalue()
        return SimpleNamespace(returncode=code,stdout=output if kwargs.get('text') else output.encode())
    monkeypatch.setattr(v6.subprocess,'run',execute_generated)
    backend=v6.OfficialBackend(tmp_path)
    if operation=='git_blob_verify':
        raw=b'{"official":true}';name='config.json';(tmp_path/name).write_bytes(raw)
        item=dict(size=len(raw),blobId=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
        result=backend.verify_target(name,item,hashlib.sha256(raw).hexdigest())
        assert result['official_git_blob_sha1']==item['blobId']
        with pytest.raises(RuntimeError):backend.verify_target(name,dict(item,blobId='0'*40),hashlib.sha256(raw).hexdigest())
    else:
        manifest=dict(scope='CPU_FIXTURE_NO_NETWORK',files={});raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
        partial=tmp_path/'official-download-manifest-v6.json.publish-v6.transfer';partial.write_bytes(raw[:13])
        digest=backend.publish_manifest(manifest)
        assert (tmp_path/'official-download-manifest-v6.json').read_bytes()==raw and not partial.exists()
        assert digest==hashlib.sha256(raw).hexdigest()==backend.publish_manifest(manifest)
