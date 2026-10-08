import hashlib, importlib.util, json, pathlib, threading, time
import pytest
spec=importlib.util.spec_from_file_location('download',pathlib.Path(__file__).with_name('download.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def item(data,lfs=True):
    value={'size':len(data),'blobId':hashlib.sha1(('blob %d\0'%len(data)).encode()+data).hexdigest()}
    if lfs:value['lfs']={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    return value
@pytest.mark.parametrize('lfs',[True,False])
def test_official_verify(tmp_path,lfs):
    p=tmp_path/'x';p.write_bytes(b'hello');assert m.verify(p,item(b'hello',lfs))['official_verified']
@pytest.mark.parametrize('lfs',[True,False])
def test_wrong_content_never_self_attested(tmp_path,lfs):
    p=tmp_path/'x';p.write_bytes(b'wrong')
    with pytest.raises(ValueError):m.verify(p,item(b'hello',lfs))
def test_metadata_exact_bytes():
    p=pathlib.Path(__file__).with_name('official-metadata.json');assert len(m.inventory(p))==24
def test_metadata_same_revision_forged(tmp_path):
    p=tmp_path/'metadata'; data=json.loads(pathlib.Path(__file__).with_name('official-metadata.json').read_bytes());data['siblings'][0]['size']=1;p.write_text(json.dumps(data))
    with pytest.raises(ValueError):m.inventory(p)
def test_resume_preserves_legacy(tmp_path):
    legacy=tmp_path/'x.resume-v1.transfer';legacy.write_bytes(b'old');partial=tmp_path/'x.h16proxy.partial';partial.write_bytes(b'ab')
    def run(n,p):assert p.read_bytes()==b'ab';p.write_bytes(b'abcdef')
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'abcdef')},run,sleeper=lambda _:None);assert d.execute();assert (tmp_path/'x').read_bytes()==b'abcdef';assert legacy.read_bytes()==b'old'
def test_failed_transfer_keeps_partial(tmp_path):
    def run(n,p):p.write_bytes(b'abc');raise RuntimeError('interrupted')
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'abcdef')},run,sleeper=lambda _:None);assert not d.execute();assert (tmp_path/'x.h16proxy.partial').read_bytes()==b'abc';assert not (tmp_path/'x').exists()
def test_wrong_full_partial_never_published(tmp_path):
    (tmp_path/'x.h16proxy.partial').write_bytes(b'wrong');d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'hello')},lambda *_:pytest.fail());assert not d.execute();assert not (tmp_path/'x').exists()
def test_existing_wrong_fails_before_network(tmp_path):
    (tmp_path/'x').write_bytes(b'wrong');d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'hello')},lambda *_:pytest.fail())
    with pytest.raises(ValueError):d.execute()
def test_publication_collision_no_overwrite(tmp_path):
    def run(n,p):p.write_bytes(b'hello');(tmp_path/'x').write_bytes(b'wrong')
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'hello')},run,sleeper=lambda _:None);assert not d.execute();assert (tmp_path/'x').read_bytes()==b'wrong';assert (tmp_path/'x.h16proxy.partial').read_bytes()==b'hello'
def test_all24_max3_and_commit(tmp_path):
    mu=threading.Lock();counts=[0,0];items={str(i):item(b'hello') for i in range(24)}
    def run(n,p):
        with mu:counts[0]+=1;counts[1]=max(counts)
        time.sleep(.005);p.write_bytes(b'hello')
        with mu:counts[0]-=1
    d=m.Downloader(tmp_path,tmp_path/'state',items,run);assert d.execute();assert counts[1]==3;assert len(json.loads((tmp_path/'state').read_text())['files'])==24
def test_symlink_rejected(tmp_path):
    (tmp_path/'x').symlink_to(tmp_path/'else')
    with pytest.raises(ValueError):m.check_regular(tmp_path/'x')
def test_proxy_https_pinned_env(monkeypatch):
    monkeypatch.setenv('ALL_PROXY','secret');assert 'ALL_PROXY' not in m.clean_env();cmd=m.curl_command('config.json',pathlib.Path('/tmp/x'),100);assert m.PROXY in cmd and '=https' in cmd and m.REVISION in cmd[-1];assert '--max-time' not in cmd
def test_handoff_live_pid_rejected(tmp_path):
    proc=tmp_path/'proc';proc.mkdir();(proc/'123').mkdir();p=tmp_path/'receipt';p.write_text(json.dumps({'scope':'OWN_H11_OFFICIAL_WRITER_HANDOFF','target':str(tmp_path),'process_identities':[{'pid':123,'ppid':1,'starttime':10}]}))
    with pytest.raises(ValueError):m.handoff_check(p,tmp_path,proc)
def test_handoff_real_absence(tmp_path):
    proc=tmp_path/'proc';proc.mkdir();p=tmp_path/'receipt';p.write_text(json.dumps({'scope':'OWN_H11_OFFICIAL_WRITER_HANDOFF','target':str(tmp_path),'process_identities':[{'pid':123,'ppid':1,'starttime':10}]}));assert m.handoff_check(p,tmp_path,proc)['actual_old_partial_fds']==[]

def test_transient_resume_retry(tmp_path):
    calls=[]
    def run(n,p):
        calls.append(p.read_bytes() if p.exists() else b'')
        if len(calls)==1:p.write_bytes(b'he');raise RuntimeError('curl transient')
        p.write_bytes(b'hello')
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'hello')},run,sleeper=lambda _:None);assert d.execute();assert calls==[b'',b'he']
def test_handoff_open_partial_rejected(tmp_path):
    proc=tmp_path/'proc';proc.mkdir();fd=proc/'123/fd';fd.mkdir(parents=True);(fd/'1').symlink_to(tmp_path/'x.resume-v1.transfer')
    p=tmp_path/'receipt';p.write_text(json.dumps({'scope':'OWN_H11_OFFICIAL_WRITER_HANDOFF','target':str(tmp_path),'process_identities':[{'pid':999,'ppid':1,'starttime':10}]}))
    with pytest.raises(ValueError):m.handoff_check(p,tmp_path,proc)

def test_many_http18_partial_offsets(tmp_path):
    offsets=[]
    def run(n,p):
        offsets.append(p.stat().st_size if p.exists() else 0)
        with open(p,'ab') as f:f.write(b'x')
        if len(offsets)<10:raise RuntimeError('curl failed rc=18')
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'x'*10)},run,sleeper=lambda _:None)
    assert d.execute();assert offsets==list(range(10));assert len(d.data['retries'])==9;assert d.data['complete'];assert d.data['files']['x']['remote_sha_verified']
def test_full_received_http18_verifies_without_retry(tmp_path):
    calls=[]
    def run(n,p):calls.append(1);p.write_bytes(b'hello');raise RuntimeError('curl failed rc=18')
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'hello')},run,sleeper=lambda _:None);assert d.execute();assert calls==[1]

def test_manifest_failure_never_complete(tmp_path,monkeypatch):
    original=m.atomic_json
    def fail(path,data):
        if str(path).endswith('.manifest.json'):raise OSError('fsync failure')
        original(path,data)
    monkeypatch.setattr(m,'atomic_json',fail)
    def run(n,p):p.write_bytes(b'hello')
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'hello')},run)
    with pytest.raises(OSError):d.execute()
    assert not d.data['complete'];assert not json.loads((tmp_path/'state').read_text())['complete']

def test_handoff_canonical_symlink_asset_root(tmp_path):
    actual=tmp_path/'actual';actual.mkdir();alias=tmp_path/'alias';alias.symlink_to(actual,target_is_directory=True)
    proc=tmp_path/'proc';proc.mkdir();fd=proc/'123/fd';fd.mkdir(parents=True);(fd/'1').symlink_to(str(actual/'x.resume-v1.transfer')+' (deleted)')
    p=tmp_path/'receipt';p.write_text(json.dumps({'scope':'OWN_H11_OFFICIAL_WRITER_HANDOFF','target':str(alias),'process_identities':[{'pid':999,'ppid':1,'starttime':10}]}))
    with pytest.raises(ValueError):m.handoff_check(p,alias,proc)

def test_real_owned_child_reaped_after_postspawn_save_failure(tmp_path,monkeypatch):
    import sys, subprocess
    partial=tmp_path/'x.h16proxy.partial'; children=[]; original=m.subprocess.Popen
    script="import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_bytes(b'preserved');time.sleep(30)"
    monkeypatch.setattr(m,'curl_command',lambda *args:[sys.executable,'-c',script,str(partial)])
    def spawn(*args,**kwargs):
        child=original(*args,**kwargs);children.append(child);return child
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'preserved-longer')})
    def save_failure():
        deadline=time.monotonic()+5
        while not partial.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert partial.exists();raise OSError('injected state fsync failure after spawn')
    monkeypatch.setattr(d,'save',save_failure)
    with pytest.raises(OSError,match='injected'):d.run_curl('x',partial)
    assert len(children)==1 and children[0].returncode is not None
    with pytest.raises(ChildProcessError):__import__('os').waitpid(children[0].pid,__import__('os').WNOHANG)
    assert partial.read_bytes()==b'preserved';assert d.data['curl_children'][0]['reaped']
def test_real_successful_child_not_terminated(tmp_path,monkeypatch):
    import sys
    partial=tmp_path/'x.h16proxy.partial'; original=m.subprocess.Popen; terminated=[]
    script="import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(b'hello')"
    monkeypatch.setattr(m,'curl_command',lambda *args:[sys.executable,'-c',script,str(partial)])
    def spawn(*args,**kwargs):
        child=original(*args,**kwargs);terminate=child.terminate
        def track():terminated.append(child.pid);return terminate()
        child.terminate=track;return child
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    d=m.Downloader(tmp_path,tmp_path/'state',{'x':item(b'hello')});d.run_curl('x',partial)
    assert not terminated;assert d.data['curl_children'][0]['returncode']==0;assert d.data['curl_children'][0]['reaped']
