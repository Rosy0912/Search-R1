"""External complete official 17-shard downloader candidate. No execution at import."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import threading
import time
import uuid

import transfer

SMALL=('config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors.index.json')
RESERVE=2*1024**3
STATE=transfer.BASE/'outputs/qwen25-32b-download-v6-state.json'
SCOPE='OFFICIAL_SINGLE_CLUSTER_RESUMABLE_TRANSPORT_V6_NO_ALGORITHM'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def source_proof(path,item):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Own complete regular source required')
    stat=path.stat();require(stat.st_uid==os.getuid() and stat.st_size==item['size'],'Official source owner/size differs')
    sha=hashlib.sha256();git=hashlib.sha1();git.update(b'blob '+str(item['size']).encode()+b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):sha.update(block);git.update(block)
    expected=(item.get('lfs') or {}).get('sha256')
    require(sha.hexdigest()==expected if expected else git.hexdigest()==item['blobId'],'Official LFS/Git blob source identity differs')
    require(path.stat().st_ino==stat.st_ino and path.stat().st_size==stat.st_size and path.stat().st_mtime_ns==stat.st_mtime_ns,'Source mutated while hashing')
    return dict(size=stat.st_size,sha256=sha.hexdigest(),official_lfs_sha256=expected,official_git_blob_sha1=None if expected else git.hexdigest(),source_inode=stat.st_ino,source_device=stat.st_dev,source_mtime_ns=stat.st_mtime_ns)


class StateStore:
    def __init__(self,path,initial):
        self.path=Path(path);self.value=initial;self.lock=threading.RLock()
        require(not self.path.is_symlink(),'State symlink refused')
    def save(self):
        self.value['updated_unix_seconds']=time.time();temporary=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.partial')
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'w') as stream:json.dump(self.value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,self.path)
        fd=os.open(self.path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def commit(self,name,proof):
        with self.lock:
            self.value['files'][name]=proof;self.save()


def validate_expected_sha(item,expected_sha):
    official=(item.get('lfs') or {}).get('sha256')
    require(not official or expected_sha is None or expected_sha==official,'Prior/transport SHA differs from trusted official LFS SHA')
    return official or expected_sha or ''


class OfficialBackend:
    def __init__(self,staging):self.staging=Path(staging)
    def download(self,name,item):
        path=self.staging/name;require(not path.is_symlink(),'Official staging source symlink refused')
        actual=path.stat().st_size if path.exists() else 0
        require(actual<=item['size'],'Oversized staging partial; preserve for review')
        require(shutil.disk_usage(self.staging).free>=item['size']-actual+RESERVE,'Insufficient source free space; preserve staging')
        url=f'https://huggingface.co/Qwen/Qwen2.5-32B/resolve/{transfer.REVISION}/{name}'
        for attempt in range(32):
            if path.exists() and path.stat().st_size==item['size']:break
            result=subprocess.run(['curl','--socks5-hostname','127.0.0.1:7892','--http1.1','--silent','--show-error','--location','--fail','--connect-timeout','15','--retry','0','--speed-time','60','--speed-limit','8192','--continue-at','-','--output',str(path),url])
            if result.returncode==0:break
            print(json.dumps(dict(status='OFFICIAL_DOWNLOAD_RESUME',name=name,attempt=attempt,current_bytes=path.stat().st_size if path.exists() else 0)),flush=True);time.sleep(3)
        else:raise RuntimeError('Official download attempts exhausted; partial preserved')
        return source_proof(path,item)
    def relay(self,name,item,proof):
        # Small nonLFS sources were verified against official Git blob above;
        # transport always uses their actual SHA256, not a fictitious LFS label.
        transport_item=dict(item,lfs=dict(sha256=proof['sha256']))
        return transfer.relay(name,transport_item)
    def verify_target(self,name,item,expected_sha=None):
        script="""import hashlib,json,os,socket,sys
from pathlib import Path
name,size,expected_sha,git_expected,root=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],Path(sys.argv[5])
assert socket.gethostname().split('.')[0]=='h100-11'
p=root/name
assert p.is_file() and not p.is_symlink() and p.stat().st_uid==os.getuid() and p.stat().st_size==size
h=hashlib.sha256();g=hashlib.sha1();g.update(b'blob '+str(size).encode()+b'\\0')
with p.open('rb') as f:
 for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw);g.update(raw)
assert not expected_sha or h.hexdigest()==expected_sha
assert not git_expected or g.hexdigest()==git_expected
print(json.dumps(dict(size=size,sha256=h.hexdigest(),official_git_blob_sha1=g.hexdigest() if git_expected else None,remote_host='h100-11',remote_directory=str(root),status='TARGET_FULL_SHA_VERIFIED')))
"""
        sha=validate_expected_sha(item,expected_sha)
        git='' if item.get('lfs') else item['blobId']
        command='/dataset/shaoshitong/search-r1-rl/topd/env/bin/python -c '+shlex.quote(script)+' '+ ' '.join(shlex.quote(str(x)) for x in (name,item['size'],sha,git,transfer.REMOTE))
        result=subprocess.run(transfer.route(command),capture_output=True,text=True,timeout=300)
        require(result.returncode==0,'Target missing/SHA mismatch; preserve local source')
        return json.loads(result.stdout)
    def free_bytes(self):return shutil.disk_usage(self.staging).free
    def publish_manifest(self,manifest):
        raw=(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();sha=hashlib.sha256(raw).hexdigest()
        script="""import hashlib,os,socket,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-11'
root=Path(sys.argv[1]);expected=sys.argv[2];raw=sys.stdin.buffer.read(2*1024**2+1)
assert len(raw)<=2*1024**2 and hashlib.sha256(raw).hexdigest()==expected
p=root/'official-download-manifest-v6.json';t=root/'official-download-manifest-v6.json.publish-v6.transfer'
assert not p.is_symlink() and not t.is_symlink()
if p.exists():
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
else:
 if t.exists():
  assert t.is_file() and t.stat().st_uid==os.getuid()
  prefix=t.read_bytes();assert len(prefix)<=len(raw) and raw.startswith(prefix)
  fd=os.open(t,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
  tail=raw[len(prefix):]
 else:
  fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);tail=raw
 with os.fdopen(fd,'ab') as f:f.write(tail);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(t.read_bytes()).hexdigest()==expected
 os.replace(t,p)
 fd=os.open(root,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
print(expected)
"""
        command='/dataset/shaoshitong/search-r1-rl/topd/env/bin/python -c '+shlex.quote(script)+' '+shlex.quote(transfer.REMOTE)+' '+shlex.quote(sha)
        result=subprocess.run(transfer.route(command),input=raw,capture_output=True,timeout=180)
        require(result.returncode==0 and result.stdout.decode().strip()==sha,'Manifest publication unverified; preserve local state')
        return sha


class Pipeline:
    def __init__(self,entries,names,staging,store,backend,max_slots=3,reserve=RESERVE):
        require(0<max_slots<=3,'Three-slot ceiling')
        self.entries=entries;self.names=names;self.staging=Path(staging);self.store=store;self.backend=backend;self.max_slots=max_slots;self.reserve=reserve
        self.active={};self.slot_lock=threading.RLock();self.failed=threading.Event()
    def remaining(self,name):
        p=self.staging/name;actual=p.stat().st_size if p.exists() else 0
        require(actual<=self.entries[name]['size'],'Oversized own partial')
        return self.entries[name]['size']-actual
    def admit(self,name):
        with self.slot_lock:
            require(not self.failed.is_set() and len(self.active)<self.max_slots,'No new source after failure; three-slot ceiling')
            needed=sum(self.remaining(n) for n in self.active)+self.remaining(name)+self.reserve
            require(self.backend.free_bytes()>=needed,'Three-slot remaining sizes plus2GiB reserve unavailable')
            self.active[name]='ADMITTED'
            with self.store.lock:
                self.store.value['inflight_files']=list(self.active);self.store.save()
    def phase(self,name,value):
        with self.slot_lock:
            self.active[name]=value
            with self.store.lock:self.store.value['phases']=dict(self.active);self.store.save()
    def unlink_verified(self,name,proof):
        path=self.staging/name
        if path.exists():
            require(not path.is_symlink() and path.stat().st_ino==proof['source_inode'] and path.stat().st_dev==proof['source_device'] and path.stat().st_mtime_ns==proof['source_mtime_ns'] and path.stat().st_size==proof['size'],'Source identity changed before approved unlink; preserve')
            path.unlink()
    def one(self,name):
        try:
            self.phase(name,'DOWNLOADING');proof=self.backend.download(name,self.entries[name])
            self.phase(name,'SOURCE_OFFICIAL_VERIFIED');transport=self.backend.relay(name,self.entries[name],proof)
            self.phase(name,'TARGET_REVERIFYING');target=self.backend.verify_target(name,self.entries[name],proof['sha256'])
            require(target['status']=='TARGET_FULL_SHA_VERIFIED' and target['size']==proof['size'] and target['sha256']==proof['sha256'],'Target fullSHA differs')
            record=dict(proof,remote_sha_verified=True,target=target,transport=transport,verified_epoch=time.time())
            self.store.commit(name,record) # Must succeed durably before unlink.
            self.phase(name,'STATE_COMMITTED');self.unlink_verified(name,proof);self.phase(name,'RELAY_LOCAL_REMOVED')
            return name
        except BaseException as error:
            with self.slot_lock:
                self.failed.set()
                with self.store.lock:
                    self.store.value['last_error']=type(error).__name__+': '+str(error);self.store.value['failed_file']=name;self.store.save()
            raise
    def run(self):
        require(len(self.names)==len(set(self.names)),'Duplicate source names')
        require(not any(p.name.endswith('.safetensors') and p.name not in self.entries for p in self.staging.iterdir()),'Unknown weight source in staging; preserve window')
        existing={p.name for p in self.staging.iterdir() if p.is_file() and p.name in self.entries}
        require(len(existing)<=3 and existing<=set(self.names),'Existing staging window has more than3/untracked files; preserve for review')
        queue=sorted(self.names,key=lambda n:(n not in existing,self.names.index(n)))
        # The caller filters/adopts verified targets before admission. Futures
        # are submitted only when their entire source→target→cleanup slot frees.
        first_error=None
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_slots) as pool:
            pending={}
            def fill():
                nonlocal first_error
                with self.slot_lock:
                    while queue and len(pending)<self.max_slots and not self.failed.is_set():
                        name=queue[0]
                        try:self.admit(name)
                        except BaseException as error:
                            self.failed.set();first_error=first_error or error
                            with self.store.lock:
                                self.store.value['last_error']=type(error).__name__+': '+str(error);self.store.value['failed_file']=name;self.store.save()
                            break
                        queue.pop(0);pending[pool.submit(self.one,name)]=name
            fill()
            while pending:
                completed,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in completed:
                    name=pending.pop(future)
                    try:future.result()
                    except BaseException as error:
                        if first_error is None:first_error=error
                    with self.slot_lock:
                        self.active.pop(name,None)
                        with self.store.lock:self.store.value['inflight_files']=list(self.active);self.store.value['phases']=dict(self.active);self.store.save()
                if first_error is None:fill()
        if first_error is not None:raise first_error
        require(not queue,'Some official files not completed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exclusive-handoff',required=True,type=Path);parser.add_argument('--ssh-compress',action='store_true');parser.add_argument('--slots',type=int,choices=[1,2,3],default=3);args=parser.parse_args()
    handoff=transfer.exclusive_handoff(args.exclusive_handoff)
    metadata=transfer.official_metadata()
    entries={x['rfilename']:x for x in metadata['siblings']};shards=sorted(n for n in entries if n.endswith('.safetensors'))
    require(len(shards)==17 and all(entries[n].get('lfs',{}).get('sha256') for n in shards),'All17 official LFS shard hashes required')
    names=list(SMALL)+shards;require(all('/' not in n and n in entries for n in names),'Pinned source names only')
    staging=transfer.STAGING;require(staging.is_dir() and not staging.is_symlink(),'Existing own staging directory required')
    transfer.ADOPT_LEGACY=True;transfer.SSH_COMPRESS=args.ssh_compress
    with (staging/'download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB) # Single parent lock; relay() never re-locks it.
        initial=json.loads(STATE.read_text()) if STATE.exists() else dict(scope=SCOPE,model='Qwen/Qwen2.5-32B',revision=transfer.REVISION,remote_directory=transfer.REMOTE,files={},complete=False)
        require(initial['scope']==SCOPE and initial['revision']==transfer.REVISION and initial['remote_directory']==transfer.REMOTE,'Foreign existing v6 state')
        initial.update(pid=os.getpid(),slots=args.slots,complete=False,exclusive_handoff_sha256=transfer.digest_file(args.exclusive_handoff),inflight_files=[],phases={})
        initial.pop('last_error',None);initial.pop('failed_file',None)
        store=StateStore(STATE,initial);backend=OfficialBackend(staging);pending=[]
        legacy_path=transfer.BASE/'outputs/qwen25-32b-download-state.json'
        legacy=json.loads(legacy_path.read_text()) if legacy_path.exists() else {'files':{}}
        if legacy_path.exists():require(legacy['revision']==transfer.REVISION and legacy['remote_directory']==transfer.REMOTE,'Foreign v5 state')
        for name in names:
            prior=initial['files'].get(name) or legacy['files'].get(name)
            if prior and prior.get('remote_sha_verified'):
                # Never inherit a Boolean without actual current target reading.
                validate_expected_sha(entries[name],prior['sha256'])
                target=backend.verify_target(name,entries[name],prior['sha256'])
                record=dict(size=entries[name]['size'],sha256=target['sha256'],official_lfs_sha256=(entries[name].get('lfs') or {}).get('sha256'),remote_sha_verified=True,target=target,adopted_verified_target=True)
                store.commit(name,record)
                path=staging/name
                if path.exists():proof=source_proof(path,entries[name]);require(proof['sha256']==target['sha256'],'Local/target identity differs');Pipeline(entries,[],staging,store,backend).unlink_verified(name,proof)
            else:pending.append(name)
        pipeline=Pipeline(entries,pending,staging,store,backend,max_slots=args.slots);pipeline.run()
        require(all(initial['files'].get(n,{}).get('remote_sha_verified') for n in names),'Not all24 official files verified')
        manifest=dict(scope=SCOPE,model='Qwen/Qwen2.5-32B',revision=transfer.REVISION,remote_directory=transfer.REMOTE,
            files={n:{k:initial['files'][n].get(k) for k in ('size','sha256','official_lfs_sha256')} for n in names})
        manifest_sha=backend.publish_manifest(manifest)
        with store.lock:initial.update(complete=True,inflight_files=[],phases={},remote_manifest_sha256=manifest_sha);store.save()
        print(json.dumps(dict(status='OFFICIAL_ALL17_SHARDS_PLUS7_METADATA_VERIFIED_NOT_EVALUATED',revision=transfer.REVISION,files=len(names),remote_manifest_sha256=manifest_sha)),flush=True)


if __name__=='__main__':main()
