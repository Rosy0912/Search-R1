"""H100 -> management RAM stream -> authorized private HF repository.

Read the token once from stdin; never persist it or stage model bytes here.
Existing destination transfers are untouched. HF receipts are upload receipts,
not evidence of completed HPC3 migration.
"""
import fcntl
import hashlib
import io
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.lfs import UploadInfo

ROOT = pathlib.Path(__file__).resolve().parent
PLAN = ROOT / 'plan.json'
CFG = pathlib.Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json')


def source_args(command):
    cfg = json.loads(CFG.read_text())
    prefix = ['env', '-u', 'LD_LIBRARY_PATH', 'ssh', '-T', '-i', cfg['management_identity_file'],
              '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
              '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=8', '-o', 'ConnectionAttempts=1',
              '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
    args = prefix.copy()
    args += ['-o', 'ProxyCommand=none', '-o', 'ProxyJump=none']
    keys = cfg.get('trusted_target_host_keys', {}).get('h100-10')
    if keys: args += ['-o', 'KnownHostsCommand=' + shlex.join(['/bin/echo', '\n'.join(keys)])]
    return args + [cfg['user'] + '@' + cfg['targets']['h100-10'], command]


def spawn_source(*args,**kwargs):
    lock=(ROOT.parent/'h100-cleanup-20261012-node10-source-start.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        process=subprocess.Popen(*args,**kwargs);time.sleep(.7);return process
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

class RemoteFile(io.BufferedIOBase):
    def __init__(self, record):
        self.record, self.position, self.process = record, 0, None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def stop_stream(self):
        if self.process:
            self.process.stdout.close()
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position + offset if whence == 1 else self.record['archive_bytes'] + offset)
        if position < 0 or position > self.record['archive_bytes']:
            raise ValueError('Invalid source offset')
        if position != self.position:
            self.stop_stream()
            self.position = position
        return position

    def read(self, size=-1):
        remaining = self.record['archive_bytes'] - self.position
        if not remaining:
            return b''
        # HTTP/SliceFileObj must use bounded reads, never materialize a shard.
        if size < 0:
            if remaining > 16 * 1048576:
                raise ValueError('Unbounded source read is forbidden')
            size = remaining
        size = min(size, remaining)
        if not self.process:
            code = ('import sys,pathlib; p=pathlib.Path(%r); assert p.stat().st_size==%d; '
                    'f=p.open("rb"); f.seek(%d)\n'
                    'for b in iter(lambda:f.read(1048576),b""):sys.stdout.buffer.write(b)\n'
                    'sys.stdout.buffer.flush()\n') % (self.record['archive'], self.record['archive_bytes'], self.position)
            self.process = spawn_source(source_args('python3 -c ' + shlex.quote(code)),
                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = self.process.stdout.read(size)
        if len(data) != size:
            raise IOError('Short source stream')
        self.position += len(data)
        if self.position//(64*1048576) != (self.position-len(data))//(64*1048576):
            progress_write(self.record,{'path':self.record['repo_path'],'source_read_bytes':self.position,'epoch':time.time()})
        return data

    def close(self):
        self.stop_stream()
        super().close()


import concurrent.futures,threading,re
SOURCE_OPEN_GATE=threading.Semaphore(2)
class ParallelRemoteFile(io.BufferedIOBase):
    BLOCK=8*1024*1024
    LANES=4
    def __init__(self,record):
        self.record=record; self.position=0; self.processes=[]; self.futures={}; self.executor=None; self.last_progress=0;self.max_read_requested=0;self.source_opens=0;self.high_water=0;self.total_read_bytes=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def stop_stream(self):
        for p in self.processes:
            if p.poll() is None:p.terminate()
        for p in self.processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            p.stdout.close()
            if p.stderr:p.stderr.close()
        if self.executor:self.executor.shutdown(wait=True,cancel_futures=True)
        self.processes=[];self.futures={};self.executor=None
    def seek(self,offset,whence=0):
        new=offset if whence==0 else (self.position+offset if whence==1 else self.record['archive_bytes']+offset)
        if not 0<=new<=self.record['archive_bytes']:raise ValueError('Invalid source offset')
        key=new//self.BLOCK
        # Defer restart until a real read; HF SliceFileObj restores old offsets between parts.
        self.position=new;return new
    def _start(self):
        base=self.position//self.BLOCK
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=self.LANES)
        for lane in range(self.LANES):
            key=base+lane;offset=key*self.BLOCK
            if offset>=self.record['archive_bytes']:break
            code="import sys,pathlib; p=pathlib.Path(%r); assert p.stat().st_size==%d; f=p.open('rb'); off=%d\nwhile off<%d:\n f.seek(off); data=f.read(min(%d,%d-off)); sys.stdout.buffer.write(data); sys.stdout.buffer.flush(); off+=%d\n"%(self.record['archive'],self.record['archive_bytes'],offset,self.record['archive_bytes'],self.BLOCK,self.record['archive_bytes'],self.BLOCK*self.LANES)
            expected=min(self.BLOCK,self.record['archive_bytes']-offset)
            for attempt in range(3):
                with SOURCE_OPEN_GATE:
                    p=subprocess.Popen(source_args('python3 -c '+shlex.quote(code)),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    self.source_opens+=1
                    prefix=p.stdout.read(min(512,expected))
                    if len(prefix)==min(512,expected):break
                    if p.poll() is None:p.terminate()
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    err=p.stderr.read(2048).decode(errors='replace')
                    err=re.sub(r'https?://\S+','[redacted-url]',err)
                    err=re.sub(r'hf_[A-Za-z0-9]+','[redacted-token]',err)
                    with (ROOT/'source-connection-errors.jsonl').open('a') as log:
                        log.write(json.dumps({'epoch':time.time(),'file':self.record['repo_path'],'lane':lane,'attempt':attempt+1,'returncode':p.returncode,'stderr':err})+'\n')
                    p.stdout.close();p.stderr.close()
                if attempt==2:raise OSError('Source SSH startup failed after bounded retries')
                time.sleep(1)
            self.processes.append(p)
            def read_first(proc=p,head=prefix,n=expected):return head+proc.stdout.read(n-len(head))
            self.futures[key]=(p,self.executor.submit(read_first))
    def read(self,size=-1):
        remaining=self.record['archive_bytes']-self.position
        self.max_read_requested=max(self.max_read_requested,size)
        if not remaining:return b''
        if size<0:
            if remaining>self.BLOCK:raise ValueError('Unbounded source read forbidden')
            size=remaining
        if size>64*1024*1024:raise ValueError('Read exceeds 64MiB response bound')
        size=min(size,remaining);requested=size;parts=[]
        if self.executor and self.position//self.BLOCK not in self.futures:self.stop_stream()
        if not self.executor:self._start()
        while size:
            key=self.position//self.BLOCK;off=self.position%self.BLOCK
            p,future=self.futures[key];data=future.result();expected=min(self.BLOCK,self.record['archive_bytes']-key*self.BLOCK)
            if len(data)!=expected:raise IOError('Short parallel source stream')
            n=min(size,len(data)-off);parts.append(data[off:off+n]);self.position+=n;size-=n
            if off+n==len(data):
                del self.futures[key];nextkey=key+self.LANES;nextoffset=nextkey*self.BLOCK
                if nextoffset<self.record['archive_bytes']:self.futures[nextkey]=(p,self.executor.submit(p.stdout.read,min(self.BLOCK,self.record['archive_bytes']-nextoffset)))
        self.high_water=max(self.high_water,self.position);self.total_read_bytes+=requested
        if time.time()-self.last_progress>=2 or self.position==self.record['archive_bytes']:
            p=ROOT/('progress-'+hashlib.sha256(self.record['repo_path'].encode()).hexdigest()[:16]+'.json'); tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'epoch':time.time(),'file':self.record['repo_path'],'read_position':self.position,'size':self.record['archive_bytes'],'lanes':self.LANES,'bounded_cache_bytes':self.BLOCK*self.LANES,'max_read_requested':self.max_read_requested,'source_opens':self.source_opens,'high_water':self.high_water,'total_read_bytes':self.total_read_bytes,'attempt':self.record.get('_attempt',0)}));tmp.replace(p)
            self.last_progress=time.time()
        return b''.join(parts)
    def close(self):
        self.stop_stream();super().close()


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
def progress_write(record,value):
    write_json(ROOT/('progress-'+hashlib.sha256(record['repo_path'].encode()).hexdigest()[:16]+'.json'),value)
def verify(api,plan,record,token,revision):
    info=api.get_paths_info(plan['repo'],[record['repo_path']],revision=revision)[0];assert info.size==record['archive_bytes']
    if info.lfs:assert info.lfs.sha256==record['archive_sha256']
    else:
        import requests
        assert record['archive_bytes']<=16*1048576
        resp=requests.get('https://huggingface.co/'+plan['repo']+'/resolve/'+revision+'/'+record['repo_path'],headers={'Authorization':'Bearer '+token},timeout=60);resp.raise_for_status();assert hashlib.sha256(resp.content).hexdigest()==record['archive_sha256']
def main():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import threading
    lock=(ROOT/'uploader.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=sys.stdin.readline().strip();assert token
    plan=json.loads(PLAN.read_text());api=HfApi(token=token);assert api.model_info(plan['repo']).private
    total=sum(r['archive_bytes'] for r in plan['records']);guard=threading.Lock();receipts={}
    write_json(ROOT/'controller.json',{'pid':os.getpid(),'epoch':time.time(),'generation':'parallel4files4lanes-gated-lazyseek-v3','concurrency':4,'cache_bytes_per_large_file':32*1048576,'max_read_bytes':64*1048576,'total_bytes':total,'route':'H100 -> MGMT RAM -> HF'})
    def mark(n,item):
        with guard:
            receipts[n]=item;write_json(ROOT/'verified-progress.json',{'epoch':time.time(),'verified_bytes':sum(r['archive_bytes'] for r in receipts.values()),'files':len(receipts),'total_bytes':total})
    def worker(n,r):
        lk=(ROOT/('file-%04d.lock'%n)).open('w');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a=HfApi(token=token);receipt=ROOT/('receipt-%04d.json'%n)
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['archive_sha256']==r['archive_sha256'];verify(a,plan,r,token,'main');mark(n,old);return
        for attempt in range(3):
            stream=ParallelRemoteFile(dict(r,_attempt=attempt)) if r['archive_bytes']>16*1048576 else RemoteFile(r)
            try:
                sample=stream.read(min(512,r['archive_bytes']));stream.seek(0);op=CommitOperationAdd(path_in_repo=r['repo_path'],path_or_fileobj=sample);op.path_or_fileobj=stream;op.upload_info=UploadInfo(size=r['archive_bytes'],sha256=bytes.fromhex(r['archive_sha256']),sample=sample)
                commit=a.create_commit(plan['repo'],operations=[op],commit_message='H100 cleanup bounded RAM backup '+r['group']);verify(a,plan,r,token,commit.oid)
                item=dict(r,status='HF_OFFICIAL_SHA_VERIFIED',commit=commit.oid,epoch=time.time());write_json(receipt,item);mark(n,item);print(json.dumps({'status':item['status'],'path':r['repo_path'],'bytes':r['archive_bytes']}),flush=True);return
            except Exception as e:
                import re,traceback
                chain=[];cause=e
                while cause and len(chain)<6:
                    chain.append({'type':type(cause).__name__,'message':re.sub(r'https?://\S+|hf_[A-Za-z0-9]+|Bearer\s+\S+','[redacted]',str(cause))[:400],'frames':[{'file':pathlib.Path(f.filename).name,'line':f.lineno,'func':f.name} for f in traceback.extract_tb(cause.__traceback__)]});cause=cause.__cause__
                diagnostic={'chain':chain,'message':re.sub(r'https?://\S+|hf_[A-Za-z0-9]+|Bearer\s+\S+','[redacted]',str(e))[:300],'frames':[{'file':pathlib.Path(f.filename).name,'line':f.lineno,'func':f.name} for f in traceback.extract_tb(e.__traceback__)]}
                write_json(ROOT/('file-%04d-error.json'%n),{'epoch':time.time(),'type':type(e).__name__,'attempt':attempt,'path':r['repo_path'],'diagnostic':diagnostic})
                print(json.dumps({'status':'UPLOAD_ERROR','type':type(e).__name__,'attempt':attempt,'path':r['repo_path'],'diagnostic':diagnostic}),flush=True)
                if attempt==2:raise RuntimeError('Bounded retries exhausted: '+type(e).__name__) from None
                time.sleep(10)
            finally:stream.close()
    records=list(enumerate(plan['records']));inference=[p for p in records if '/training_state/' not in p[1]['archive']];native=[p for p in records if '/training_state/' in p[1]['archive']]
    for label,phase in [('inference',inference),('native',native)]:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(worker,n,r) for n,r in sorted(phase,key=lambda p:(not p[1]['archive'].endswith('.safetensors'),-p[1]['archive_bytes']))]
            for f in as_completed(futures):f.result()
        if phase:
            rev=api.model_info(plan['repo']).sha
            for n,r in phase:verify(api,plan,r,token,rev)
            write_json(ROOT/(label+'-complete.json'),{'status':'PHASE_HF_FRESH_SHA_VERIFIED','phase':label,'epoch':time.time(),'revision':rev,'files':len(phase),'bytes':sum(r['archive_bytes'] for n,r in phase)})
    rev=api.model_info(plan['repo']).sha
    for r in plan['records']:verify(api,plan,r,token,rev)
    write_json(ROOT/'complete.json',{'status':'ALL_FILES_HF_OFFICIAL_SHA_VERIFIED','records':list(receipts.values()),'epoch':time.time(),'revision':rev})
if __name__=='__main__':main()
