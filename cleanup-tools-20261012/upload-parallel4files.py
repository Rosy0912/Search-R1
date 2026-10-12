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
import requests

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
    args += ['-o','ProxyCommand=none','-o','ProxyJump=none']
    keys = cfg.get('trusted_target_host_keys',{}).get('h100-8')
    if keys:
        args += ['-o', 'KnownHostsCommand=' + shlex.join(['/bin/echo', '\n'.join(keys)])]
    return args + [cfg['user'] + '@' + cfg['targets']['h100-8'], command]


class RemoteFile(io.BufferedIOBase):
    def __init__(self, record):
        self.record, self.position, self.process = record, 0, None
        self.last_progress=0

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
            if remaining > 8 * 1048576:
                raise ValueError('Unbounded source read is forbidden')
            size = remaining
        size = min(size, remaining)
        if not self.process:
            code = ('import sys,pathlib; p=pathlib.Path(%r); assert p.stat().st_size==%d; '
                    'f=p.open("rb"); f.seek(%d)\n'
                    'for b in iter(lambda:f.read(1048576),b""):sys.stdout.buffer.write(b)\n'
                    'sys.stdout.buffer.flush()\n') % (self.record['archive'], self.record['archive_bytes'], self.position)
            self.process = subprocess.Popen(source_args('python3 -c ' + shlex.quote(code)),
                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = self.process.stdout.read(size)
        if len(data) != size:
            raise IOError('Short source stream')
        self.position += len(data)
        if time.time()-self.last_progress >= 2 or self.position == self.record['archive_bytes']:
            (ROOT / 'progress.json').write_text(json.dumps({'epoch':time.time(),'file':self.record['repo_path'],'position':self.position,'total':self.record['archive_bytes']}))
            self.last_progress=time.time()
        return data

    def close(self):
        self.stop_stream()
        super().close()


import concurrent.futures,threading,re
SOURCE_OPEN_GATE=threading.Semaphore(2)
class ParallelRemoteFile(io.BufferedIOBase):
    BLOCK=256*1024
    LANES=4
    def __init__(self,record):
        self.record=record; self.position=0; self.processes=[]; self.futures={}; self.executor=None; self.last_progress=0;self.max_read_requested=0;self.source_opens=0;self.highwater=0
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
            if self.record['archive_bytes']>10*1024*1024:raise ValueError('Unbounded read allowed only for bounded small metadata')
            size=remaining
        if size>64*1024*1024:raise ValueError('Read exceeds 64MiB response bound')
        size=min(size,remaining);parts=[]
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
        self.highwater=max(self.highwater,self.position)
        if time.time()-self.last_progress>=2 or self.position==self.record['archive_bytes']:
            p=ROOT/('progress-'+hashlib.sha256(self.record['repo_path'].encode()).hexdigest()[:16]+'.json'); tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'epoch':time.time(),'file':self.record['repo_path'],'position':self.position,'total':self.record['archive_bytes'],'lanes':self.LANES,'bounded_cache_bytes':self.BLOCK*self.LANES,'max_read_requested':self.max_read_requested,'source_opens':self.source_opens,'read_highwater':self.highwater}));tmp.replace(p)
            self.last_progress=time.time()
        return b''.join(parts)
    def close(self):
        self.stop_stream();super().close()

def verify_file(api,plan,record,token,revision):
    info=api.get_paths_info(repo_id=plan['repo'],paths=[record['repo_path']],revision=revision)[0]
    assert info.size==record['archive_bytes']
    if info.lfs:assert info.lfs.sha256==record['archive_sha256']
    else:
        assert record['archive_bytes']<10*1024*1024
        url='https://huggingface.co/'+plan['repo']+'/resolve/'+revision+'/'+record['repo_path']
        resp=requests.get(url,headers={'Authorization':'Bearer '+token},timeout=90);resp.raise_for_status()
        assert hashlib.sha256(resp.content).hexdigest()==record['archive_sha256']

def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor,as_completed
    lock=(ROOT/'uploader.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=sys.stdin.readline().strip();assert token
    plan=json.loads(PLAN.read_text());api=HfApi(token=token);assert api.model_info(plan['repo']).private
    total=sum(r['archive_bytes'] for r in plan['records']);guard=threading.Lock();receipts={}
    def write(path,value):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
    write(ROOT/'controller.json',{'pid':os.getpid(),'epoch':time.time(),'repo':plan['repo'],'model_disk_staging':False,'file_concurrency':4,'source_lanes_per_file':4,'cache_per_file_bytes':ParallelRemoteFile.BLOCK*ParallelRemoteFile.LANES,'total_bytes':total})
    def mark(record,result):
        with guard:
            receipts[record['repo_path']]=result
            write(ROOT/'verified-progress.json',{'epoch':time.time(),'verified_bytes':sum(r['archive_bytes'] for r in receipts.values()),'files':len(receipts),'total_bytes':total})
    def worker(record):
        rid=hashlib.sha256(record['repo_path'].encode()).hexdigest()[:16];fl=(ROOT/(rid+'.file.lock')).open('w');fcntl.flock(fl,fcntl.LOCK_EX|fcntl.LOCK_NB)
        api=HfApi(token=token);receipt=ROOT/(rid+'-hf-uploaded.json')
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['archive_sha256']==record['archive_sha256'];assert old['archive_bytes']==record['archive_bytes'];verify_file(api,plan,record,token,old['commit']);mark(record,old);return
        for attempt in range(4):
            stream=ParallelRemoteFile(record);started=time.time()
            try:
                sample=stream.read(min(512,record['archive_bytes']));stream.seek(0)
                op=CommitOperationAdd(path_in_repo=record['repo_path'],path_or_fileobj=sample);op.path_or_fileobj=stream;op.upload_info=UploadInfo(size=record['archive_bytes'],sha256=bytes.fromhex(record['archive_sha256']),sample=sample)
                commit=api.create_commit(repo_id=plan['repo'],operations=[op],commit_message='H100 GiGPO selected best checkpoint backup: '+record['repo_path'])
                verify_file(api,plan,record,token,commit.oid)
                result={'status':'HF_UPLOAD_VERIFIED','group':record['group'],'archive_bytes':record['archive_bytes'],'archive_sha256':record['archive_sha256'],'repo_path':record['repo_path'],'commit':commit.oid,'epoch':time.time(),'elapsed_seconds':time.time()-started,'model_disk_staging':False}
                write(receipt,result);mark(record,result);print(json.dumps(result),flush=True);return
            except Exception as e:
                chain=[];err=e;seen=set()
                import traceback,re
                while err is not None and id(err) not in seen and len(chain)<8:
                    seen.add(id(err));msg=re.sub(r'https?://\S+','[redacted-url]',str(err));msg=re.sub(r'hf_[A-Za-z0-9]+','[redacted-token]',msg);msg=re.sub(r'Bearer\s+\S+','Bearer [redacted]',msg)
                    chain.append({'type':type(err).__name__,'message':msg[:4000],'frames':[{'file':pathlib.Path(t.filename).name,'line':t.lineno,'function':t.name} for t in traceback.extract_tb(err.__traceback__)]});err=err.__cause__ or err.__context__
                write(ROOT/(rid+'-error.json'),{'file':record['repo_path'],'type':type(e).__name__,'count':attempt+1,'epoch':time.time(),'read_position':stream.position,'max_read_requested':stream.max_read_requested,'source_opens':stream.source_opens,'causes':chain})
                if attempt==3:raise RuntimeError(type(e).__name__) from None
                time.sleep(10*(attempt+1))
            finally:
                stream.close()

    phase1=[r for r in plan['records'] if r['group'] in ['gigpo-best512-algorithm25','gigpo-best512-step25'] and '/training_state/' not in r['repo_path']]
    phase2=[r for r in plan['records'] if r not in phase1]
    for phase,records in enumerate([phase1,phase2],1):
        records=sorted(records,key=lambda r:(r['group']=='searchg1-best512-step5','/training_state/' in r['repo_path'],-r['archive_bytes']))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(worker,r) for r in records]):f.result()
        if phase==1 and phase1:
            write(ROOT/'best-inference-complete.json',{'status':'BEST_INFERENCE_ALL_COMPONENTS_HF_SHA_VERIFIED','epoch':time.time(),'files':len(phase1),'bytes':sum(r['archive_bytes'] for r in phase1),'records':phase1})
    final=api.model_info(plan['repo']).sha
    for record in plan['records']:verify_file(api,plan,record,token,final)
    write(ROOT/'upload-complete.json',{'status':'ALL_SELECTED_HF_FILES_SHA_VERIFIED','epoch':time.time(),'files':len(receipts),'bytes':total,'revision':final})
if __name__=='__main__':main()
