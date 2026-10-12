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


import concurrent.futures,threading
class ParallelRemoteFile(io.BufferedIOBase):
    BLOCK=8*1024*1024
    LANES=4
    def __init__(self,record):
        self.record=record; self.position=0; self.processes=[]; self.futures={}; self.executor=None; self.last_progress=0;self.max_read_requested=0
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
        if self.executor:self.executor.shutdown(wait=True,cancel_futures=True)
        self.processes=[];self.futures={};self.executor=None
    def seek(self,offset,whence=0):
        new=offset if whence==0 else (self.position+offset if whence==1 else self.record['archive_bytes']+offset)
        if not 0<=new<=self.record['archive_bytes']:raise ValueError('Invalid source offset')
        key=new//self.BLOCK
        if self.futures and key not in self.futures and new!=self.record['archive_bytes']:self.stop_stream()
        self.position=new;return new
    def _start(self):
        base=self.position//self.BLOCK
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=self.LANES)
        for lane in range(self.LANES):
            key=base+lane;offset=key*self.BLOCK
            if offset>=self.record['archive_bytes']:break
            code="import sys,pathlib; p=pathlib.Path(%r); assert p.stat().st_size==%d; f=p.open('rb'); off=%d\nwhile off<%d:\n f.seek(off); data=f.read(min(%d,%d-off)); sys.stdout.buffer.write(data); sys.stdout.buffer.flush(); off+=%d\n"%(self.record['archive'],self.record['archive_bytes'],offset,self.record['archive_bytes'],self.BLOCK,self.record['archive_bytes'],self.BLOCK*self.LANES)
            p=subprocess.Popen(source_args('python3 -c '+shlex.quote(code)),stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            self.processes.append(p);self.futures[key]=(p,self.executor.submit(p.stdout.read,min(self.BLOCK,self.record['archive_bytes']-offset)))
    def read(self,size=-1):
        remaining=self.record['archive_bytes']-self.position
        self.max_read_requested=max(self.max_read_requested,size)
        if not remaining:return b''
        if size<0:
            if remaining>self.BLOCK:raise ValueError('Unbounded source read forbidden')
            size=remaining
        if size>64*1024*1024:raise ValueError('Read exceeds 64MiB response bound')
        size=min(size,remaining);parts=[]
        if not self.executor:self._start()
        while size:
            key=self.position//self.BLOCK;off=self.position%self.BLOCK
            p,future=self.futures[key];data=future.result();expected=min(self.BLOCK,self.record['archive_bytes']-key*self.BLOCK)
            if len(data)!=expected:raise IOError('Short parallel source stream')
            n=min(size,len(data)-off);parts.append(data[off:off+n]);self.position+=n;size-=n
            if off+n==len(data):
                del self.futures[key];nextkey=key+self.LANES;nextoffset=nextkey*self.BLOCK
                if nextoffset<self.record['archive_bytes']:self.futures[nextkey]=(p,self.executor.submit(p.stdout.read,min(self.BLOCK,self.record['archive_bytes']-nextoffset)))
        if time.time()-self.last_progress>=2 or self.position==self.record['archive_bytes']:
            (ROOT/'progress.json').write_text(json.dumps({'epoch':time.time(),'file':self.record['repo_path'],'position':self.position,'total':self.record['archive_bytes'],'lanes':self.LANES,'bounded_cache_bytes':self.BLOCK*self.LANES,'max_read_requested':self.max_read_requested}))
            self.last_progress=time.time()
        return b''.join(parts)
    def close(self):
        self.stop_stream();super().close()

def main():
    lock = (ROOT / 'uploader.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    token = sys.stdin.readline().strip()
    assert token
    plan = json.loads(PLAN.read_text())
    api = HfApi(token=token)
    assert api.model_info(plan['repo']).private, 'Private repository required'
    (ROOT / 'controller.json').write_text(json.dumps({'pid': os.getpid(), 'epoch': time.time(), 'repo': plan['repo'], 'model_disk_staging': False}))
    records = sorted(plan['records'], key=lambda r: ({'gigpo-best512-algorithm25':0,'gigpo-best512-step25':1,'searchg1-best512-step5':2}.get(r['group'],0), '/training_state/' in r['repo_path'], r['repo_path']))
    for record in records:
        receipt = ROOT / (hashlib.sha256(record['repo_path'].encode()).hexdigest()[:16] + '-hf-uploaded.json')
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old.get('archive_sha256') == record['archive_sha256']:
                continue
        failures = 0
        while failures < 4:
            stream = ParallelRemoteFile(record)
            started = time.time()
            try:
                sample = stream.read(min(512, record['archive_bytes']))
                stream.seek(0)
                op = CommitOperationAdd(path_in_repo=record['repo_path'], path_or_fileobj=sample)
                op.path_or_fileobj = stream
                op.upload_info = UploadInfo(size=record['archive_bytes'], sha256=bytes.fromhex(record['archive_sha256']), sample=sample)
                commit = api.create_commit(repo_id=plan['repo'], operations=[op], commit_message='H100 disk cleanup backup: ' + record['group'])
                info=list(api.get_paths_info(repo_id=plan['repo'],paths=[op.path_in_repo],revision=commit.oid))[0]
                assert info.size==record['archive_bytes']
                if info.lfs:
                    assert info.lfs.sha256==record['archive_sha256']
                else:
                    assert record['archive_bytes'] < 10*1024*1024
                    url='https://huggingface.co/'+plan['repo']+'/resolve/'+commit.oid+'/'+op.path_in_repo
                    data=requests.get(url,headers={'Authorization':'Bearer '+token},timeout=90).content
                    assert hashlib.sha256(data).hexdigest()==record['archive_sha256']
                result = {'status': 'HF_UPLOAD_VERIFIED', 'group': record['group'], 'archive_bytes': record['archive_bytes'],
                          'archive_sha256': record['archive_sha256'], 'repo_path': op.path_in_repo, 'commit': commit.oid,
                          'epoch': time.time(), 'elapsed_seconds': time.time() - started, 'model_disk_staging': False}
                receipt.write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(result), flush=True)
                break
            except Exception as error:
                failures += 1
                # Never log HTTP requests/headers/presigned URLs or credentials.
                (ROOT/'failure-latest.json').write_text(json.dumps({'group':record['group'],'error_type':type(error).__name__,'count':failures,'epoch':time.time()}))
                print(json.dumps({'group': record['group'], 'status': 'RETRY', 'error_type': type(error).__name__,
                                  'count': failures, 'epoch': time.time(), 'elapsed_seconds': time.time() - started}), flush=True)
                time.sleep(min(180, 30 * failures))
            finally:
                stream.close()
        if not receipt.exists():
            raise RuntimeError('Upload failed after bounded retries')
    (ROOT / 'upload-complete.json').write_text(json.dumps({'status': 'ALL_PLANNED_HF_UPLOADS_COMMITTED', 'epoch': time.time()}))


if __name__ == '__main__':
    main()
