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
    args += ['-o', 'ProxyCommand=' + shlex.join(prefix + ['-W', '%h:%p', cfg['user'] + '@' + cfg['gateway_ip']])]
    keys = cfg['trusted_target_host_keys']['h100-11']
    args += ['-o', 'KnownHostsCommand=' + shlex.join(['/bin/echo', '\n'.join(keys)])]
    return args + [cfg['user'] + '@' + cfg['targets']['h100-11'], command]


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
        if self.position - getattr(self, 'last_report', 0) >= 64 * 1048576:
            self.last_report = self.position
            p = ROOT / ('progress-' + hashlib.sha256(self.record['relative'].encode()).hexdigest()[:16] + '.json')
            tmp = p.with_suffix('.tmp')
            tmp.write_text(json.dumps({'epoch': time.time(), 'file': self.record['relative'], 'read_position': self.position, 'size': self.record['archive_bytes']}))
            tmp.replace(p)
        return data

    def close(self):
        self.stop_stream()
        super().close()


def verify(api, plan, record, dest, token, revision):
    info=api.get_paths_info(repo_id=plan['repo'],paths=[dest],revision=revision)[0]
    assert info.size==record['size'], 'HF size mismatch'
    if info.lfs:
        assert info.lfs.sha256==record['sha256'], 'HF LFS SHA mismatch'
    else:
        import requests
        from huggingface_hub import hf_hub_url
        assert record['size']<=8*1048576
        resp=requests.get(hf_hub_url(plan['repo'],dest,revision=revision),headers={'Authorization':'Bearer '+token},timeout=60)
        resp.raise_for_status()
        assert hashlib.sha256(resp.content).hexdigest()==record['sha256']
    return {'path':dest,'size':info.size,'sha256':record['sha256'],'revision':revision}

def main():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import threading
    lock=(ROOT/'uploader.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=sys.stdin.readline().strip();assert token
    plan=json.loads((ROOT/'plan.json').read_text())
    api=HfApi(token=token);assert api.model_info(plan['repo']).private
    (ROOT/'controller.json').write_text(json.dumps({'pid':os.getpid(),'epoch':time.time(),'total_bytes':plan['total_bytes'],'generation':'parallel4','concurrency':4,'bounded_ram':True}))
    receipts={};guard=threading.Lock()
    def write_json(path,value):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
    def mark(num,item):
        with guard:
            receipts[num]=item
            write_json(ROOT/'verified-progress.json',{'epoch':time.time(),'verified_bytes':sum(x['size'] for x in receipts.values()),'files':len(receipts),'total_bytes':plan['total_bytes'],'generation':'parallel4'})
    def worker(num,record):
        fl=(ROOT/('file-%04d.lock'%num)).open('w');fcntl.flock(fl,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker_api=HfApi(token=token);dest=plan['prefix']+record['relative'];receipt=ROOT/('file-%04d-verified.json'%num)
        if receipt.exists():
            old=json.loads(receipt.read_text());verify(worker_api,plan,record,dest,token,'main');mark(num,old);return
        rec=dict(record,archive=record['path'],archive_bytes=record['size'],archive_sha256=record['sha256'])
        for attempt in range(3):
            stream=RemoteFile(rec)
            try:
                sample=stream.read(min(512,record['size']));stream.seek(0)
                op=CommitOperationAdd(path_in_repo=dest,path_or_fileobj=sample)
                op.path_or_fileobj=stream;op.upload_info=UploadInfo(size=record['size'],sha256=bytes.fromhex(record['sha256']),sample=sample)
                commit=worker_api.create_commit(repo_id=plan['repo'],operations=[op],commit_message='H100 BAPO best150 backup '+record['relative'])
                item=verify(worker_api,plan,record,dest,token,commit.oid);item['epoch']=time.time()
                write_json(receipt,item);mark(num,item);print(json.dumps(item),flush=True);return
            except Exception as e:
                write_json(ROOT/('file-%04d-error.json'%num),{'epoch':time.time(),'file':dest,'type':type(e).__name__,'attempt':attempt+1})
                if attempt==2:raise RuntimeError(type(e).__name__) from None
                time.sleep(10)
            finally:stream.close()
    # Large independent files first; all paths have one writer protected by flock.
    order=sorted(enumerate(plan['records']),key=lambda pair:-pair[1]['size'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,num,record) for num,record in order]
        for future in as_completed(futures):future.result()
    final_revision=api.model_info(plan['repo']).sha
    for record in plan['records']:verify(api,plan,record,plan['prefix']+record['relative'],token,final_revision)
    write_json(ROOT/'complete.json',{'status':'ALL_SELECTED_HF_FILES_SHA_VERIFIED','epoch':time.time(),'total_bytes':plan['total_bytes'],'files':len(receipts),'revision':final_revision,'generation':'parallel4'})
if __name__=='__main__':main()
