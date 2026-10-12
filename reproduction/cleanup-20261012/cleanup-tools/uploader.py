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
    keys = cfg.get('trusted_target_host_keys', {}).get('h100-10')
    if keys: args += ['-o', 'KnownHostsCommand=' + shlex.join(['/bin/echo', '\n'.join(keys)])]
    return args + [cfg['user'] + '@' + cfg['targets']['h100-10'], command]


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
            self.process = subprocess.Popen(source_args('python3 -c ' + shlex.quote(code)),
                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = self.process.stdout.read(size)
        if len(data) != size:
            raise IOError('Short source stream')
        self.position += len(data)
        if self.position//(64*1048576) != (self.position-len(data))//(64*1048576):
            (ROOT/'progress.json').write_text(json.dumps({'path':self.record['repo_path'],'source_read_bytes':self.position,'epoch':time.time()}))
        return data

    def close(self):
        self.stop_stream()
        super().close()


def main():
    lock=(ROOT/'uploader.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=sys.stdin.readline().strip();assert token
    plan=json.loads(PLAN.read_text());api=HfApi(token=token);assert api.model_info(plan['repo']).private
    done=[]
    for n,record in enumerate(plan['records']):
        receipt=ROOT/('receipt-%04d.json'%n)
        if receipt.exists():done.append(json.loads(receipt.read_text()));continue
        for attempt in range(3):
            stream=RemoteFile(record)
            try:
                sample=stream.read(min(512,record['archive_bytes']));stream.seek(0)
                op=CommitOperationAdd(path_in_repo=record['repo_path'],path_or_fileobj=sample);op.path_or_fileobj=stream
                op.upload_info=UploadInfo(size=record['archive_bytes'],sha256=bytes.fromhex(record['archive_sha256']),sample=sample)
                commit=api.create_commit(repo_id=plan['repo'],operations=[op],commit_message='H100 disk cleanup verified backup '+record['group'])
                info=api.get_paths_info(plan['repo'],[record['repo_path']],revision=commit.oid)[0]
                if info.lfs:
                    assert info.size==record['archive_bytes'] and info.lfs.sha256==record['archive_sha256']
                else:
                    import requests
                    response=requests.get('https://huggingface.co/'+plan['repo']+'/resolve/'+commit.oid+'/'+record['repo_path'],headers={'Authorization':'Bearer '+token},timeout=60);response.raise_for_status();assert hashlib.sha256(response.content).hexdigest()==record['archive_sha256']
                result=dict(record,status='HF_OFFICIAL_SHA_VERIFIED',commit=commit.oid,epoch=time.time());receipt.write_text(json.dumps(result));done.append(result);print(json.dumps({'status':result['status'],'path':record['repo_path'],'bytes':record['archive_bytes']}),flush=True);break
            except Exception as err:
                print(json.dumps({'status':'UPLOAD_ERROR','type':type(err).__name__,'attempt':attempt,'path':record['repo_path']}),flush=True)
                if attempt==2:raise RuntimeError('Upload or verification failed') from None
                time.sleep(10)
            finally:stream.close()
    (ROOT/'complete.json').write_text(json.dumps({'status':'ALL_FILES_HF_OFFICIAL_SHA_VERIFIED','records':done,'epoch':time.time()}))
if __name__=='__main__':main()
