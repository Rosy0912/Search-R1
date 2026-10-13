"""HKUST3-only archive of one terminal failed job's historical rollout audits.

The password arrives through stdin and stays in RAM. This never trains, loads
models, allocates a GPU or removes source files. Source release is a separate
action after independent fresh destination SHA and dependency checks.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

SOURCE='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011/failed-job-709923/rollouts'
DEST=Path('/backup/shaoshitong/qprobe-teacher-warmup200-20261014/storage-relief/failed-job-709923-rollouts')
SSH=['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes',
     '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
     '-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts',
     '-oConnectTimeout=8','mou176@10.120.81.30']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def atomic(path,value):
    path=Path(path);temporary=path.with_suffix('.partial')
    with temporary.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    temporary.replace(path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def remote(env,code,timeout=120):
    result=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],env=env,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr[-4000:])
    return result.stdout


def source_manifest(env):
    code='''import hashlib,json,subprocess
from pathlib import Path
root=Path(SOURCE_VALUE)
assert root.resolve()==root and root.is_dir() and not root.is_symlink()
actual=subprocess.check_output(['sacct','-j','709923','-X','-n','-P','--format=JobID,State,ExitCode,User,JobName'],universal_newlines=True).strip()
assert actual=='709923|FAILED|1:0|mou176|cosearch-instruct200',actual
files=[]
for p in sorted(root.rglob('*')):
 assert not p.is_symlink()
 if p.is_dir():continue
 rel=p.relative_to(root);assert len(rel.parts)==2 and rel.parts[0] in ('main','reranker') and p.suffix=='.jsonl' and p.stem.isdecimal() and 1<=int(p.stem)<=139
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 files.append(dict(relative_path=str(rel),bytes=after.st_size,sha256=h.hexdigest(),inode=after.st_ino,mtime_ns=after.st_mtime_ns))
assert len(files)==278
preservation=root.parent/'preservation-receipt.json';p=json.loads(preservation.read_text())
assert p['latest_complete_outer']==139 and p['recovery_checkpoint']==100 and p['recompute']==[101,140]
value=dict(source=str(root),source_job=709923,actual_terminal=actual,files=files,total_bytes=sum(x['bytes'] for x in files),preservation_receipt_sha256=hashlib.sha256(preservation.read_bytes()).hexdigest(),scope='historical audits of terminal failed job; not current suffix or frozen-teacher bootstrap labels')
print(json.dumps(value))
'''.replace('SOURCE_VALUE',repr(SOURCE))
    return json.loads(remote(env,code))


def verify_destination(manifest):
    for value in manifest['files']:
        path=DEST/value['relative_path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=value['bytes'] or sha(path)!=value['sha256']:
            raise ValueError('Historical audit destination SHA differs: '+value['relative_path'])


def run():
    password=sys.stdin.readline().rstrip('\n')
    if not password:raise ValueError('Missing RAM-only source authentication')
    env=dict(os.environ,SSHPASS=password);password=None
    DEST.mkdir(parents=True,exist_ok=True)
    if DEST.is_symlink() or DEST.resolve()!=DEST:raise ValueError('Archive destination is not canonical')
    lock=(DEST/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    process=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        source=SOURCE,destination=str(DEST),source_code_sha256=sha(__file__),checked_unix=time.time(),
        gpu_allocation=False,model_loading=False,training=False)
    def state(stage,**values):atomic(DEST/'controller.json',dict(process,status=stage,epoch=time.time(),**values))
    try:
        state('VERIFYING_TERMINAL_SOURCE')
        manifest=source_manifest(env)
        existing=DEST/'source-manifest.json'
        if existing.exists():
            if json.loads(existing.read_text())!=manifest:raise ValueError('Source namespace changed across archive recovery')
        else:atomic(existing,manifest)
        if not (DEST/'complete.json').exists():
            state('TRANSFERRING',planned_files=len(manifest['files']),planned_bytes=manifest['total_bytes'])
            transport=' '.join(shlex.quote(x) for x in SSH[:-1])
            subprocess.run(['rsync','-a','--partial','--append-verify','-e',transport,
                'mou176@10.120.81.30:'+SOURCE+'/',str(DEST)+'/'],env=env,check=True)
            state('VERIFYING_DESTINATION_FULL_SHA',planned_bytes=manifest['total_bytes'])
            verify_destination(manifest)
            # Source must remain byte-for-byte stable across the transfer.
            if source_manifest(env)!=manifest:raise ValueError('Source changed while archiving failed audits')
            receipt=dict(status='OLD_FAILED_JOB_AUDITS_DESTINATION_FULL_SHA_PASS',
                source_manifest_sha256=sha(existing),files=len(manifest['files']),bytes=manifest['total_bytes'],
                source=SOURCE,destination=str(DEST),source_code_sha256=process['source_code_sha256'],
                historical_only=True,source_deleted=False,completed_unix=time.time())
            atomic(DEST/'complete.json',receipt)
        else:
            verify_destination(manifest)
            receipt=json.loads((DEST/'complete.json').read_text())
            if receipt['source_manifest_sha256']!=sha(existing):raise ValueError('Completed archive manifest changed')
        state('COMPLETE',receipt=receipt,exit_code=0)
        return 0
    except Exception as error:
        state('FAILED',error_type=type(error).__name__,error=str(error),exit_code=1)
        raise
    finally:
        env.pop('SSHPASS',None);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':sys.exit(run())
