"""Stage verified original retrieval inputs inside this run's native allocation.

The temporary RAM files belong to one run/job and must be removed by its owned
cleanup. No GPU work, foreign service, cross-cluster relay or global mount.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from reproduction.common8.native_allocation import fields

ASSETS={
    'e5_Flat.index':dict(size=64559075373,sha256='69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166'),
    'wiki-18.jsonl':dict(size=14393579520,sha256='85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9')}
RESERVE=192*1024**3
SOURCE='10.33.0.13'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk=f.read(8*1024*1024)
            if not chunk:break
            h.update(chunk)
    return h.hexdigest()


def memory_bytes(tres):
    m=re.search(r'(?:^|,)mem=(\d+)([KMGT]?)(?:,|$)',tres)
    assert m, 'No actual memory allocation'
    # Slurm expresses unsuffixed memory in MiB.
    scale={'':2**20,'K':2**10,'M':2**20,'G':2**30,'T':2**40}
    return int(m.group(1))*scale[m.group(2)]


def resource_check(allocated,available,shm_free):
    size=sum(a['size'] for a in ASSETS.values())
    # Temporary inputs + simultaneous original CPU index + explicit process
    # reserve. This is a conservative launch floor, not a measured peak proof.
    floor=size+ASSETS['e5_Flat.index']['size']+RESERVE
    assert allocated>=floor, 'Native job memory request insufficient for retrieval staging'
    assert available>=floor, 'Actual available host memory insufficient for staging'
    assert shm_free>=size+1024**3, 'Insufficient private temporary-file capacity'
    return dict(input_bytes=size,required_memory_floor_bytes=floor,
        actual_allocation_bytes=allocated,actual_available_bytes=available,
        actual_shm_free_bytes=shm_free,scope='launch floor; actual model/coexistence peak remains unverified')


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    temp.replace(path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def binding(base):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8' and run['completed_updates']==0
    admission=json.loads((base/'logs/native-allocation-admission.json').read_text())
    assert admission['accepted'] is True and admission['run_id']==run['run_id']
    job_id=os.environ['SLURM_JOB_ID']
    assert admission['slurm_job_id']==job_id
    job=fields(subprocess.check_output(['scontrol','show','job','-o',job_id],text=True))
    assert job['JobState']=='RUNNING' and job['NodeList']==socket.gethostname().split('.')[0]
    assert job['JobId']==job_id
    # This version stages only onto the already verified H100-8 target.
    assert job['NodeList']=='h100-8', 'Re-admit another node before changing target'
    return run,job,admission


def stage(base,port):
    run,job,admission=binding(base)
    assert 1024<=port<=65535
    mem={line.split(':',1)[0]:int(line.split()[1])*1024
        for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    resources=resource_check(memory_bytes(job['TRES']),mem['MemAvailable'],shutil.disk_usage('/dev/shm').free)
    root=Path('/dev/shm')/('search-g1-common8-'+run['run_id']+'-'+job['JobId'])
    assert not root.exists() and not root.is_symlink(), 'Staging namespace already exists'
    claim_path=base/'logs/retrieval-staging-claim.json'
    claim=dict(run_id=run['run_id'],slurm_job_id=job['JobId'],uid=os.getuid(),root=str(root),
        pid=os.getpid(),start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
        phase='RESERVED',created_unix=time.time(),files=list(ASSETS))
    # Durably reserve this unique namespace before mkdir or any RAM allocation.
    with claim_path.open('x') as f:
        json.dump(claim,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(str(claim_path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    status=dict(run_id=run['run_id'],slurm_job_id=job['JobId'],stage='CREATING',accepted=False,
        root=str(root),resources=resources,files={},source_node='h100-9',target_node='h100-8')
    target=base/'logs/retrieval-staging-status.json'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        root.mkdir(mode=0o700,exist_ok=False)
        claim.update(phase='CREATED',root_device=root.stat().st_dev,root_inode=root.stat().st_ino)
        atomic(claim_path,claim)
        atomic(root/'owner.json',claim)
        status['stage']='TRANSFERRING';atomic(target,status)
        for name,expected in ASSETS.items():
            partial=root/(name+'.partial')
            request=urllib.request.Request(f'http://{SOURCE}:{port}/assets-root/assets/wiki/{name}')
            h=hashlib.sha256();count=0
            with opener.open(request,timeout=60) as response,partial.open('xb') as output:
                assert response.status==200 and int(response.headers['Content-Length'])==expected['size']
                while True:
                    chunk=response.read(8*1024*1024)
                    if not chunk:break
                    output.write(chunk);h.update(chunk);count+=len(chunk)
                output.flush();os.fsync(output.fileno())
            assert count==expected['size'] and h.hexdigest()==expected['sha256'], 'Transferred original asset differs: '+name
            final=root/name;assert not final.exists();partial.rename(final)
            # Independently re-read the actual target, not just network bytes.
            assert final.stat().st_size==expected['size'] and digest(final)==expected['sha256']
            status['files'][name]=dict(expected,path=str(final));atomic(target,status)
        status.update(stage='COMPLETE',accepted=True,completed_unix=time.time())
        atomic(target,status)
        claim.update(phase='STAGED');atomic(claim_path,claim)
        with (base/'logs/retrieval-assets.json').open('x') as f:
            json.dump(status,f,indent=2);f.flush();os.fsync(f.fileno())
        return status
    except BaseException as e:
        status.update(stage='FAILED',error=repr(e),failed_unix=time.time())
        claim.update(phase='FAILED',error=repr(e))
        try:
            atomic(claim_path,claim);atomic(target,status)
        except BaseException:pass  # Durable initial claim still owns startup remnants.
        # Failed target files remain owned until explicit pipeline cleanup;
        # never retry or silently overwrite this run's staging namespace.
        raise


def assert_quiescent(base,claim):
    from reproduction.a800.owned_processes import identity
    records=[dict(pid=claim['pid'],start_ticks=claim['start_ticks'])]
    ledger=base/'logs/process-ledger.json'
    if ledger.exists():
        saved=json.loads(ledger.read_text());assert saved['run_id']==claim['run_id']
        records.extend(record for job in saved['jobs'].values() for record in job['ledger'].values())
    for record in records:
        try:actual=identity(record['pid'])
        except (FileNotFoundError,ProcessLookupError):continue
        if actual['start_ticks']!=record['start_ticks'] or actual['state']=='Z':continue
        # A dedicated cleanup CLI may itself be in its caller's ledger;
        # it does no GPU/file-reader work. The original stager cannot clean.
        if record['pid']==os.getpid() and record is not records[0]:continue
        raise AssertionError('Recorded project reader/stager is still live')


def cleanup(base):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8'
    claim_path=base/'logs/retrieval-staging-claim.json'
    claim=json.loads(claim_path.read_text())
    assert claim['run_id']==run['run_id'] and claim['uid']==os.getuid()
    expected=Path('/dev/shm')/('search-g1-common8-'+run['run_id']+'-'+claim['slurm_job_id'])
    root=Path(claim['root']);assert root==expected and not root.is_symlink()
    assert_quiescent(base,claim)
    # Durable reservation is authoritative even when startup owner/status write
    # failed. Missing root after partial cleanup is safely repeatable.
    sizes=claim.get('cleanup_file_sizes',{})
    if root.exists():
        assert root.stat().st_uid==claim['uid']
        if 'root_inode' in claim:
            assert (root.stat().st_dev,root.stat().st_ino)==(claim['root_device'],claim['root_inode'])
        owner_path=root/'owner.json'
        if owner_path.exists():
            assert not owner_path.is_symlink()
            owner=json.loads(owner_path.read_text())
            assert all(owner[k]==claim[k] for k in ('run_id','slurm_job_id','uid','root'))
        allowed={'owner.json','owner.tmp'}|set(ASSETS)|{n+'.partial' for n in ASSETS}
        children=list(root.iterdir())
        assert all(p.name in allowed and p.is_file() and not p.is_symlink() for p in children), 'Unfamiliar temporary files: refuse broad deletion'
        sizes.update({p.name:p.stat().st_size for p in children})
        claim.update(phase='CLEANING',cleanup_file_sizes=sizes);atomic(claim_path,claim)
        # Remove the owner proof last; the persistent external claim also
        # survives an interruption between unlink and rmdir.
        for p in sorted(children,key=lambda p:p.name in ('owner.json','owner.tmp')):p.unlink()
        root.rmdir()
    proof=dict(run_id=run['run_id'],slurm_job_id=claim['slurm_job_id'],cleaned=True,
        root=str(root),removed_file_sizes=sizes,checked_unix=time.time())
    claim.update(phase='CLEANED');atomic(claim_path,claim)
    atomic(base/'logs/retrieval-staging-cleanup.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--port',type=int);p.add_argument('--cleanup',action='store_true');a=p.parse_args()
    if a.cleanup:print(json.dumps(cleanup(a.base)))
    else:
        assert a.port is not None
        print(json.dumps(stage(a.base,a.port)))
