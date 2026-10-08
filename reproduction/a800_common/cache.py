"""Claim and clean only this fresh local run's disk-backed corpus cache."""
import hashlib
import json
import os
import shutil
import stat
import tarfile
import time
from pathlib import Path
from reproduction.a800.owned_processes import identity
from reproduction.common8.stage_retrieval import ASSETS,atomic
from reproduction.common8.temporary_cache import CORPUS_MEMBER,capacity_floor


def create(base,run,execution):
    assert run['execution_backend']==execution['execution_backend']=='ssh_local'
    assert run['execution_id']==execution['execution_id']
    root=base/'cache/retrieval'/run['execution_id']
    assert not root.exists() and not root.is_symlink()
    root.parent.mkdir(parents=True,exist_ok=True)
    claim=dict(run_id=run['run_id'],slurm_job_id=None,execution_backend='ssh_local',
        execution_id=run['execution_id'],uid=os.getuid(),root=str(root),
        creator=identity(os.getpid()),phase='RESERVED',created_unix=time.time(),
        cache_storage='private_run_local_disk',original_assets_owned=False)
    path=base/'logs/corpus-cache-claim.json'
    with path.open('x') as f:
        json.dump(claim,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)
    root.mkdir(mode=0o700)
    claim.update(phase='CREATED',root_device=root.stat().st_dev,root_inode=root.stat().st_ino)
    atomic(path,claim)
    return root


def _binding(base,root):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    claim=json.loads((base/'logs/corpus-cache-claim.json').read_text())
    assert run['execution_backend']==claim['execution_backend']=='ssh_local'
    assert run['run_id']==claim['run_id'] and run['execution_id']==claim['execution_id']
    assert claim['slurm_job_id'] is None and claim['uid']==os.getuid()
    assert root==Path(claim['root'])==base/'cache/retrieval'/run['execution_id']
    assert not root.is_symlink()
    if root.exists():
        assert root.stat().st_uid==claim['uid']
        assert (root.stat().st_dev,root.stat().st_ino)==(claim['root_device'],claim['root_inode'])
    return run,claim


def extract_corpus(base,root,archive):
    run,claim=_binding(base,root)
    creator=identity(os.getpid())
    assert all(creator[k]==claim['creator'][k] for k in ('pid','start_ticks','sid','uid'))
    assert claim['phase']=='CREATED' and 'corpus_extraction' not in claim
    assert archive==Path(run['retrieval_asset_paths']['wiki-18.jsonl']) and not archive.is_symlink()
    available=shutil.disk_usage(root).free
    checkpoints=json.loads((base/'logs/checkpoint-disk-admission.json').read_text())
    assert checkpoints['accepted'] is True and checkpoints['run_id']==run['run_id']
    combined=capacity_floor()+checkpoints['required_free_bytes']
    capacity=dict(required_capacity_bytes=capacity_floor(),actual_disk_free_bytes=available,
        required_checkpoint_free_bytes=checkpoints['required_free_bytes'],required_combined_free_bytes=combined,
        checked_unix=time.time(),scope='private local JSONL and simultaneous Arrow temporary/final disk allowance, retaining space for all requested checkpoints')
    claim['capacity_admission']=capacity;atomic(base/'logs/corpus-cache-claim.json',claim)
    assert available>=combined, 'Insufficient combined disk space for private corpus cache and all requested checkpoints'
    partial=root/'wiki_dump.jsonl.partial';target=root/'wiki_dump.jsonl'
    assert not partial.exists() and not target.exists()
    record=None
    try:
        source_fd=os.open(archive,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(source_fd,'rb') as source:
            source_stat=os.fstat(source.fileno())
            assert stat.S_ISREG(source_stat.st_mode) and source_stat.st_size==ASSETS['wiki-18.jsonl']['size']
            with tarfile.open(fileobj=source,mode='r:*') as bundle:
                members=bundle.getmembers();assert len(members)==1
                member=members[0]
                metadata=dict(name=member.name,size=member.size,uid=member.uid,gid=member.gid,
                    mode=member.mode,type=member.type.decode(),linkname=member.linkname)
                assert metadata==CORPUS_MEMBER and member.isfile(), 'Original corpus TAR member changed'
                content=bundle.extractfile(member);assert content is not None
                with partial.open('xb') as output:
                    st=os.fstat(output.fileno())
                    record=dict(phase='EXTRACTING',source=str(archive),source_device=source_stat.st_dev,
                        source_inode=source_stat.st_ino,member=metadata,output_device=st.st_dev,
                        output_inode=st.st_ino,started_unix=time.time())
                    claim['corpus_extraction']=record;atomic(base/'logs/corpus-cache-claim.json',claim)
                    h=hashlib.sha256();count=0
                    while True:
                        chunk=content.read(8*1024*1024)
                        if not chunk:break
                        output.write(chunk);h.update(chunk);count+=len(chunk)
                    output.flush();os.fsync(output.fileno())
                assert count==CORPUS_MEMBER['size']
                assert (archive.lstat().st_dev,archive.lstat().st_ino)==(source_stat.st_dev,source_stat.st_ino)
                assert (partial.lstat().st_dev,partial.lstat().st_ino)==(record['output_device'],record['output_inode'])
                os.link(partial,target,follow_symlinks=False);partial.unlink()
                st=target.lstat();assert stat.S_ISREG(st.st_mode) and st.st_uid==claim['uid'] and st.st_nlink==1
                assert (st.st_dev,st.st_ino,st.st_size)==(record['output_device'],record['output_inode'],count)
                fd=os.open(root,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
                try:os.fsync(fd)
                finally:os.close(fd)
                record.update(phase='READY',content_sha256=h.hexdigest(),content_size=count,
                    completed_unix=time.time(),scope='verified original TAR member bytes/order in private local cache')
                atomic(base/'logs/corpus-cache-claim.json',claim)
                atomic(base/'logs/corpus-extraction.json',record)
                return target
    except BaseException as error:
        if record is not None:
            record.update(phase='FAILED',error=repr(error),failed_unix=time.time())
            try:atomic(base/'logs/corpus-cache-claim.json',claim)
            except OSError:pass
        raise


def cleanup(base):
    saved=json.loads((base/'logs/corpus-cache-claim.json').read_text())
    root=Path(saved['root']);run,claim=_binding(base,root)
    ledger=json.loads((base/'logs/process-ledger.json').read_text())
    assert ledger['run_id']==run['run_id']
    records=[claim['creator']]+[r for job in ledger['jobs'].values() for r in job['ledger'].values()]
    for record in records:
        try:actual=identity(record['pid'])
        except (FileNotFoundError,ProcessLookupError):continue
        assert actual['state']=='Z' or actual['start_ticks']!=record['start_ticks'], 'Recorded cache reader remains live'
    sizes=claim.get('removed_file_sizes',{})
    if root.exists():
        entries=list(root.rglob('*'))
        assert all(not p.is_symlink() and p.stat().st_uid==claim['uid'] and
            (p.is_file() or p.is_dir()) for p in entries), 'Unowned/symlink/special cache entry; refuse deletion'
        sizes.update({str(p.relative_to(root)):p.stat().st_size for p in entries if p.is_file()})
        claim.update(phase='CLEANING',removed_file_sizes=sizes);atomic(base/'logs/corpus-cache-claim.json',claim)
        for p in sorted(entries,key=lambda p:len(p.parts),reverse=True):
            if p.is_dir():p.rmdir()
            else:p.unlink()
        root.rmdir()
    claim.update(phase='CLEANED',removed_file_sizes=sizes);atomic(base/'logs/corpus-cache-claim.json',claim)
    proof=dict(run_id=run['run_id'],slurm_job_id=None,execution_backend='ssh_local',
        execution_id=run['execution_id'],cleaned=True,root=str(root),removed_file_sizes=sizes,
        original_assets_removed=False,checked_unix=time.time())
    atomic(base/'logs/corpus-cache-cleanup.json',proof)
    return proof
