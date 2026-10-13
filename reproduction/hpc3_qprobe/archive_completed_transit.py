"""Retain one completed historical retriever transit archive on HKUST3.

Reuse the pure-stdlib copy/full-SHA/reader machinery with a separate immutable
namespace. Only the archived compressed transit duplicate can be released;
installed environments, original provenance and all active assets remain.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import archive_closed_evidence as engine

ROOT=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
TRANSIT='migration-hf-20261009/retriever-supplement.tar.zst'
DEST=Path('/backup/shaoshitong/qprobe-teacher-warmup200-20261014/storage-relief/completed-retriever-transit')
ORIGINAL_SHA='edc951c6a9cade2134ca0badc58b03852aae800b73d55d498d1a4c3a95331787'
ORIGINAL_BYTES=3483281305
PREFIX='COMPLETED_TRANSIT_ARCHIVE'


def local_manifest():
    if ROOT.resolve()!=ROOT or ROOT.is_symlink():raise ValueError('Transit source namespace changed')
    folder=ROOT/'migration-hf-20261009'
    lock=(folder/'supplement-target.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        completed=json.loads((folder/'complete.json').read_text())
        historical=json.loads((folder/'target-fresh-verified.json').read_text())
        bound=json.loads((folder/'archive-verified.json').read_text())
        if completed['status']!='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' or historical['status']!='SUPPLEMENT_ALL_ORIGINAL_FILES_FRESH_SHA_VERIFIED':
            raise ValueError('Transit migration did not finish')
        if historical['archive_sha256']!=ORIGINAL_SHA or bound['archive_sha256']!=ORIGINAL_SHA or bound['archive_bytes']!=ORIGINAL_BYTES:
            raise ValueError('Historical transit identity differs')
        names=[TRANSIT]+['migration-hf-20261009/'+n for n in ('manifest.jsonl','complete.json','archive-verified.json','target-fresh-verified.json')]
        files=[]
        for name in names:
            p=ROOT/name
            if p.resolve()!=p or p.is_symlink() or not p.is_file():raise ValueError('Transit file is not canonical')
            before=p.stat();digest=engine.sha(p);after=p.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError('Transit source changed during full SHA')
            files.append(dict(relative_path=name,bytes=after.st_size,allocated_bytes=after.st_blocks*512,
                sha256=digest,inode=after.st_ino,mtime_ns=after.st_mtime_ns,release_duplicate=name==TRANSIT))
        if files[0]['bytes']!=ORIGINAL_BYTES or files[0]['sha256']!=ORIGINAL_SHA:raise ValueError('Actual transit full SHA differs')
        return dict(source=str(ROOT),files=files,total_bytes=sum(x['bytes'] for x in files),release_paths=[TRANSIT],
            scope='complete historical migration compressed transit only; all original archive bytes retained remotely; deployed environments retained',
            historical_migration_complete=True,deployed_current_files_not_claimed_as_historical_bytes=True)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def source_manifest(env):
    source=env['CLOSED_EVIDENCE_SOURCE']
    command='cd '+shlex.quote(source)+' && python3 -m reproduction.hpc3_qprobe.archive_completed_transit --manifest'
    p=subprocess.run(engine.SSH+[command],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        universal_newlines=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout)


engine.ROOT=ROOT;engine.DATA=(TRANSIT,);engine.DEST=DEST
engine.local_manifest=local_manifest;engine.source_manifest=source_manifest


def archive(source):
    engine.archive(source)
    complete=json.loads((DEST/'complete.json').read_text());complete['status']=PREFIX+'_DESTINATION_FULL_SHA_PASS'
    engine.atomic(DEST/'complete.json',complete)
    state=json.loads((DEST/'controller.json').read_text());state['receipt']=complete
    engine.atomic(DEST/'controller.json',state)


def verify():
    proof=engine.independent_verify();proof['status']=PREFIX+'_FRESH_DESTINATION_FULL_SHA_PASS'
    engine.atomic(DEST/'independent-fresh-verification.json',proof);return proof


def release(proof):
    receipt=engine.BASE/'preparation/completed-transit-released.json'
    if receipt.exists() or os.environ.get('SLURM_JOB_ID'):raise ValueError('Release already recorded or not on controller')
    d=proof['destination_readback'];manifest=d['source_manifest'];scans=proof['reader_scans']
    if d['status']!=PREFIX+'_FRESH_DESTINATION_FULL_SHA_PASS' or d['source_deleted'] or d['writer_alive'] or d['destination']!=str(DEST):
        raise ValueError('Missing independent closed destination verification')
    if hashlib.sha256(json.dumps(manifest,indent=2).encode()).hexdigest()!=d['source_manifest_sha256'] or local_manifest()!=manifest:
        raise ValueError('Source manifest or actual transit bytes changed')
    nodes=set(subprocess.check_output(['squeue','-u','mou176','-h','-t','RUNNING,COMPLETING','-o','%N'],universal_newlines=True).split())|{socket.gethostname().split('.')[0]}
    if {x['host'] for x in scans}!=nodes or any(x['uid']!=os.geteuid() or x['roots']!=[str(ROOT/TRANSIT)] or x['readers'] or x['unknown'] or not 0<=time.time()-x['epoch']<300 for x in scans):
        raise ValueError('Reader admission missing, occupied or stale')
    current=engine.readers()
    if current['readers'] or current['unknown']:raise ValueError('Controller reader appeared before release')
    engine.once(engine.BASE/'preparation/completed-transit-retained-location.json',proof)
    lock=(ROOT/'migration-hf-20261009/supplement-target.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        p=ROOT/TRANSIT;item=manifest['files'][0]
        if item['relative_path']!=TRANSIT or engine.sha(p)!=ORIGINAL_SHA:raise ValueError('Transit selection changed')
        p.unlink()
        engine.once(p.parent/'RETRIEVER_TRANSIT_ARCHIVED_ON_HKUST3.json',dict(destination=str(DEST),
            source_manifest_sha256=d['source_manifest_sha256'],archive_sha256=ORIGINAL_SHA,archive_bytes=ORIGINAL_BYTES))
        result=dict(status='VERIFIED_COMPLETED_TRANSIT_DUPLICATE_RELEASED',files=1,
            released_bytes=item['bytes'],released_allocated_bytes=item['allocated_bytes'],retained_archive=str(DEST),
            original_provenance_retained=True,deployed_environment_modified=False,epoch=time.time())
        engine.once(receipt,result);return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--manifest',action='store_true');g.add_argument('--archive',metavar='PUBLISHED_SOURCE')
    g.add_argument('--verify',action='store_true');g.add_argument('--readers',action='store_true');g.add_argument('--release',action='store_true')
    a=p.parse_args()
    r=local_manifest() if a.manifest else archive(a.archive) if a.archive else verify() if a.verify else engine.readers() if a.readers else release(json.load(sys.stdin))
    if r is not None:print(json.dumps(r))
