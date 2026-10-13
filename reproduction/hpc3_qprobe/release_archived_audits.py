"""Release verified duplicate historical audits; retain their remote archive.

Requires independent fresh HKUST3 SHA proof, terminal source job, unchanged
source bytes and fresh own-process reader checks on every active Slurm node.
No model/optimizer/active cache/current rollout or training source is removed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

SOURCE=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011/failed-job-709923/rollouts')
BUILD=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/flash-attn-281-hpc-build')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):value.update(c)
    return value.hexdigest()


def reader_scan():
    """Only matching path names, never process arguments or environment secrets."""
    roots=(str(SOURCE),str(BUILD/'tmp'))
    out=dict(host=socket.gethostname().split('.')[0],uid=os.geteuid(),epoch=time.time(),
             roots=list(roots),own_processes_inspected=0,readers=[],unknown=[])
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            if p.stat().st_uid!=os.geteuid():continue
            out['own_processes_inspected']+=1
            paths=[]
            for fd in (p/'fd').iterdir():
                try:paths.append(os.readlink(str(fd)))
                except FileNotFoundError:pass
            maps=(p/'maps').read_text()
            for root in roots:
                matched=[v for v in paths if v.startswith(root+'/') or v==root]
                if matched or root+'/' in maps:out['readers'].append(dict(pid=int(p.name),root=root,fd_paths=matched,mapped=root+'/' in maps))
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:out['unknown'].append(int(p.name))
    return out


def run(proof,receipt_path):
    from search_g1.qprobe_warmup import write_once
    receipt_path=Path(receipt_path)
    if receipt_path.exists():raise ValueError('Release already recorded; inspect instead of repeating')
    if os.environ.get('SLURM_JOB_ID'):raise ValueError('Source release must run on the controller, not a training allocation')
    destination=proof['destination_readback'];manifest=proof['source_manifest'];scans=proof['reader_scans']
    if destination['status']!='OLD_FAILED_JOB_AUDITS_FRESH_DESTINATION_FULL_SHA_PASS' or destination['source_deleted'] or destination['writer_alive']:
        raise ValueError('Archive is not independently verified and closed')
    if destination['source_manifest_sha256']!=proof['source_manifest_sha256'] or destination['files']!=278 or destination['bytes']!=manifest['total_bytes']:
        raise ValueError('Archive receipt and actual source manifest differ')
    if manifest['source']!=str(SOURCE) or SOURCE.is_symlink() or SOURCE.resolve()!=SOURCE:
        raise ValueError('Historical release namespace differs')
    nodes=set(subprocess.check_output(['squeue','-u','mou176','-h','-t','RUNNING,COMPLETING','-o','%N'],text=True).split())
    expected=nodes|{socket.gethostname().split('.')[0]}
    if {x['host'] for x in scans}!=expected or any(x['uid']!=os.geteuid() or x['readers'] or x['unknown'] or time.time()-x['epoch']>300 for x in scans):
        raise ValueError('Source reader admission missing, occupied or stale')
    terminal=subprocess.check_output(['sacct','-j','709923','-X','-n','-P','--format=JobID,State,ExitCode,User,JobName'],text=True).strip()
    if terminal!='709923|FAILED|1:0|mou176|cosearch-instruct200':raise ValueError('Source job is not the terminal failed own job')
    expected_files={x['relative_path'] for x in manifest['files']}
    files={str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}
    if files!=expected_files or len(files)!=278:raise ValueError('Source file inventory changed')
    for item in manifest['files']:
        path=SOURCE/item['relative_path'];s=path.stat()
        if path.is_symlink() or (s.st_ino,s.st_size,s.st_mtime_ns)!=(item['inode'],item['bytes'],item['mtime_ns']) or sha(path)!=item['sha256']:
            raise ValueError('Source bytes changed before duplicate release: '+str(path))
    final_scan=reader_scan()
    if final_scan['readers'] or final_scan['unknown']:raise ValueError('A source reader appeared on the controller')
    # Persist exact retained location and proof before releasing any duplicate.
    write_once(receipt_path.with_name('historical-audit-retained-location.json'),dict(
        destination=destination,source_manifest_sha256=proof['source_manifest_sha256'],
        source_manifest=manifest,source_job_terminal=terminal,reader_scans=scans,
        scope='historical failed-job evidence preserved remotely; not new training labels'))
    released=0
    for item in manifest['files']:
        path=SOURCE/item['relative_path'];path.unlink();released+=item['bytes']
    write_once(SOURCE/'ARCHIVED_ON_HKUST3.json',dict(destination=destination['destination'],
        source_manifest_sha256=proof['source_manifest_sha256'],full_sha_files=278,bytes=released,
        historical_only=True,checked_unix=time.time()))
    result=dict(status='VERIFIED_HISTORICAL_DUPLICATE_RELEASED',released_logical_bytes=released,
        retained_archive=destination['destination'],files=278,source_job=709923,
        active_training_modified=False,checked_unix=time.time())
    write_once(receipt_path,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scan-readers',action='store_true');p.add_argument('--receipt')
    args=p.parse_args()
    print(json.dumps(reader_scan() if args.scan_readers else run(json.load(sys.stdin),args.receipt)))
