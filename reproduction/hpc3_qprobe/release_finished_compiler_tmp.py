"""Reclaim generated nvcc temporary files from a completed own CPU build.

Keep its source tree, object files, build logs, final library and installed
environment. Requires fresh reader proof for all active owned Slurm nodes.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

BUILD=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/flash-attn-281-hpc-build')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()


def run(proof):
    from search_g1.qprobe_warmup import write_once
    from reproduction.hpc3_qprobe.release_archived_audits import reader_scan
    base=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
    receipt=base/'preparation/finished-compiler-tmp-released.json'
    if receipt.exists():raise ValueError('Compiler temporary release already recorded')
    actual=subprocess.check_output(['sacct','-j','705702','-X','-n','-P','--format=JobID,State,ExitCode,JobName'],text=True).strip()
    if actual!='705702|COMPLETED|0:0|cosearch-flash-cpu-build':raise ValueError('Own CPU build is not actually complete')
    completion=json.loads((BUILD/'build-complete.json').read_text())
    if completion['status']!='FLASH_ATTN_HPC_BUILD_IMPORT_PASS':raise ValueError('Missing actual completed library import')
    nodes=set(subprocess.check_output(['squeue','-u','mou176','-h','-t','RUNNING,COMPLETING','-o','%N'],text=True).split())|{socket.gethostname().split('.')[0]}
    scans=proof['reader_scans']
    if {x['host'] for x in scans}!=nodes or any(x['readers'] or x['unknown'] or time.time()-x['epoch']>300 for x in scans):raise ValueError('Fresh reader admission missing')
    lock=(BUILD/'build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        temporary=BUILD/'tmp'
        if temporary.is_symlink() or temporary.resolve()!=temporary:raise ValueError('Compiler temporary namespace differs')
        files=list(temporary.iterdir())
        if any(p.is_symlink() or not p.is_file() or not p.name.startswith('tmpxft_') for p in files):raise ValueError('Unexpected temporary file; do not remove')
        libraries=list((BUILD/'lib').glob('*.so'))
        if not libraries:raise ValueError('Final library is missing')
        kept={str(p):sha(p) for p in libraries+[BUILD/'build.sh',BUILD/'build-complete.json']}
        inventory=[dict(path=str(p),bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512,inode=p.stat().st_ino) for p in files]
        current=reader_scan()
        if current['readers'] or current['unknown']:raise ValueError('Reader appeared before temporary release')
        write_once(base/'preparation/finished-compiler-tmp-plan.json',dict(actual_build=actual,completion=completion,files=inventory,kept_full_sha=kept,reader_scans=scans))
        for p in files:p.unlink()
        if any(sha(Path(p))!=h for p,h in kept.items()):raise ValueError('Retained build source/library changed')
        result=dict(status='COMPLETED_BUILD_GENERATED_TMP_RELEASED',files=len(files),
            released_allocated_bytes=sum(x['allocated_bytes'] for x in inventory),kept_full_sha=kept,
            actual_completed_build=actual,source_and_environment_retained=True,checked_unix=time.time())
        write_once(receipt,result);return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':print(json.dumps(run(json.load(sys.stdin))))
