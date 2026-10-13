"""Archive two explicitly selected closed failure artifacts from HPC3 to HK3.

Keep their small original evidence and all active training assets. Release is
separate from copying and requires independent full SHA and all-node readers.
No training, model loading, GPU reservation, or workstation data staging.
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

from reproduction.hpc3_qprobe.archive_failed_rollouts import SSH, atomic, remote, sha

ROOT=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011')
BASE=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
DEST=Path('/backup/shaoshitong/qprobe-teacher-warmup200-20261014/storage-relief/closed-failure-evidence')
DATA=(
    'historical-evidence/raw/controller-generation-return-2509698-1791710651417218753.pt',
    'failed-job-709923/logs/retriever.log',
)
HISTORICAL_SHA='f1287cbe5a5a5c331ae349f262843a2fb0243b5b9cdf795d4d77a070a8599d5a'
EXPECTED_TERMINALS={
    '709265|FAILED|1:0|mou176|cosearch-instruct200',
    '709923|FAILED|1:0|mou176|cosearch-instruct200',
}


def terminals():
    actual=subprocess.check_output(['sacct','-j','709265,709923','-X','-n','-P',
        '--format=JobID,State,ExitCode,User,JobName'],universal_newlines=True)
    lines=set(actual.strip().splitlines())
    if lines!=EXPECTED_TERMINALS:raise ValueError('Selected source jobs are not the verified terminal failures')
    return sorted(lines)


def local_manifest():
    terminal=terminals()
    if ROOT.resolve()!=ROOT or ROOT.is_symlink():raise ValueError('Historical source namespace changed')
    paths=[ROOT/p for p in DATA]
    paths+=sorted((ROOT/'historical-evidence').glob('*.jsonl'))
    paths+=[ROOT/'failed-job-709923/preservation-receipt.json']
    files=[]
    for p in paths:
        if p.is_symlink() or not p.is_file() or p.resolve()!=p:raise ValueError('Selected source path is not a canonical regular file')
        before=p.stat();digest=sha(p);after=p.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Source changed during full readback')
        files.append(dict(relative_path=str(p.relative_to(ROOT)),bytes=after.st_size,
            allocated_bytes=after.st_blocks*512,sha256=digest,inode=after.st_ino,mtime_ns=after.st_mtime_ns,
            release_duplicate=str(p.relative_to(ROOT)) in DATA))
    by_path={x['relative_path']:x for x in files}
    if by_path[DATA[0]]['sha256']!=HISTORICAL_SHA:raise ValueError('Historical validation raw differs from original evidence')
    events=[json.loads(line) for line in (ROOT/'historical-evidence/runtime-2509698.jsonl').read_text().splitlines()]
    payload=[e['values'] for e in events if e.get('kind')=='raw_payload']
    if len(payload)!=1 or payload[0]['path']!=str(ROOT/DATA[0]) or payload[0]['sha256']!=HISTORICAL_SHA:
        raise ValueError('Historical raw lacks its original provenance')
    for role in ('main','reranker'):
        event=json.loads((ROOT/('historical-evidence/'+role+'-0.jsonl')).read_text().splitlines()[0])
        if event['job']!='709265':raise ValueError('Historical worker lineage differs')
    return dict(source=str(ROOT),actual_terminals=terminal,files=files,
        total_bytes=sum(x['bytes'] for x in files),release_paths=list(DATA),
        scope='failed-job historical regenerated validation5 raw and closed retriever log; no active suffix or checkpoint')


def source_manifest(env):
    # Execute the already published, SHA-verified same helper on the source.
    source=env.pop('CLOSED_EVIDENCE_SOURCE')
    command='cd '+shlex.quote(source)+' && python3 -m reproduction.hpc3_qprobe.archive_closed_evidence --manifest'
    result=subprocess.run(SSH+[command],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        universal_newlines=True,timeout=60)
    env['CLOSED_EVIDENCE_SOURCE']=source
    if result.returncode:raise RuntimeError(result.stderr[-3000:])
    return json.loads(result.stdout)


def verify_files(manifest):
    if manifest['source']!=str(ROOT) or manifest['release_paths']!=list(DATA):raise ValueError('Archive selection differs')
    for value in manifest['files']:
        rel=Path(value['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive path')
        p=DEST/rel
        if p.is_symlink() or not p.is_file() or p.resolve()!=p or p.stat().st_size!=value['bytes'] or sha(p)!=value['sha256']:
            raise ValueError('Destination full SHA differs: '+str(rel))


def archive(source):
    password=sys.stdin.readline().rstrip('\n')
    if not password:raise ValueError('Missing RAM-only authentication')
    env=dict(os.environ,SSHPASS=password,CLOSED_EVIDENCE_SOURCE=source);password=None
    DEST.mkdir(parents=True,exist_ok=True)
    if DEST.resolve()!=DEST or DEST.is_symlink():raise ValueError('Archive destination is not canonical')
    lock=(DEST/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        source_code_sha256=sha(__file__),source_code_path=source,destination=str(DEST),
        gpu_allocation=False,model_loading=False,training=False)
    def state(status,**values):atomic(DEST/'controller.json',dict(identity,status=status,epoch=time.time(),**values))
    try:
        state('VERIFYING_TERMINAL_SOURCE')
        manifest=source_manifest(env)
        path=DEST/'source-manifest.json'
        if path.exists():
            if json.loads(path.read_text())!=manifest:raise ValueError('Existing archive source manifest differs')
        else:atomic(path,manifest)
        fs=os.statvfs(str(DEST))
        if fs.f_bavail*fs.f_frsize<manifest['total_bytes']+1024**3:raise ValueError('Destination free space too low')
        state('TRANSFERRING',planned_bytes=manifest['total_bytes'],planned_files=len(manifest['files']))
        transport=' '.join(shlex.quote(x) for x in SSH[:-1])
        subprocess.run(['rsync','-a','--partial','--append-verify','--files-from=-','-e',transport,
            'mou176@10.120.81.30:'+str(ROOT)+'/',str(DEST)+'/'],
            input=''.join(x['relative_path']+'\n' for x in manifest['files']),env=env,
            universal_newlines=True,check=True)
        state('VERIFYING_DESTINATION_FULL_SHA',planned_bytes=manifest['total_bytes'])
        verify_files(manifest)
        if source_manifest(env)!=manifest:raise ValueError('Source changed across archival')
        receipt=dict(status='CLOSED_FAILURE_EVIDENCE_DESTINATION_FULL_SHA_PASS',
            source_manifest_sha256=sha(path),files=len(manifest['files']),bytes=manifest['total_bytes'],
            source=str(ROOT),destination=str(DEST),source_deleted=False,completed_unix=time.time())
        atomic(DEST/'complete.json',receipt);state('COMPLETE',receipt=receipt,exit_code=0)
    except Exception as error:
        state('FAILED',error_type=type(error).__name__,error=str(error),exit_code=1);raise
    finally:
        env.pop('SSHPASS',None);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def independent_verify():
    lock=(DEST/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        controller=json.loads((DEST/'controller.json').read_text())
        proc=Path('/proc')/str(controller['pid'])
        if proc.exists():
            actual=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
            if actual==controller['start_ticks']:raise ValueError('Archive writer is still alive')
        if controller['status']!='COMPLETE' or controller['exit_code']!=0:raise ValueError('Archive is not complete')
        manifest=json.loads((DEST/'source-manifest.json').read_text());verify_files(manifest)
        complete=json.loads((DEST/'complete.json').read_text())
        if complete['source_manifest_sha256']!=sha(DEST/'source-manifest.json'):raise ValueError('Completion manifest changed')
        proof=dict(complete,status='CLOSED_FAILURE_EVIDENCE_FRESH_DESTINATION_FULL_SHA_PASS',
            writer_alive=False,epoch=time.time(),source_manifest=manifest)
        atomic(DEST/'independent-fresh-verification.json',proof);return proof
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def readers():
    roots=[str(ROOT/x) for x in DATA]
    out=dict(host=socket.gethostname().split('.')[0],uid=os.geteuid(),epoch=time.time(),
        roots=roots,own_processes_inspected=0,readers=[],unknown=[],zombies_without_address_space=[])
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            if p.stat().st_uid!=os.geteuid():continue
            out['own_processes_inspected']+=1
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]=='Z' and int(fields[20])==0 and (p/'maps').read_text()=='':
                out['zombies_without_address_space'].append(int(p.name));continue
            fds=[]
            for fd in (p/'fd').iterdir():
                try:fds.append(os.readlink(str(fd)))
                except FileNotFoundError:pass
            maps=(p/'maps').read_text()
            for root in roots:
                matched=[x for x in fds if x==root or x==root+' (deleted)']
                if matched or root in maps:out['readers'].append(dict(pid=int(p.name),path=root,fd=matched,mapped=root in maps))
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:out['unknown'].append(int(p.name))
    return out


def once(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def release(proof):
    receipt=BASE/'preparation/closed-failure-evidence-released.json'
    if receipt.exists():raise ValueError('Release already recorded; inspect instead of repeating')
    if os.environ.get('SLURM_JOB_ID'):raise ValueError('Release must run on the controller')
    d=proof['destination_readback'];scans=proof['reader_scans'];manifest=d['source_manifest']
    if d['status']!='CLOSED_FAILURE_EVIDENCE_FRESH_DESTINATION_FULL_SHA_PASS' or d['source_deleted'] or d['writer_alive'] or d['destination']!=str(DEST):
        raise ValueError('Missing independent closed destination verification')
    if hashlib.sha256(json.dumps(manifest,indent=2).encode()).hexdigest()!=d['source_manifest_sha256']:
        raise ValueError('Transferred source manifest digest differs')
    if local_manifest()!=manifest:raise ValueError('Source changed before duplicate release')
    nodes=set(subprocess.check_output(['squeue','-u','mou176','-h','-t','RUNNING,COMPLETING','-o','%N'],universal_newlines=True).split())|{socket.gethostname().split('.')[0]}
    if {x['host'] for x in scans}!=nodes or any(x['uid']!=os.geteuid() or x['roots']!=[str(ROOT/v) for v in DATA] or x['readers'] or x['unknown'] or not 0<=time.time()-x['epoch']<300 for x in scans):
        raise ValueError('Reader admission is occupied, incomplete or stale')
    live=readers()
    if live['readers'] or live['unknown']:raise ValueError('A controller reader appeared before release')
    once(BASE/'preparation/closed-failure-evidence-retained-location.json',proof)
    released=[x for x in manifest['files'] if x['release_duplicate']]
    if {x['relative_path'] for x in released}!=set(DATA):raise ValueError('Release selection differs')
    for value in released:(ROOT/value['relative_path']).unlink()
    for name in ('historical-evidence/raw','failed-job-709923/logs'):
        once(ROOT/name/'ARCHIVED_ON_HKUST3.json',dict(destination=str(DEST),
            source_manifest_sha256=d['source_manifest_sha256'],files=[x for x in released if x['relative_path'].startswith(name+'/')],historical_only=True))
    result=dict(status='VERIFIED_CLOSED_FAILURE_DUPLICATES_RELEASED',files=len(released),
        released_bytes=sum(x['bytes'] for x in released),released_allocated_bytes=sum(x['allocated_bytes'] for x in released),
        retained_archive=str(DEST),original_small_provenance_retained=True,active_training_modified=False,epoch=time.time())
    once(receipt,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--manifest',action='store_true');g.add_argument('--archive',metavar='PUBLISHED_SOURCE')
    g.add_argument('--verify',action='store_true');g.add_argument('--readers',action='store_true');g.add_argument('--release',action='store_true')
    a=p.parse_args()
    result=local_manifest() if a.manifest else archive(a.archive) if a.archive else independent_verify() if a.verify else readers() if a.readers else release(json.load(sys.stdin))
    if result is not None:print(json.dumps(result))
