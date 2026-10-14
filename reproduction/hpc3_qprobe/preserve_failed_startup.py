"""Preserve the two observed terminal startup failures via the shared account.

This is an observer archive, not owner-authorized recovery or a new submission.
Original logs, source, claims and node-local caches remain untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
TRAIN=ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery717827-20261014'
PRIOR=ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery717765-20261014'
SOURCE='594d43b73c5a1e5159a24fb0d97880ff22fb0056'


def accounting():
    command=['sacct','-a','-nP','-j','717938,717934','-X',
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition,Start,End,NodeList']
    raw=subprocess.check_output(command,text=True,timeout=30)
    selected=[row for line in raw.splitlines() if (row:=line.split('|'))[0] in ('717938','717934')]
    assert len(selected)==2,'Missing or duplicated terminal accounting row'
    rows={row[0]:row for row in selected}
    assert len(rows)==2
    for job,node,partition in (('717938','ACD1-12','emergency_acd'),('717934','ACD1-67','acd_ue')):
        assert rows[job][:8]==[job,'FAILED','1:0','jxu957','14090','jxu957','formal-user',partition]
        assert rows[job][-1]==node and rows[job][8] not in ('Unknown','None') and rows[job][9] not in ('Unknown','None')
    return dict(command=command,stdout=raw,rows=rows,checked_unix=time.time())


def preserve():
    assert os.getuid() in (21930,14090)
    before=accounting()
    run=json.loads((TRAIN/'logs/run-manifest.json').read_text())
    assert run['source_commit']==SOURCE and run['completed_updates']==0 and run['training_world_size']==4
    for name in ('training-launch-claim.json','actual-updates.jsonl','actual-attempts.jsonl','search-g1-common8-training.log'):
        assert not (TRAIN/'logs'/name).exists(),'Actor/teacher work must use full-state preservation instead'
    assert not list((TRAIN/'teacher-warmup-cache').glob('chunk-*.json'))
    main=json.loads((TRAIN/'preparation/training-submission.json').read_text())
    cleanup=json.loads((PRIOR/'preparation/cleanup717827-submission.json').read_text())
    assert main['job']=='717938' and main['commit']==SOURCE and main['returncode']==0
    assert cleanup['job']=='717934' and cleanup['commit']==SOURCE and cleanup['returncode']==0
    stage=json.loads((TRAIN/'logs/pipeline-status.json').read_text())
    closed=json.loads((TRAIN/'logs/cleanup-status.json').read_text())
    assert stage['stage']=='FAILED' and stage['error']=="AssertionError('model-initial failed; see project log')"
    assert closed['cleaned'] is True and closed['run_id']==run['run_id']==stage['run_id']
    assert 'Internal and actual separate actor logprob entrypoints disagree' in (TRAIN/'logs/model-initial.log').read_text()
    cleanup_log=PRIOR/'logs/cleanup717827-717934.log'
    assert "KeyError: 'pid'" in cleanup_log.read_text()
    targets=[(TRAIN,'717938',list((TRAIN/'logs').rglob('*'))+
              [TRAIN/'preparation'/n for n in ('training-submission.json','training-submit-claim.json')]),
             (PRIOR,'717934',[cleanup_log]+[PRIOR/'preparation'/n for n in
                ('cleanup717827-submission.json','cleanup717827-submit-claim.json')]+
              list((PRIOR/'failure-cleanup717827/logs').glob('*.json')))]
    result=[]
    for base,job,paths in targets:
        evidence=base/('failed-job-'+job+'-observer')
        marker=evidence/'terminal-and-files.json'
        if marker.exists():
            assert not marker.is_symlink()
            saved=json.loads(marker.read_text())
            assert saved['kind']=='SHARED_ACCOUNT_OBSERVER_ONLY' and saved['job']==job and saved['source_commit']==SOURCE
            assert saved['owner_recovery_authorized_by_this_record'] is False
            for name,record in saved['files'].items():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                assert not (base/name).is_symlink() and not (evidence/name).is_symlink()
                assert hashlib.sha256((base/name).read_bytes()).hexdigest()==record['sha256']
                assert hashlib.sha256((evidence/name).read_bytes()).hexdigest()==record['sha256']
            result.append(dict(job=job,path=str(evidence),status='OBSERVED_FAILED_STARTUP_ARCHIVE_SHA_PASS'))
            continue
        assert not evidence.exists(),'Inspect partial failure archive before repeating'
        paths=[p for p in paths if p.is_file()]
        assert sum(p.stat().st_size for p in paths)<30*1024**2,'Small code/log archive only'
        evidence.mkdir(mode=0o770);records={}
        for path in paths:
            assert not path.is_symlink()
            name=path.relative_to(base).as_posix();stat=path.stat();raw=path.read_bytes()
            target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            after=path.stat()
            assert (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            sha=hashlib.sha256(raw).hexdigest()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==sha==hashlib.sha256(target.read_bytes()).hexdigest()
            records[name]=dict(sha256=sha,bytes=len(raw),source_uid=stat.st_uid)
        after=accounting();assert before['rows']==after['rows']
        value=dict(kind='SHARED_ACCOUNT_OBSERVER_ONLY',observer_uid=os.getuid(),job=job,
            accounting_before=before,accounting_after=after,source_commit=SOURCE,files=records,
            teacher_trajectories=0,formal_updates=0,owner_recovery_authorized_by_this_record=False,
            original_files_unchanged=True,checked_unix=time.time())
        with marker.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        result.append(dict(job=job,path=str(evidence),files=len(records),bytes=sum(r['bytes'] for r in records.values()),
            manifest_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),status='OBSERVED_FAILED_STARTUP_ARCHIVE_SHA_PASS'))
    return dict(status='ACTUAL_TERMINAL_FAILURES_PRESERVED',archives=result,checked_unix=time.time())


if __name__=='__main__':print(json.dumps(preserve()),flush=True)
