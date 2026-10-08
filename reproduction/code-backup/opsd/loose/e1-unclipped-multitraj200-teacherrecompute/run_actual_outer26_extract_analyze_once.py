"""Authorized local original-byte extraction and one observed26 analysis only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

H = Path(__file__).resolve().parent
PYTHON = '/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python'
C = H / 'paired-outer26-candidate'
KERNEL = H / 'paired-trajectory-cpu-analysis-candidate/analyze_outer.py'
RAW5 = H / 'job3304-first5-complete-originals-v3/raw'
REPO = H.parent / 'opsd-unclipped-multitraj200-teacherrecompute'
COPY = H / 'outer26-local-byte-copy-once-v2-original'
ANALYSIS = H / 'outer26-analysis-once-v2-original'
EXECUTION = H / 'outer26-extract-analysis-execution-once-v2-original'
COLLECTOR_SHA = '15e37516e98ac06d7fe03ec507c6805be56d8f8b20fcf2d1f92a39e95231580e'
INNER_SHA = '5bb17295306494a30bd610b9bd8751cf4d9535b5eeb00781f367b525444faeef'
PINS = {
    C / 'collect_outer26.py': COLLECTOR_SHA,
    C / 'analyze_observed26.py': '67791b2d3f1b284d9b96c36c589f7479b01c114f2b9391dc333340df6c257c49',
    C / 'AUTHORITY-v2.original.json': 'b64e39864be651423dee39b423191a3d87a6c401d496356330c37d186e6b0fd9',
    KERNEL: 'fa1a3dd12d762e8d18e2e577f9232642f8232d8fc03001ab0ffe406a4be6920d',
    H / 'OUTER26_JSON_DOCUMENT_EOF_REPAIR_INDEPENDENT_SOFTWARE_REVIEW.json':
        '20ac3f12398fcf2ada5212819441aa21f1309bbbfc394f6ad9c5fcb210ec1c69',
    H / 'paired-trajectory-cpu-analysis-candidate/actual-outer1-originals-v2/analysis/report.json':
        '371d6f99f2c4751d4f2de620c3ae88c4246445ae710399b750a4a170fc3f6687',
    H / 'outer26-collection-once-v2-original/wait.original.json': INNER_SHA,
    H / 'outer26-caller-execution-once-v2-original/wait.original.json':
        'a756f00193ec68ab0b496f35ac25e8342a000ff74d2c5542dbe5e2d3662347b8',
    H / 'outer26-collection-once-v2-original/stdout.original.gz':
        '0e39fa35573d3db66073953fff3beb9de6bdb403040c3005ab0b67d42a110fe3',
    H / 'outer26-collection-once-v2-original/caller-source.original.py':
        'b83ec638e6d40834168d19b23077c74976ef2cfac5781db3800088da2dae64da',
    RAW5 / 'bindings/source.json': '9a420576b556c1194d81e22886beb2734fd19b79a561faa5e791b2e0e51c64c8',
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record(path, expected=None):
    assert path.resolve(strict=True) == path
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    attrs = ('st_dev', 'st_ino', 'st_uid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    sig = lambda value: tuple(getattr(value, key) for key in attrs)
    assert sig(before) == sig(after) == sig(path.stat())
    assert expected is None or digest.hexdigest() == expected
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=before.st_size,
                device=before.st_dev, inode=before.st_ino, uid=before.st_uid)


def save(name, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    with (EXECUTION / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def git():
    results = []
    for args in (['rev-parse', 'HEAD'], ['status', '--porcelain']):
        argv = ['git', '--no-optional-locks', '-C', str(REPO)] + args
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0
        results.append(dict(argv=argv, waited=True, returncode=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
    assert results[0]['stdout'].strip() == 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
    assert results[1]['stdout'] == ''
    return results


def command(name, argv):
    save(name + '-command.original.json', dict(argv=argv, intent_utc=now()))
    start = time.monotonic()
    with (EXECUTION / (name + '-stdout.original')).open('xb') as out, (EXECUTION / (name + '-stderr.original')).open('xb') as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err)
        save(name + '-start.original.json', dict(pid=process.pid, started_utc=now()))
        rc = process.wait()
        out.flush(); err.flush()
        os.fsync(out.fileno()); os.fsync(err.fileno())
    value = dict(argv=argv, pid=process.pid, waited=True, returncode=rc, ended_utc=now(),
                 wall_seconds=time.monotonic() - start,
                 stdout=record(EXECUTION / (name + '-stdout.original')),
                 stderr=record(EXECUTION / (name + '-stderr.original')))
    save(name + '-wait.original.json', value)
    assert type(rc) is int and rc == 0
    return value


def main():
    EXECUTION.mkdir(exist_ok=False)
    for directory in (EXECUTION.parent, EXECUTION):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(fd); os.close(fd)
    save('executed-wrapper.original.py', Path(__file__).read_bytes())
    before = {str(p): record(p, pin) for p, pin in PINS.items()}
    inner = json.loads((H / 'outer26-collection-once-v2-original/wait.original.json').read_bytes())
    caller = json.loads((H / 'outer26-caller-execution-once-v2-original/wait.original.json').read_bytes())
    assert inner['waited'] is True and type(inner['returncode']) is int and inner['returncode'] == 0
    assert inner['authority_input_complete'] is True and inner['local_errors'] == []
    assert inner['runtime_sha256'] == COLLECTOR_SHA and inner['compressed_bytes'] == 27447567
    assert inner['caller_sha256'] == caller['caller_sha256'] == PINS[H / 'outer26-collection-once-v2-original/caller-source.original.py']
    assert caller['waited'] is True and type(caller['returncode']) is int and caller['returncode'] == 0
    source = json.loads((RAW5 / 'bindings/source.json').read_bytes())
    source_before = {rel: record(REPO / rel, pin) for rel, pin in source['source_files'].items()}
    git_before = git()
    save('input-qualification.original.json', dict(runtime=record(Path(__file__)), records=before,
         actual_inner_wait=inner, actual_caller_wait=caller, source_records=source_before, git=git_before,
         scope='LOCAL_BYTE_COPY_AND_OBSERVED26_ANALYSIS_NOT_TRAINING_OR_EFFECT_ACCEPTANCE'))
    base = [PYTHON, '-B', str(C / 'analyze_observed26.py'), '--kernel', str(KERNEL),
            '--collector', str(C / 'collect_outer26.py'), '--collector-sha256', COLLECTOR_SHA]
    extract = command('extract', base + ['extract', '--gzip', str(H / 'outer26-collection-once-v2-original/stdout.original.gz'),
              '--output', str(COPY), '--wait-record', str(H / 'outer26-collection-once-v2-original/wait.original.json'),
              '--wait-sha256', INNER_SHA])
    inventory = record(COPY / 'observed-inventory.json')
    analyze = command('analyze', base + ['analyze', '--raw26', str(COPY), '--inventory-sha256', inventory['sha256'],
              '--raw5', str(RAW5), '--source-repo', str(REPO), '--outer1-report',
              str(H / 'paired-trajectory-cpu-analysis-candidate/actual-outer1-originals-v2/analysis/report.json'),
              '--output', str(ANALYSIS)])
    after = {str(p): record(p, pin) for p, pin in PINS.items()}
    source_after = {rel: record(REPO / rel, pin) for rel, pin in source['source_files'].items()}
    git_after = git()
    assert before == after and source_before == source_after and git_before == git_after
    report = record(ANALYSIS / 'report.json')
    save('completed.original.json', dict(kind='ACTUAL_LOCAL_OUTER26_COPY_AND_ANALYSIS_EXECUTION',
         ended_utc=now(), runtime=record(Path(__file__)), pins_before=before, pins_after=after,
         source_before=source_before, source_after=source_after, git_before=git_before, git_after=git_after,
         extract=extract, analyze=analyze, observed_inventory=inventory, report=report,
         actual_gpu_execution=False, actual_remote_execution=False, canonical_status_modified=False))
    print(json.dumps(dict(report=report, inventory=inventory, waited=True, returncode=0)))


if __name__ == '__main__':
    main()
