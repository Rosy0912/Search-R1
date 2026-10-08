import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/mnt/data/shaoshitong/gigpo-e1-instruct200-independent-backup-20261009/backup5-once-v1')
ORIGINAL = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs/gigpo-e1-fresh200-live-job3340-1791441919057393097/checkpoints/actor/global_step_5')
SOURCE = Path('/mnt/shared/workspace/training_data_node_8') / ORIGINAL.relative_to('/mnt/data')
MANIFEST_SHA = '65b7263e10e5c3c6dbc20d3f53893ab3e11fb96beea88017458176f68c214c23'
RATE = 20 * 1024 * 1024
RESERVE = 128 * 1024**3

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def record(name, value):
    tmp = ROOT / (name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ROOT / name)

def admit():
    assert socket.gethostname() == 'h100-10'
    dest_mount = subprocess.check_output(['findmnt', '-n', '-T', '/mnt/data', '-o', 'FSTYPE'], text=True).strip()
    source_mount = subprocess.check_output(['findmnt', '-n', '-T', str(SOURCE), '-o', 'SOURCE'], text=True)
    assert dest_mount == 'ext4' and '10.32.0.8:/mnt/data' in source_mount
    assert os.stat('/mnt/data').st_dev != SOURCE.stat().st_dev
    raw = (SOURCE / 'training_state/manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA
    m = json.loads(raw)
    assert m['job_id'] == '3340' and m['logical_step'] == 5 and len(m['files']) == 22
    assert m['code_commit'] == 'ebe91bfebd56c3bd445219dcd94342d815f899f0'
    assert m['data_identity'] == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    entries = []
    for row in m['files']:
        rel = Path(row['path']).relative_to(ORIGINAL.parent.parent)
        assert rel.parts[:2] in [('actor', 'global_step_5'), ('algorithm', 'global_step_5')]
        assert '..' not in rel.parts and not (SOURCE.parent.parent / rel).is_symlink()
        assert (SOURCE.parent.parent / rel).is_file() and (SOURCE.parent.parent / rel).stat().st_size == row['bytes']
        entries.append({'relative_path': str(rel), 'bytes': row['bytes'], 'sha256': row['sha256']})
    assert len({r['relative_path'] for r in entries}) == 22
    entries.append({'relative_path': 'actor/global_step_5/training_state/manifest.json', 'bytes': len(raw), 'sha256': MANIFEST_SHA})
    need = sum(r['bytes'] for r in entries)
    fs = os.statvfs('/mnt/data')
    free = fs.f_bavail * fs.f_frsize
    assert free >= need + RESERVE, (free, need, RESERVE)
    return entries, need, free

def paced_read(path, target=None):
    start = time.monotonic()
    total = 0
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            if target is not None:
                target.write(block)
            digest.update(block)
            total += len(block)
            time.sleep(max(0, total / RATE - (time.monotonic() - start)))
    return total, digest.hexdigest()

def main():
    os.nice(19)
    result = {'utc_start': utc(), 'pid': os.getpid(), 'hostname': socket.gethostname(),
              'scope': 'ONLY_ACCEPTED_PARENT3340_CHECKPOINT5', 'all_experiment_backup_complete': False,
              'source': str(SOURCE), 'manifest_sha256': MANIFEST_SHA, 'bytes_per_second_limit': RATE}
    try:
        entries, need, free = admit()
        assert (ROOT / 'INTENT.json').is_file()
        staging = ROOT / 'checkpoint5.partial'
        staging.mkdir()
        copied = []
        for row in entries:
            src = SOURCE.parent.parent / row['relative_path']
            dst = staging / row['relative_path']
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open('xb') as f:
                n, sha = paced_read(src, f)
                f.flush()
                os.fsync(f.fileno())
            assert n == row['bytes'] and sha == row['sha256'], row['relative_path']
            n2, sha2 = paced_read(dst)
            assert n2 == row['bytes'] and sha2 == row['sha256'], row['relative_path']
            copied.append(dict(row, destination_readback_sha256=sha2))
            record('PROGRESS.json', {'utc': utc(), 'pid': os.getpid(), 'completed_files': len(copied),
                                    'completed_bytes': sum(r['bytes'] for r in copied), 'expected_bytes': need,
                                    'expected_files': len(entries), 'last_file': row['relative_path']})
        # Keep source manifest bytes and absolute provenance intact; publish the mirror atomically.
        assert not (ROOT / 'checkpoint5').exists()
        os.rename(staging, ROOT / 'checkpoint5')
        result.update(status='CHECKPOINT5_INDEPENDENT_COPY_FULL_SHA_PASS', utc_end=utc(),
                      destination=str(ROOT / 'checkpoint5'), bytes=need, files=copied,
                      admitted_free_bytes=free, independent_destination_mount='h100-10:/mnt/data ext4',
                      semantic_reload_test_performed=False)
        record('RESULT.json', result)
    except BaseException as e:
        result.update(status='FAILED_RETAIN_PARTIAL_AND_SOURCE', utc_end=utc(), error=repr(e))
        record('RESULT.json', result)
        raise

if __name__ == '__main__':
    main()
