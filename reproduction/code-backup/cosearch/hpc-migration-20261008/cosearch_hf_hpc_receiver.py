"""Task-scoped HPC supplement receiver and safe, fresh original-file verifier."""
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

BASE = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
ROOT = BASE / 'source'
CONTROL = BASE / 'migration-hf-20261009'
SHA = 'edc951c6a9cade2134ca0badc58b03852aae800b73d55d498d1a4c3a95331787'
SIZE = 3483281305
MANIFEST_SHA = '968611a77c53c4782c6b1715650d49434bd2ea51ddc65adbd98405a44de383ca'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, p)


def destination(root, name):
    rel = PurePosixPath(name)
    assert name and not rel.is_absolute() and '..' not in rel.parts
    p = root
    assert not root.is_symlink()
    for part in rel.parts[:-1]:
        p = p / part
        assert not p.is_symlink(), name
        p.mkdir(exist_ok=True)
        assert p.is_dir(), name
    return p / rel.name


def check_file(p, row):
    assert not p.is_symlink() and p.is_file(), row['path']
    assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], row['path']


def extract_members(tar, root, rows):
    expected = {r['path']: r for r in rows if r['kind'] in ('file', 'symlink')}
    assert len(expected) == sum(r['kind'] in ('file', 'symlink') for r in rows)
    seen = set()
    root.mkdir(parents=True, exist_ok=True)
    for member in tar:
        assert member.name in expected and member.name not in seen, member.name
        row = expected[member.name]
        p = destination(root, member.name)
        if row['kind'] == 'symlink':
            assert member.issym() and member.linkname == row['target'], member.name
            # Frozen source links may be absolute; never traverse them here.
            tmp = p.with_name(p.name + '.hf-link-' + uuid.uuid4().hex)
            os.symlink(row['target'], tmp)
            os.replace(tmp, p)
        else:
            assert member.isfile() or member.islnk(), member.name
            fd, name = tempfile.mkstemp(prefix=p.name + '.hf-file-', dir=p.parent)
            os.close(fd)
            tmp = Path(name)
            if member.islnk():
                assert member.linkname in seen and expected[member.linkname]['kind'] == 'file'
                source = destination(root, member.linkname)
                check_file(source, expected[member.linkname])
                with source.open('rb') as inp, tmp.open('wb') as out:
                    shutil.copyfileobj(inp, out, 1024 * 1024)
                    out.flush()
                    os.fsync(out.fileno())
            else:
                assert member.size == row['bytes']
                with tar.extractfile(member) as inp, tmp.open('wb') as out:
                    shutil.copyfileobj(inp, out, 1024 * 1024)
                    out.flush()
                    os.fsync(out.fileno())
            check_file(tmp, row)
            os.chmod(tmp, member.mode & 0o777)
            os.replace(tmp, p)
        seen.add(member.name)
    assert seen == set(expected), 'Missing archive members'


def fresh_verify(root, rows):
    files = total = links = 0
    for row in rows:
        if row['kind'] not in ('file', 'symlink'):
            continue
        p = destination(root, row['path'])
        if row['kind'] == 'file':
            check_file(p, row)
            files += 1
            total += row['bytes']
        else:
            assert p.is_symlink() and os.readlink(p) == row['target']
            links += 1
    return files, total, links


def main():
    CONTROL.mkdir(exist_ok=True)
    lock = (CONTROL / 'supplement-target.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    part = CONTROL / ('retriever-supplement.' + SHA + '.part')
    final = CONTROL / 'retriever-supplement.tar.zst'
    action = sys.argv[1]
    if action == 'state':
        print(json.dumps({'offset': part.stat().st_size if part.exists() else 0,
                          'final_exists': final.exists(), 'epoch': time.time()}))
        return
    if action == 'receive':
        offset = int(sys.argv[2])
        assert not part.is_symlink() and 0 <= offset <= SIZE
        assert (part.stat().st_size if part.exists() else 0) == offset
        with part.open('ab', buffering=0) as out:
            while True:
                b = sys.stdin.buffer.read(1024 * 1024)
                if not b:
                    break
                assert offset + len(b) <= SIZE
                view = memoryview(b)
                while view:
                    n = os.write(out.fileno(), view)
                    assert n > 0
                    view = view[n:]
                offset += len(b)
            os.fsync(out.fileno())
        assert offset == SIZE
        return
    assert action == 'verify-extract'
    manifest = CONTROL / 'manifest.jsonl'
    assert digest(manifest) == MANIFEST_SHA
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert rows[-1]['kind'] == 'complete' and rows[-1]['files'] == 39714
    assert sum(r.get('bytes', 0) for r in rows if r['kind'] == 'file') == 7529241082
    archive = part if part.exists() else final
    assert not archive.is_symlink() and archive.stat().st_size == SIZE and digest(archive) == SHA
    if archive == part:
        os.replace(part, final)
    save(CONTROL / 'archive-verified.json', {'status': 'HPC3_ARCHIVE_SHA_VERIFIED',
         'archive_bytes': SIZE, 'archive_sha256': SHA, 'epoch': time.time()})
    z = subprocess.Popen(['zstd', '-dq', '-c', str(final)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=z.stdout, mode='r|') as tar:
            extract_members(tar, ROOT, rows)
        assert z.wait(timeout=30) == 0
    finally:
        if z.poll() is None:
            z.terminate()
            z.wait(timeout=15)
    files, total, links = fresh_verify(ROOT, rows)
    assert (files, total, links) == (39714, 7529241082, 4)
    result = {'status': 'SUPPLEMENT_ALL_ORIGINAL_FILES_FRESH_SHA_VERIFIED',
              'verified_files': files, 'verified_bytes': total, 'symlinks': links,
              'archive_sha256': SHA, 'fresh_original_files_verified': True,
              'epoch': time.time(), 'training_started': False}
    # Remote proof only; control-side acceptance must separately collect all hop receipts.
    save(CONTROL / 'target-fresh-verified.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
