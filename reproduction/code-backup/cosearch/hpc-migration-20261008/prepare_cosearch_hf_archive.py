"""Prepare the explicitly selected immutable supplement on H100; no local staging."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    source, root = map(Path, sys.argv[1:3])
    lock = (root / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def save(name, obj):
        temp = root / (name + '.tmp')
        temp.write_text(json.dumps(obj, indent=2) + '\n')
        os.replace(str(temp), str(root / name))
    rows = [json.loads(x) for x in (root / 'manifest.jsonl').read_text().splitlines()]
    selected = [x for x in rows if x['kind'] in ('file', 'symlink')]
    assert len({x['path'] for x in selected}) == len(selected)
    for x in selected:
        p = Path(x['path'])
        assert not p.is_absolute() and '..' not in p.parts
    files = [x for x in selected if x['kind'] == 'file']
    assert rows[-1]['kind'] == 'complete' and rows[-1]['files'] == len(files)
    final = root / 'retriever-supplement.tar.zst'
    receipt = root / 'archive-verified.json'
    manifest_sha = digest(root / 'manifest.jsonl')
    if final.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        assert old['manifest_sha256'] == manifest_sha
        assert old['archive_bytes'] == final.stat().st_size
        assert old['archive_sha256'] == digest(final)
        return
    save('prepare-progress.json', {'pid': os.getpid(), 'status': 'VERIFYING_SOURCE', 'epoch': time.time()})
    count = 0
    for x in selected:
        p = source / x['path']
        if x['kind'] == 'symlink':
            assert p.is_symlink() and os.readlink(str(p)) == x['target'], x['path']
        else:
            assert not p.is_symlink() and p.stat().st_size == x['bytes'], x['path']
            assert digest(p) == x['sha256'], x['path']
            count += 1
        if count % 1000 == 0:
            save('prepare-progress.json', {'pid': os.getpid(), 'status': 'VERIFYING_SOURCE', 'files': count, 'epoch': time.time()})
    save('prepare-progress.json', {'pid': os.getpid(), 'status': 'ARCHIVING', 'files': count, 'epoch': time.time()})
    names = b''.join(x['path'].encode() + b'\0' for x in selected)
    part = root / 'retriever-supplement.tar.zst.part'
    with part.open('wb') as out:
        tar = subprocess.Popen(['tar', '-C', str(source), '--null', '--verbatim-files-from', '-T', '-', '-cf', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        zstd = subprocess.Popen(['zstd', '-T2', '-1', '-c'], stdin=tar.stdout, stdout=out)
        tar.stdout.close()
        tar.stdin.write(names)
        tar.stdin.close()
        assert tar.wait() == 0 and zstd.wait() == 0
        out.flush()
        os.fsync(out.fileno())
    archive_sha = digest(part)
    os.replace(str(part), str(final))
    save('archive-verified.json', {'status': 'SOURCE_ARCHIVE_AND_FILES_SHA_VERIFIED', 'group': 'retriever-supplement', 'archive': str(final), 'archive_bytes': final.stat().st_size, 'archive_sha256': archive_sha, 'manifest_sha256': manifest_sha, 'files': len(files), 'original_bytes': sum(x['bytes'] for x in files), 'symlinks': len(selected) - len(files), 'epoch': time.time(), 'source_originals_retained': True})
    save('prepare-progress.json', {'pid': os.getpid(), 'status': 'SOURCE_ARCHIVE_READY', 'epoch': time.time()})


if __name__ == '__main__':
    main()
