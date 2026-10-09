"""Copy a minimal public CPython candidate into CoSearch; no frozen-env edits."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

BASE = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
SOURCE = Path('/share/anaconda3/envs/video_dit')
ROOT = BASE / 'runtime/retriever-python310-candidate'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    p = ROOT / name
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(q, p)


def ldd(p):
    r = subprocess.run(['ldd', str(p)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=15)
    assert r.returncode == 0 and 'not found' not in r.stdout, str(p)
    return r.stdout


def main():
    assert BASE.is_dir() and not BASE.is_symlink()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save('controller.json', {'pid': os.getpid(), 'epoch': time.time(),
         'source': str(SOURCE), 'target': str(ROOT), 'training_started': False})
    selected = {'bin/python3.10': SOURCE / 'bin/python3.10'}
    stdlib = SOURCE / 'lib/python3.10'
    for directory, dirs, files in os.walk(str(stdlib)):
        dirs[:] = [d for d in dirs if d not in ('site-packages', '__pycache__')]
        for name in files:
            p = Path(directory) / name
            if p.is_file():
                selected[str(p.relative_to(SOURCE))] = p
    candidates = [SOURCE / 'bin/python3.10'] + list((stdlib / 'lib-dynload').glob('*.so'))
    deps = set()
    for p in candidates:
        text = ldd(p)
        for name in re.findall(r'=> (/\S+)', text):
            q = Path(os.path.normpath(name))
            if str(q).startswith(str(SOURCE) + '/'):
                deps.add(q)
    # Native wheels may require these compiler runtimes too. Copy public files;
    # neither the public prefix nor any experiment environment is modified.
    deps.update(p for name in ('libstdc++.so.6', 'libgcc_s.so.1', 'libpython3.10.so.1.0')
                for p in [SOURCE / 'lib' / name] if p.is_file())
    for p in deps:
        selected[str(p.relative_to(SOURCE))] = p
    for rel, p in selected.items():
        assert '..' not in Path(rel).parts and not Path(rel).is_absolute()
        assert str(p.resolve()).startswith(str(SOURCE) + '/'), str(p)
    total = sum(p.stat().st_size for p in selected.values())
    assert total < 1024 ** 3, 'Unexpectedly large minimal runtime selection'
    rows = []
    for index, (rel, p) in enumerate(sorted(selected.items())):
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert str(dst.parent.resolve()).startswith(str(ROOT.resolve()) + '/')
        assert not dst.is_symlink()
        source_sha = digest(p)
        if not dst.exists() or digest(dst) != source_sha:
            tmp = dst.with_name(dst.name + '.prepare-' + uuid.uuid4().hex)
            shutil.copy2(str(p), str(tmp), follow_symlinks=True)
            assert digest(tmp) == source_sha
            os.replace(tmp, dst)
        rows.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': source_sha,
                     'source': str(p), 'source_resolved': str(p.resolve())})
        if index % 100 == 0:
            save('progress.json', {'status': 'COPYING_PUBLIC_RUNTIME', 'files': index,
                 'total_files': len(selected), 'epoch': time.time()})
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(LD_LIBRARY_PATH=str(ROOT / 'lib'), CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1')
    code = ('import sys,json,ssl,sqlite3,bz2,lzma,ctypes;'
            'assert sys.version_info[:3]==(3,10,16);'
            'assert sys.prefix==' + repr(str(ROOT)) + ';'
            'print(json.dumps({"version":sys.version,"prefix":sys.prefix,'
            '"openssl":ssl.OPENSSL_VERSION,"sqlite":sqlite3.sqlite_version}))')
    r = subprocess.run([str(ROOT / 'bin/python3.10'), '-I', '-B', '-c', code], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=30)
    if r.returncode:
        save('smoke-failure.json', {'status': 'STDLIB_SMOKE_FAILED', 'stderr': r.stderr[-2000:]})
        raise RuntimeError('Candidate runtime smoke failed')
    # Every selected destination file is reread, not merely counted as copied.
    for row in rows:
        p = ROOT / row['path']
        assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256']
    save('manifest.json', {'files': rows, 'source': str(SOURCE)})
    save('candidate-verified.json', {'status': 'OWN_CPYTHON_CANDIDATE_FILES_SHA_AND_STDLIB_PASS',
         'files': len(rows), 'bytes': total, 'smoke': json.loads(r.stdout),
         'epoch': time.time(), 'original_python_version': '3.10.12',
         'original_abi_dependency_retrieval_parity': 'PENDING', 'training_started': False})


if __name__ == '__main__':
    main()
