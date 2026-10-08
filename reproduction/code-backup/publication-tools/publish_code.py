#!/usr/bin/env python3
"""Preserve eight experiment implementations without changing their checkouts."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
STAMP = HERE.name.removeprefix('code-publication-')
REMOTE = 'git@github.com:shaoshitong/Search-R1.git'
BASE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02')
PREFIX = 'users-shaoshitong-documents-remote-codex-control'
EXTENSIONS = {'.py', '.sh', '.sbatch', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.patch', '.diff', '.requirements'}
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'site-packages', 'dist-packages', 'checkpoints', 'model', 'models', 'weights', 'logs', 'evidence', 'outputs', '.venv', 'venv', 'env'}
SECRET = re.compile(rb'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')

def run(args, cwd=None, data=None, check=True):
    p = subprocess.run(args, cwd=cwd, input=data, capture_output=True, env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))
    if check and p.returncode:
        raise RuntimeError(f'{args[0]} failed ({p.returncode}): {p.stderr.decode(errors="replace")[-1800:]}')
    return p

def git(repo, *args, **kwargs):
    return run(['git', '-C', str(repo), *args], **kwargs)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def stable_bytes(path):
    for _ in range(3):
        a = path.stat(); raw = path.read_bytes(); b = path.stat()
        if (a.st_size, a.st_mtime_ns, a.st_ino) == (b.st_size, b.st_mtime_ns, b.st_ino):
            if SECRET.search(raw):
                raise RuntimeError(f'Credential pattern detected; publication stopped for {path}')
            return raw
    raise RuntimeError(f'File changed repeatedly during snapshot: {path}')

def source_config():
    configs = []
    for label, suffix, checkout, branch, prior in [
        ('gigpo', '', 'gigpo-e1-fresh200-cpufixturefix-20261007', 'codex/gigpo-e1-fresh200-cpufixturefix-20261007', '2262942142f9df8f1666ce3f1b9ebfe9c9239241'),
        ('tree-grpo', '-2', 'tree-current-chain-parent3331-resume-source', 'codex/tree-current-chain-parent3331-resume', '296217a4b676a26a92fc729f5d9882b785dbfa96'),
        ('bapo', '-3', 'bapo-fresh200-controller-initfix-20261006', 'codex/bapo-fresh200-controller-initfix', '1cceda2f7e201de3f68c876bb9a3b895563c20ce'),
        ('opsd', '-4', 'opsd-unclipped-multitraj200-teacherrecompute', 'codex/opsd-unclipped-multitraj200-teacherrecompute', '188f4e5e7b63cf479290d0888a4a075499886750'),
        ('topd', '-5', 'topd-terminal-fixed413', None, '413a10c50f32febaafc83db8170e2b005e0e979f'),
        ('cosearch', '-6', 'cosearch', None, 'ee5c83970af92ff494fa4f5ffb35215002f989dd'),
    ]:
        root = BASE / (PREFIX + suffix)
        configs.append(dict(label=label, root=str(root), repo=str(root/'work'/checkout), loose=str(root/'work'), source_branch=branch, prior=prior))
    configs += [
        dict(label='search-g1', root='/Users/shaoshitong/Documents/Search-G1', repo='/Users/shaoshitong/Documents/Search-G1/repo', loose='/Users/shaoshitong/Documents/Search-G1/preparation', source_branch=None, prior=None),
        dict(label='search-e1', root='/Users/shaoshitong/Documents/Codex/2026-10-03/xia', repo='/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish', loose='/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work', source_branch=None, prior=None),
    ]
    return configs

def loose_files(config):
    root = Path(config['loose'])
    for current, dirs, files in os.walk(root):
        p = Path(current)
        if (p/'.git').exists():
            dirs[:] = []; continue
        dirs[:] = [n for n in dirs if n not in SKIP_DIRS and not re.search(r'(?:^|[-_])(venv|env)(?:$|[-_])', n) and not (p/n).is_symlink()]
        if config['label'] == 'search-e1' and p == root:
            dirs[:] = [n for n in dirs if n.startswith(('search-e1', 'opsd-from-archive100', 'retriever-segmented'))]
        for name in sorted(files):
            f = p/name
            if f.is_symlink(): continue
            keep = f.suffix.lower() in EXTENSIONS or name in {'Dockerfile', 'Makefile', 'requirements.txt'}
            if f.suffix.lower() == '.json':
                keep = bool(re.search(r'(config|recipe|manifest|files|budget|inputs|binding)', name, re.I)) and f.stat().st_size <= 10*1024*1024
                if re.search(r'(auth|credential|private.*key|token)', name, re.I): keep = False
            if keep:
                yield f, f.relative_to(root)

def inspect_new_history(config, sha):
    repo = Path(config['repo']); prior = config.get('prior') or sha
    ids = git(repo, 'rev-list', '--objects', sha, '^'+prior).stdout.decode().splitlines()
    seen = set(); blobs = []
    for line in ids:
        oid = line.split(' ', 1)[0]
        if oid in seen: continue
        seen.add(oid)
        kind = git(repo, 'cat-file', '-t', oid).stdout.strip()
        if kind == b'blob': blobs.append(oid)
    for oid in blobs:
        raw = git(repo, 'cat-file', 'blob', oid).stdout
        if SECRET.search(raw): raise RuntimeError('Unpublished history contains a credential pattern; stopped before push')
    return len(blobs)

def verify_files(repo, commit, files):
    p = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for f in files:
            p.stdin.write((commit+':'+f['destination']+'\n').encode()); p.stdin.flush()
            h = p.stdout.readline().split()
            if len(h) != 3 or h[1] != b'blob': raise RuntimeError('Missing committed file: '+f['destination'])
            raw = p.stdout.read(int(h[2])); p.stdout.read(1)
            if digest(raw) != f['sha256']: raise RuntimeError('Committed content hash mismatch: '+f['destination'])
    finally:
        p.stdin.close(); p.wait()

def reconcile_prepared(config, target):
    label = config['label']; mf = target/'reproduction/code-backup'/label/'FILES-SHA256.json'
    meta = json.loads(mf.read_text())
    git(target, 'add', '--force', '--', 'reproduction/code-backup/'+label)
    if git(target, 'diff', '--cached', '--quiet', check=False).returncode:
        git(target, 'commit', '--quiet', '-m', 'Include exact runtime configurations excluded by inherited ignore rules')
    commit = git(target, 'rev-parse', 'HEAD').stdout.decode().strip()
    verify_files(target, commit, meta['files'])
    result = dict(label=label, repo=str(target), source_commit=meta['source_commit'], source_branch=meta['source_branch'], branch=meta['backup_branch'], commit=commit, snapshot_files=len(meta['files']), snapshot_bytes=sum(f['bytes'] for f in meta['files']), committed_content_sha256_verified=True, status='committed_pending_push', captured_utc=meta['captured_utc'])
    (HERE/(label+'-result.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result

def prepare(config):
    label = config['label']; original = Path(config['repo'])
    sha = git(original, 'rev-parse', 'HEAD').stdout.decode().strip()
    inspect_new_history(config, sha)
    target = HERE / label
    if target.exists(): return reconcile_prepared(config, target)
    run(['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout', str(original), str(target)])
    branch = 'codex/backup-'+label+'-latest-'+STAMP.lower()
    git(target, 'checkout', '--quiet', '-b', branch, sha)
    git(target, 'remote', 'set-url', 'origin', REMOTE)
    files = []
    def save(path, relative, role):
        raw = stable_bytes(path); dest = target/relative
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
        dest.chmod(path.stat().st_mode & 0o777)
        files.append(dict(source=str(path), destination=str(relative), bytes=len(raw), sha256=digest(raw), role=role))
    # Copy dirty text implementation and docs without staging anything in the owner's checkout.
    dirty = git(original, 'diff', '--name-only', '-z', 'HEAD').stdout.split(b'\0')
    untracked = git(original, 'ls-files', '--others', '--exclude-standard', '-z').stdout.split(b'\0')
    for name in sorted(set(n.decode() for n in dirty+untracked if n)):
        f = original/name
        if f.is_file() and not f.is_symlink() and f.suffix.lower() in EXTENSIONS|{'.md', '.json', '.txt'} and f.stat().st_size <= 10*1024*1024:
            save(f, Path(name), 'current_checkout_uncommitted')
    for f, relative in loose_files(config):
        save(f, Path('reproduction/code-backup')/label/'loose'/relative, 'loose_implementation_or_runtime_configuration')
    if label == 'topd':
        source = Path(config['root'])/'work/topd-instruct300-source-v1'
        for f in source.rglob('*'):
            if f.is_file() and not f.is_symlink() and f.suffix.lower() in EXTENSIONS:
                save(f, f.relative_to(source), 'active_topd_student_source_overlay')
    if label == 'cosearch':
        author = Path(config['root'])/'work/cosearch-author-review'
        paths = git(author, 'ls-tree', '-r', '--name-only', '763bf8c4d7caa363ad845d39ddfd53b81ae377bd').stdout.decode().splitlines()
        for name in paths:
            if Path(name).suffix.lower() in EXTENSIONS|{'.md', '.txt', '.json'}:
                raw = git(author, 'show', '763bf8c4d7caa363ad845d39ddfd53b81ae377bd:'+name).stdout
                if SECRET.search(raw): raise RuntimeError('Author source contains a credential pattern')
                dest = target/'reproduction/code-backup/cosearch/author_source'/name
                dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
                files.append(dict(source='author_commit:763bf8c4:'+name, destination=str(dest.relative_to(target)), bytes=len(raw), sha256=digest(raw), role='exact_author_native_source'))
    # Verify both current core and added loose files from Git objects, independent of worktree bytes.
    tracked = git(target, 'ls-files', '-z').stdout.split(b'\0')
    for n in tracked:
        if not n: continue
        f = target/n.decode()
        if f.is_file() and SECRET.search(f.read_bytes()): raise RuntimeError(f'Credential pattern in source snapshot: {n.decode()}')
    meta = dict(label=label, captured_utc=datetime.now(timezone.utc).isoformat(), original_repo=str(original), source_commit=sha, source_branch=config['source_branch'], backup_branch=branch, files=files, excludes='Model weights, datasets, environments, credentials, caches and bulk logs; known Git checkouts preserved by source commit. Historical/draft loose implementations retain original directory names and are not execution authorization.')
    mb = target/'reproduction/code-backup'/label; mb.mkdir(parents=True, exist_ok=True)
    (mb/'FILES-SHA256.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    (mb/'README.md').write_text(f'# {label} implementation backup\n\nCaptured {meta["captured_utc"]}. Core source commit `{sha}`.\n\nThe branch preserves core implementation plus loose runtime and configuration files with original paths and SHA256 in FILES-SHA256.json. Draft and historical directory names are retained. This is a source backup and does not certify that any job completed or authorize replaying submission scripts. No weights or credentials are included.\n')
    git(target, 'add', '--all')
    # These directories contain only the screened source/config snapshot. Preserve
    # files such as config/data/*.yaml despite inherited repository ignore rules.
    git(target, 'add', '--force', '--', 'reproduction/code-backup/'+label)
    # Preserve original source bytes, including pre-existing style warnings.
    whitespace = git(target, 'diff', '--cached', '--check', check=False)
    git(target, 'commit', '--quiet', '-m', 'Preserve latest '+label+' implementation and runtime source before node release')
    commit = git(target, 'rev-parse', 'HEAD').stdout.decode().strip()
    verify_files(target, commit, files)
    result = dict(label=label, repo=str(target), source_commit=sha, source_branch=config['source_branch'], branch=branch, commit=commit, snapshot_files=len(files), snapshot_bytes=sum(f['bytes'] for f in files), committed_content_sha256_verified=True, original_whitespace_warning_lines=len(whitespace.stdout.splitlines()), status='committed_pending_push', captured_utc=meta['captured_utc'])
    (HERE/(label+'-result.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result

def push(result):
    repo = Path(result['repo']); branch = result['branch']; sha = result['commit']
    git(repo, '-c', 'core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=15', 'push', 'origin', sha+':refs/heads/'+branch)
    remote = git(repo, '-c', 'core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=15', 'ls-remote', 'origin', 'refs/heads/'+branch).stdout.decode().strip().split()[0]
    if remote != sha: raise RuntimeError('Remote branch does not resolve to committed snapshot')
    result.update(status='pushed_remote_ref_verified', remote_commit=remote, verified_utc=datetime.now(timezone.utc).isoformat(), url='https://github.com/shaoshitong/Search-R1/tree/'+branch)
    (HERE/(result['label']+'-result.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--push', action='store_true'); args = parser.parse_args()
    if args.prepare:
        for config in source_config():
            if not (HERE/(config['label']+'-result.json')).exists(): prepare(config)
    if args.push:
        results = [json.loads(p.read_text()) for p in sorted(HERE.glob('*-result.json'))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(push, [r for r in results if r['status'] == 'committed_pending_push']))
