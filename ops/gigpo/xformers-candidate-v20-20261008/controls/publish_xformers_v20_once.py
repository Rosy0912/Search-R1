"""Persist the frozen candidate; never executes experiment adapters."""
import datetime, hashlib, json, os, pathlib, shutil, subprocess

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs/gigpo-e1-instruct200'
REPO = BASE / 'work/legacy3329-completion-publication-20261008'
PREFIX = 'ops/gigpo/xformers-candidate-v20-20261008'
RECEIPT = OUT / 'FRESH200-XFORMERS-V20-PUBLICATION-v1.json'
BRANCH = 'codex/gigpo-legacy3329-completion-20261008'
ENV = dict(os.environ, GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=10 -o "ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p"')

def run(args, cwd=REPO, **kw):
    return subprocess.run(args, cwd=cwd, env=ENV, check=True, capture_output=True, **kw).stdout

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    if RECEIPT.exists():
        raise RuntimeError('Publication receipt already exists; inspect instead of replaying')
    inventory_path = OUT / 'FRESH200-XFORMERS-V20-SHA-MANIFEST-v1.json'
    inventory = json.loads(inventory_path.read_bytes())
    assert inventory.get('status') and not run(['git', 'status', '--porcelain']).strip()
    assert run(['git', 'branch', '--show-current']).decode().strip() == BRANCH
    before = run(['git', 'rev-parse', 'HEAD']).decode().strip()
    rows = []
    inputs = inventory['runtime44'] + inventory['files']
    extra_paths = [inventory_path, pathlib.Path(__file__).resolve(),
        pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct200/FAILURES-AND-FIXES-20261008.md'),
        BASE / 'work/stop_job3338_human_xformers_once_v1.py',
        BASE / 'work/read_job3338_human_stop_terminal_v1.sh',
        BASE / 'work/read_gigpo_job3338_retained18_actual_template.sh',
        OUT / 'JOB3338-HUMAN-STOP-XFORMERS-LOCAL-INTENT-v1.json',
        OUT / 'JOB3338-HUMAN-STOP-XFORMERS-TRANSPORT-v1.json',
        OUT / 'RETAINED18-JOB3338-ACTUAL-STATUS-v11-human-stop-terminal.json',
        OUT / 'CROSS-SESSION-CUDA-FAULT-SOLUTIONS-20261008.md',
        OUT / 'FRESH200-XFORMERS-V20-GATE-TEMPLATE-v1.json',
        OUT / 'CONFIG-COMPATIBILITY.md',
        pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct200/REPORT.md')]
    inputs += [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in extra_paths]
    seen = set()
    for row in inputs:
        src = pathlib.Path(row['path'])
        raw = src.read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256'], str(src)
        assert src.name not in seen, src.name
        seen.add(src.name)
        rel = PREFIX + '/' + ('runtime/' if row in inventory['runtime44'] else 'controls/') + src.name
        dst = REPO / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists()
        dst.write_bytes(raw)
        rows.append(dict(path=rel, bytes=len(raw), sha256=sha(raw), original_source_path=str(src)))
    run(['git', 'add', '--', PREFIX])
    run(['git', 'commit', '-m', 'Apply XFormers workaround to fresh GiGPO training after retained CUDA failure'])
    commit = run(['git', 'rev-parse', 'HEAD']).decode().strip()
    run(['git', 'push', 'origin', 'HEAD:refs/heads/' + BRANCH], timeout=180)
    remote = run(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH], timeout=60).decode().split()[0]
    assert commit == remote
    fresh = BASE / 'work/xformers-v20-independent-git-readback.git'
    assert not fresh.exists()
    fresh.mkdir()
    run(['git', 'init', '--bare', str(fresh)])
    run(['git', 'remote', 'add', 'origin', 'git@github.com:shaoshitong/Search-R1.git'], cwd=fresh)
    run(['git', 'config', 'remote.origin.promisor', 'true'], cwd=fresh)
    run(['git', 'config', 'remote.origin.partialclonefilter', 'blob:none'], cwd=fresh)
    run(['git', 'fetch', '--depth=1', '--filter=blob:none', 'origin', commit], cwd=fresh, timeout=180)
    tree = run(['git', 'ls-tree', '-r', commit, '--', PREFIX], cwd=fresh).decode().splitlines()
    blobs = {line.split('\t', 1)[1]: line.split()[2] for line in tree}
    assert set(blobs) == {row['path'] for row in rows}
    objects = list(dict.fromkeys(blobs.values()))
    run(['git', 'fetch', '--no-tags', 'origin'] + objects, cwd=fresh, timeout=180)
    batch = run(['git', 'cat-file', '--batch'], cwd=fresh, input=('\n'.join(objects) + '\n').encode(), timeout=120)
    payloads = {}; offset = 0
    for oid in objects:
        newline = batch.index(b'\n', offset)
        header = batch[offset:newline].decode().split()
        assert header[0] == oid and header[1] == 'blob'
        size = int(header[2]); start = newline + 1
        payloads[oid] = batch[start:start + size]
        assert batch[start + size:start + size + 1] == b'\n'
        offset = start + size + 1
    assert offset == len(batch)
    for row in rows:
        raw = payloads[blobs[row['path']]]
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256'], row['path']
        row['downloaded_bytes_match'] = True
    receipt = dict(status='PUSHED_AND_INDEPENDENT_GIT_DOWNLOAD_VERIFIED',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), new_commit=commit,
        remote_sha=remote, branch=BRANCH, previous_publication_commit=before,
        source_commit='ebe91bfebd56c3bd445219dcd94342d815f899f0', manifest=rows,
        verification=dict(all_file_sha256_and_bytes_match=True, independent_bare_repository=str(fresh), no_alternates=True),
        execution=False, tests=False, static_review='HUMAN_WAIVED_STATIC_REVIEW')
    with RECEIPT.open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(dict(receipt=str(RECEIPT), commit=commit, files=len(rows), sha256=sha(RECEIPT.read_bytes()))))

if __name__ == '__main__':
    main()
