"""Fixed Mac-only reclamation of 76 SHA-proven loose duplicate files.

Original archives, historical receipts, directories and failed partials survive.
This is separate from the USTAR-only compact checkpoint queue.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
PREFIX = 'experiments/search_e1/control/opsd_backup_duplicate_reclaim_v1/'
ANALYSIS = Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output')
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
SCOPES = (
    dict(name='protected-0-1-5-11', root=str(ANALYSIS/'opsd-online-launch-20261008/checkpoint-backup/protected-0-1-5-11-20261008'),
         receipt_sha='0169e6b4f9be5429e3fb0d39211cec16cf6a234e208eb32f6af6ad9a6718db65',
         inventory='node-member-inventory.json', inventory_sha='3c99d9a41228665117664e515dc2725de1008363e7319aedf0721454c818ae13',
         archive='protected-checkpoints.tar', archive_sha='0646f2da8b338108f753f880b30859a96831681cc99767cc6c47a50da93e9dd1',
         archive_bytes=6239805440, member_count=58, payload_bytes=6239760772, format='GNU'),
    dict(name='CP89', root=str(ANALYSIS/'opsd-ablation-execution-20261008/cp89-resume-871dc46-20261009'),
         receipt_sha='a7ac51348ad52a4dad8762b08c93192ee5cbfa96a205512135265bcc35b2d3dc',
         inventory='fresh-inventory.json', inventory_sha='0b5b269ee8e818de004b3ba262e994180da45ce4adb6da94e05341052043f28c',
         archive='checkpoint.tar', archive_sha='6f25944be9ce8a6936f422220fe2fe2179812583212ed5cc4c3fbe1416537cb9',
         archive_bytes=2041190400, member_count=18, payload_bytes=2041167791, format='USTAR'),
)
META = ('path', 'device', 'inode', 'bytes', 'mode', 'uid', 'nlink', 'mtime_ns', 'ctime_ns')


def require(test, message):
    if not test:
        raise RuntimeError(message)


def checked_path(path, directory=False):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts, 'Absolute canonical path required')
    for part in [*reversed(p.parents), p]:
        s = part.lstat()
        require(not stat.S_ISLNK(s.st_mode), 'Symlink component refused: '+str(part))
        require(stat.S_ISDIR(s.st_mode) if part != p or directory else stat.S_ISREG(s.st_mode),
                'Nonregular path refused: '+str(part))
    return p


def metadata(path):
    p = checked_path(path); s = p.stat()
    require(s.st_nlink == 1 and s.st_uid == os.getuid(), 'Hardlink/foreign owner refused: '+str(p))
    return dict(path=str(p), device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mode=s.st_mode,
                uid=s.st_uid, nlink=s.st_nlink, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def stable(ref):
    require(metadata(ref['path']) == {k: ref[k] for k in META}, 'File metadata changed: '+ref['path'])


def fileproof(path):
    before = metadata(path); h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        require(f.fileno() >= 0 and os.fstat(f.fileno()).st_ino == before['inode'], 'Handle differs')
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    stable(before)
    return dict(**before, sha256=h.hexdigest())


def checked_json(path, sha):
    before = metadata(path); require(before['bytes'] <= 1 << 20, 'Oversized metadata JSON refused')
    raw = Path(path).read_bytes(); stable(before)
    require(hashlib.sha256(raw).hexdigest() == sha, 'JSON SHA differs: '+str(path))
    return json.loads(raw), dict(**before, sha256=sha)


def fsync_dir(path):
    fd = os.open(checked_path(path, directory=True), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    fsync_dir(Path(path).parent)
    return fileproof(path)


def safe_name(name):
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and '..' not in p.parts and str(p) == name
            and '\x00' not in name and '\\' not in name, 'Noncanonical member name refused')
    return name


def extraction_set(root):
    checked_path(root, directory=True); result = set()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            checked_path(Path(parent)/name, directory=True)
        for name in files:
            result.add(str(checked_path(Path(parent)/name)))
    return result


def no_open_files(paths):
    # Inspection itself keeps no loose-file handles open during this check.
    require(shutil.which('lsof') is not None, 'lsof capability missing')
    if not paths:
        return
    r = subprocess.run(['lsof', '-nP', '-F', 'pfn', *paths], capture_output=True, text=True)
    require(r.returncode == 1 and not r.stdout.strip() and not r.stderr.strip(),
            'Loose file is open or lsof inspection uncertain; nothing reclaimed')


def publication(args):
    m, mp = checked_json(HERE/'source-manifest.json', args.source_manifest_sha256)
    require(set(p.name for p in HERE.iterdir()) == set(m['files'])|{'source-manifest.json'}, 'Source inventory differs')
    sources = [fileproof(HERE/n) for n in m['files']]
    for row in sources:
        require({k: row[k] for k in ('bytes', 'sha256')} == m['files'][Path(row['path']).name], 'Source bytes differ')
    p, pp = checked_json(args.publication, args.publication_sha256)
    require(p.get('accepted') is True and p.get('branch') == 'search-e1'
            and p.get('source_commit') == p.get('remote_readback_sha') == args.source_commit
            and re.fullmatch('[0-9a-f]{40}', args.source_commit)
            and p.get('source_manifest_sha256') == mp['sha256'], 'Publication differs')
    require(p.get('run_id') == RUN or RUN in p.get('run_ids', {}).values(), 'Old run publication binding missing')
    for row in sources+[mp]:
        hits = [r for r in p['files'] if r['path'] == PREFIX+Path(row['path']).name]
        require(len(hits) == 1 and hits[0].get('equal') is True
                and all(hits[0][k] == row[k] for k in ('bytes', 'sha256')), 'Remote byte readback differs')
    return dict(manifest=mp, publication=pp, source_commit=args.source_commit, sources=sources)


def inventory(scope):
    root = checked_path(scope['root'], directory=True)
    receipt, rp = checked_json(root/'complete-sha-verified.json', scope['receipt_sha'])
    inv, ip = checked_json(root/scope['inventory'], scope['inventory_sha'])
    require(receipt.get('accepted') is True and inv.get('accepted') is True
            and receipt['run_id'] == inv['run_id'] == RUN, 'Historical acceptance/run differs')
    rows = {safe_name(r['relative']): r for r in inv['files']}
    require(len(rows) == len(inv['files']) == scope['member_count']
            and set(rows) == set(receipt['files'])
            and sum(r['bytes'] for r in rows.values()) == scope['payload_bytes'], 'Historical member scope differs')
    for name, row in rows.items():
        old = receipt['files'][name]
        require(old['sha256'] == row['sha256'] and old['bytes'] == row['bytes']
                and Path(old.get('local_path', old.get('path'))) == root/'verified-files'/name,
                'Historical loose identity differs')
    ar = receipt['archive']
    require((ar['path'] if isinstance(ar, dict) else ar) == str(root/scope['archive']), 'Archive path differs')
    require((ar['sha256'] if isinstance(ar, dict) else receipt['archive_sha256']) == scope['archive_sha']
            and (ar['bytes'] if isinstance(ar, dict) else receipt['archive_bytes']) == scope['archive_bytes'], 'Archive identity differs')
    return rows, [rp, ip]


def verify_archive(scope, expected):
    """One exact raw pass: whole SHA, canonical original-format member SHAs."""
    path = Path(scope['root'])/scope['archive']; before = metadata(path)
    require(before['bytes'] == scope['archive_bytes'], 'Archive size differs')
    whole = hashlib.sha256(); count = 0; seen = {}
    def read(f, n):
        nonlocal count
        value = f.read(n); count += len(value); whole.update(value)
        require(count <= before['bytes'], 'Archive exceeds fixed bound')
        return value
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        require(os.fstat(f.fileno()).st_ino == before['inode'], 'Archive handle differs')
        while True:
            h = read(f, 512); require(len(h) == 512, 'Truncated tar header')
            if h == bytes(512):
                require(read(f, 512) == bytes(512), 'Two zero tar end blocks required')
                for tail in iter(lambda: read(f, 4 << 20), b''):
                    require(not any(tail), 'Trailing archive/garbage refused')
                break
            magic = b'ustar  \x00' if scope['format'] == 'GNU' else b'ustar\x0000'
            require(h[257:265] == magic, 'Original tar format differs')
            member = tarfile.TarInfo.frombuf(h, 'utf-8', 'strict'); name = safe_name(member.name)
            require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.pax_headers
                    and name in expected and name not in seen and member.size == expected[name]['bytes'],
                    'Links/extensions/special/extra/duplicate/member size refused')
            digest = hashlib.sha256(); remaining = member.size
            while remaining:
                data = read(f, min(4 << 20, remaining)); require(data, 'Truncated member')
                remaining -= len(data); digest.update(data)
            require(digest.hexdigest() == expected[name]['sha256'], 'Archive member SHA differs')
            require(read(f, (-member.size) % 512) == bytes((-member.size) % 512), 'Member padding differs')
            seen[name] = dict(bytes=member.size, sha256=digest.hexdigest())
    stable(before)
    require(count == before['bytes'] and count % 512 == 0 and whole.hexdigest() == scope['archive_sha']
            and set(seen) == set(expected), 'Whole archive SHA/size/member set differs')
    return dict(**before, sha256=whole.hexdigest()), seen


def prepare(output, source):
    output = Path(output); checked_path(output.parent, directory=True)
    require(output.is_absolute() and not output.exists(), 'Fresh absolute plan directory required')
    for scope in SCOPES:
        require(Path(scope['root']) not in output.parents and output not in Path(scope['root']).parents,
                'Plan directory must be independent of backup packages')
    packages = []; files = []
    for scope in SCOPES:
        rows, refs = inventory(scope); extract = Path(scope['root'])/'verified-files'
        expected_paths = {str(extract/n) for n in rows}
        require(extraction_set(extract) == expected_paths, 'Unexpected/missing loose files')
        no_open_files(sorted(expected_paths | {str(Path(scope['root'])/scope['archive'])}))
        archive, members = verify_archive(scope, rows)
        for name, row in sorted(rows.items()):
            fp = fileproof(extract/name)
            require(fp['bytes'] == row['bytes'] and fp['sha256'] == row['sha256'] == members[name]['sha256'],
                    'Loose/archive/historical member SHA differs')
            files.append(dict(scope=scope['name'], relative=name, **fp))
        packages.append(dict(scope=scope, preserve=[archive, *refs], member_count=len(rows)))
    no_open_files([r['path'] for r in files]+[p['preserve'][0]['path'] for p in packages])
    for package in packages:
        for ref in package['preserve']:
            stable(ref)
    for row in files:
        stable(row)
    output.mkdir(mode=0o700); fsync_dir(output.parent)
    plan = dict(schema='opsd-old-backup-loose-duplicate-plan-v1', accepted=True, run_id=RUN,
                created_unix=time.time(), source=source, packages=packages, files=files,
                logical_duplicate_bytes=sum(r['bytes'] for r in files),
                triple_sha_verified=True, preserve_archives_receipts_directories_partials_node=True,
                scope='Only named regular loose duplicates. No model/numerical/training/Node acceptance.')
    return write_json(output/'plan.json', plan)


def execute(plan_path, plan_sha, source):
    plan, pref = checked_json(plan_path, plan_sha); output = Path(plan_path).parent
    require(plan.get('schema') == 'opsd-old-backup-loose-duplicate-plan-v1' and plan.get('accepted') is True
            and plan['run_id'] == RUN and plan['source'] == source and plan['triple_sha_verified'] is True,
            'Immutable plan/source differs')
    require([p['scope'] for p in plan['packages']] == list(SCOPES), 'Unknown reclaim scope refused')
    expected = []
    for package in plan['packages']:
        rows, refs = inventory(package['scope'])
        require(package['preserve'][1:] == refs, 'Preserved historical refs differ')
        ar = package['preserve'][0]; scope = package['scope']
        require(ar['path'] == str(Path(scope['root'])/scope['archive'])
                and ar['sha256'] == scope['archive_sha'] and ar['bytes'] == scope['archive_bytes'], 'Preserved archive differs')
        for ref in package['preserve']:
            stable(ref)
        expected.extend((package['scope']['name'], name) for name in sorted(rows))
    require([(r['scope'], r['relative']) for r in plan['files']] == expected, 'Planned members differ')
    for row in plan['files']:
        scope = next(s for s in SCOPES if s['name'] == row['scope'])
        require(row['path'] == str(Path(scope['root'])/'verified-files'/safe_name(row['relative'])), 'Plan escape refused')
        rows, _ = inventory(scope)
        require(row['bytes'] == rows[row['relative']]['bytes'] and row['sha256'] == rows[row['relative']]['sha256'],
                'Plan member hash/size differs from original inventory')
    require(plan['logical_duplicate_bytes'] == sum(r['bytes'] for r in plan['files']), 'Plan total differs')
    intent = output/'execution-intent.json'; journal = output/'journal'
    if intent.exists():
        saved, _ = checked_json(intent, fileproof(intent)['sha256'])
        require(saved == dict(plan=pref, source=source, files=plan['files']), 'Existing durable intent differs')
        checked_path(journal, directory=True)
    else:
        for row in plan['files']:
            stable(row)
        no_open_files([r['path'] for r in plan['files']]+[p['preserve'][0]['path'] for p in plan['packages']])
        journal.mkdir(mode=0o700); fsync_dir(output)
        write_json(intent, dict(plan=pref, source=source, files=plan['files']))
    for package in plan['packages']:
        actual = extraction_set(Path(package['scope']['root'])/'verified-files')
        wanted = {r['path'] for r in plan['files'] if r['scope'] == package['scope']['name'] and Path(r['path']).exists()}
        require(actual == wanted, 'Foreign extraction file refused before unlink')
    outcomes = []
    for index, row in enumerate(plan['files']):
        for package in plan['packages']:
            for ref in package['preserve']:
                stable(ref)
        pending = journal/('%03d-pending.json' % index); done = journal/('%03d-done.json' % index)
        intended = dict(plan=pref, index=index, file=row)
        if done.exists():
            old, _ = checked_json(done, fileproof(done)['sha256'])
            require(old['intent'] == intended and not Path(row['path']).exists(), 'Completed member reappeared/differs')
            outcomes.append(old); continue
        if pending.exists():
            old, _ = checked_json(pending, fileproof(pending)['sha256'])
            require(old == intended, 'Pending file identity differs')
        else:
            stable(row); no_open_files([row['path']]); write_json(pending, intended)
        path = Path(row['path'])
        if path.exists():
            stable(row); no_open_files([row['path']]+[p['preserve'][0]['path'] for p in plan['packages']])
            # Fixed parent handle prevents a later parent-symlink traversal.
            parent = checked_path(path.parent, directory=True)
            fd = os.open(parent, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                ps = os.fstat(fd); now = parent.stat()
                require((ps.st_dev, ps.st_ino) == (now.st_dev, now.st_ino), 'Unlink parent handle changed')
                s = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                require((s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_uid, s.st_nlink, s.st_mtime_ns, s.st_ctime_ns)
                        == tuple(row[k] for k in ('device','inode','bytes','mode','uid','nlink','mtime_ns','ctime_ns')),
                        'Unlink exact file identity changed')
                stable(row); os.unlink(path.name, dir_fd=fd); os.fsync(fd)
            finally:
                os.close(fd)
            outcome = 'exact_planned_file_unlinked'
        else:
            outcome = 'absent_after_durable_pending_actual_unlink_not_inferred'
        value = dict(intent=intended, outcome=outcome, observed_unix=time.time())
        write_json(done, value); outcomes.append(value)
    for package in plan['packages']:
        require(not extraction_set(Path(package['scope']['root'])/'verified-files'), 'Unexpected final residual file')
        for ref in package['preserve']:
            stable(ref)
    stable(pref)
    receipt = output/'reclaim-complete.json'
    if receipt.exists():
        value, ref = checked_json(receipt, fileproof(receipt)['sha256'])
        require(value['plan'] == pref and value['journal'] == outcomes, 'Completion differs')
        return ref
    return write_json(receipt, dict(accepted=True, plan=pref, source=source, journal=outcomes,
        logical_duplicate_bytes=plan['logical_duplicate_bytes'], archive_receipt_node_and_directories_preserved=True,
        actual_apfs_reclaimed_bytes_not_measured=True, scope='Loose duplicate reclamation; original exact tar retained'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'execute'])
    p.add_argument('--source-manifest-sha256', required=True); p.add_argument('--publication', required=True)
    p.add_argument('--publication-sha256', required=True); p.add_argument('--source-commit', required=True)
    p.add_argument('--output'); p.add_argument('--plan'); p.add_argument('--plan-sha256')
    args = p.parse_args()
    require(sys.platform == 'darwin' and os.getuid() == os.geteuid() == 501, 'Only actual Mac owner501 allowed')
    source = publication(args)
    locks = []
    try:
        for scope in SCOPES:
            root = checked_path(scope['root'], directory=True); path = root/'.loose-duplicate-reclaim.lock'
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            require(os.fstat(fd).st_nlink == 1, 'Lease hardlink refused')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(fd)
        if args.mode == 'prepare':
            require(args.output and not args.plan and not args.plan_sha256, 'prepare requires only --output')
            result = prepare(args.output, source)
        else:
            require(args.plan and args.plan_sha256 and not args.output, 'execute requires only --plan/pathSHA')
            result = execute(args.plan, args.plan_sha256, source)
        print(json.dumps(dict(accepted=True, mode=args.mode, result=result)))
    finally:
        for fd in reversed(locks):
            os.close(fd)


if __name__ == '__main__':
    main()
