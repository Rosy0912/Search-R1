"""Fixed software publication only. Root must compile these exact reviewed bytes.

No integrity runtime, GPU, training, recovery, signature or job submission runs.
An attempted publication has durable local intent; never remove or repeat it.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import sys
import types

W = Path(__file__).absolute().parents[1]
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
REPO = ROOT / 'repo-188f4e5e7b63'
TARGET_NAME = 'post-terminal-integrity-b8d8ff319cec'
PIN = 'b8d8ff319cece2d94634054961b2b3ee56ed7f5d9768568cc3bdfea7362e214d'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
TRANSPORT = 'a64bf52a97cbef6e3b0cc7948e095546687c8f1690094469c4937180f9449a85'
OVERLAP = '20c9fb1b720bd345f92b3cdfca0b4b19dc6d260e84c2d16429af497f8de45062'
SOURCE_BINDING = 'fbe330119f8fd07317345ef1ba0842d9374e65c6eb6592423ab101dba2841ad4'
PEER = '3574afe4742a83808394393dc48ebe178e757cd70fdb97d1c049b9b65766ad68'
MEMBERS = {
 'post_terminal_integrity.py': '90f7b3e9a8d7942fa2e44b5f8c76d8cd0104952f6718d5eea23acbbed4da69f4',
 'SOFTWARE_CANDIDATE_READINESS.json': 'e523bc3c1bedf9ea844ad1151a99cd1d6e515c77cb6f11a4f44ff9f4db90a3a4',
 'POST_TERMINAL_INTEGRITY_INDEPENDENT_SOFTWARE_REVIEW.json': 'ea1450711eb0734bb4b6f8efd7ae0bedcb24082570c05165665156b5fa12f80e',
 'POST_TERMINAL_SOURCE_BINDING_REUSE_READ_ONLY_PLAN.json': '40f8608305c63cfb2597925b0489ff55d5edd2ae63b5daa9b3c7a3e3a73ec8b5',
 'FINAL_COMPLETION_CONDITIONS_INDEPENDENT_REVIEW.json': '1d7da46d5bdf60b582e7b261333e5b7e9896ddd4a8dded7cbccbca1416ed5006'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def unique(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: need(False, 'Nonfinite JSON'))
    json.dumps(value, allow_nan=False)
    return value


def meta(value):
    return (value.st_dev, value.st_ino, value.st_uid, value.st_mode,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def directory_identity(value):
    return value.st_dev, value.st_ino, value.st_uid


def physical_fd(path, directory=False):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'Absolute physical path required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        flags = os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else 0)
        return os.open(path.name, flags, dir_fd=fd)
    finally:
        os.close(fd)


def read(path, pin):
    need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'Exact source/content pin required')
    fd = physical_fd(path)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
             and before.st_size <= 32 * 1024 * 1024, 'Own bounded regular bytes required')
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(fd)
        other = physical_fd(path)
        try:
            path_after = os.fstat(other)
        finally:
            os.close(other)
        need(meta(before) == meta(after) == meta(path_after)
             and len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == pin,
             'Exact bytes or FD/path identity changed')
        return raw
    finally:
        os.close(fd)


def compiled_module(name, path, pin):
    raw = read(path, pin)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def validate_packet(packet):
    need(type(packet) is dict and set(packet) == {'manifest_b64', 'members'}, 'Fixed packet schema required')
    mraw = base64.b64decode(packet['manifest_b64'], validate=True)
    need(hashlib.sha256(mraw).hexdigest() == PIN, 'Fixed b8d8 manifest differs')
    manifest = unique(mraw)
    need(type(manifest['version']) is int and manifest['version'] == 1
         and manifest['kind'] == 'POST_TRAINING_INTEGRITY_SOFTWARE_ONLY_IMMUTABLE_BUNDLE'
         and manifest['source_commit'] == SOURCE and manifest['runtime'] == 'post_terminal_integrity.py',
         'Fixed software bundle identity differs')
    need(type(packet['members']) is dict and set(packet['members']) == set(MEMBERS)
         and set(manifest['members']) == set(MEMBERS), 'Exactly five fixed members required')
    members = {}
    for name, pin in MEMBERS.items():
        raw = base64.b64decode(packet['members'][name], validate=True)
        rec = manifest['members'][name]
        need(set(rec) == {'bytes', 'sha256'} and type(rec['bytes']) is int
             and rec['bytes'] == len(raw) and rec['sha256'] == pin
             and hashlib.sha256(raw).hexdigest() == pin, 'Member exact bytes/schema differs')
        members[name] = raw
    return mraw, members


def prepare_packet():
    bundle = W / TARGET_NAME
    need(bundle.resolve(strict=True) == bundle and bundle.stat().st_uid == os.getuid()
         and stat.S_IMODE(bundle.stat().st_mode) == 0o555, 'Fixed immutable local bundle required')
    need({p.name for p in bundle.iterdir()} == set(MEMBERS) | {'manifest.json'}, 'Local bundle extra/missing member')
    peer = unique(read(W / 'POST_TERMINAL_B8D8_IMMUTABLE_PACKAGING_INDEPENDENT_REVIEW.json', PEER))
    need(peer['status'] == 'PASS_LOCAL_IMMUTABLE_PACKAGING_BYTES_MODES_AND_REFERENCE_BINDINGS_ONLY'
         and peer['manifest_sha256'] == PIN and peer['source_commit'] == SOURCE,
         'Exact independent packaging qualification required')
    mraw = read(bundle / 'manifest.json', PIN)
    packet = {'manifest_b64': base64.b64encode(mraw).decode(), 'members': {}}
    inodes = set()
    for name, pin in dict(MEMBERS, **{'manifest.json': PIN}).items():
        raw = read(bundle / name, pin)
        value = (bundle / name).stat()
        need(stat.S_IMODE(value.st_mode) == 0o444, 'Local immutable member mode differs')
        identity = value.st_dev, value.st_ino
        need(identity not in inodes, 'Immutable bundle member inode aliases')
        inodes.add(identity)
        if name != 'manifest.json':
            packet['members'][name] = base64.b64encode(raw).decode()
    validate_packet(packet)
    return packet


def source_anchor(c):
    before = c.allocation()
    raw = read(RUN / 'bindings/source.json', SOURCE_BINDING)
    source = unique(raw)
    need(source['source_commit'] == SOURCE and source['repo'] == str(REPO)
         and source['clean'] is True and source['formal_manifest_sha256'] == 'bb5f4872142f34a330abab14b5ae3869bc212c2e1eb7e247cf895f5aa3319d7c'
         and type(source['source_files']) is dict and len(source['source_files']) == 16,
         'Fixed source188 binding differs')
    files = {}
    for name, pin in source['source_files'].items():
        need(type(name) is str and not Path(name).is_absolute()
             and not any(v in ('', '.', '..') for v in name.split('/')),
             'Pinned source name escapes repository')
        value = read(REPO / name, pin)
        files[name] = {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
    return {'allocation': before, 'source_binding_sha256': SOURCE_BINDING, 'source_files': files}


def exclusive(path, raw):
    path = Path(path)
    parent = physical_fd(path.parent, True)
    try:
        parent_id = directory_identity(os.fstat(parent))
        need(parent_id[2] == os.getuid(), 'Own physical evidence parent required')
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            saved = meta(os.fstat(handle.fileno()))
        os.fsync(parent)
        need(read(path, hashlib.sha256(raw).hexdigest()) == raw and meta(path.stat()) == saved,
             'Original evidence bytes or identity changed')
    finally:
        os.close(parent)


def publish_files(parent, mraw, members):
    """Create only the fixed new child; this never runs a copied member."""
    parent = Path(parent)
    parent_fd = physical_fd(parent, True)
    child = parent / TARGET_NAME
    directory_fd = None
    try:
        parent_id = directory_identity(os.fstat(parent_fd))
        need(parent_id[2] == os.getuid(), 'Own physical tools parent required')
        os.mkdir(TARGET_NAME, 0o700, dir_fd=parent_fd)  # Existing entries/symlinks fail atomically.
        directory_fd = os.open(TARGET_NAME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        directory_id = directory_identity(os.fstat(directory_fd))
        published, inodes, saved_ids = [], set(), {}
        for name, raw in list(members.items()) + [('manifest.json', mraw)]:
            if name == 'manifest.json':
                os.fsync(directory_fd)  # Durably write all five members before the manifest.
            need(directory_identity(child.stat()) == directory_id, 'Target directory identity changed')
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444, dir_fd=directory_fd)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(raw)
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
                saved = os.fstat(handle.fileno())
            need(read(child / name, hashlib.sha256(raw).hexdigest()) == raw, 'Publication full readback differs')
            current = (child / name).stat()
            need(meta(saved) == meta(current), 'Published file inode/UID changed during readback')
            key = current.st_dev, current.st_ino
            need(key not in inodes, 'Published files alias')
            inodes.add(key)
            saved_ids[name] = meta(current)
            published.append({'name': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                              'device': current.st_dev, 'inode': current.st_ino, 'uid': current.st_uid, 'mode': '0o444'})
        need(published[-1]['name'] == 'manifest.json' and len(published) == 6
             and {p.name for p in child.iterdir()} == set(MEMBERS) | {'manifest.json'}, 'Fixed manifest-last inventory differs')
        os.fchmod(directory_fd, 0o555)
        os.fsync(directory_fd)
        os.fsync(parent_fd)
        check = physical_fd(parent, True)
        try:
            need(directory_identity(os.fstat(check)) == parent_id, 'Tools parent replaced')
        finally:
            os.close(check)
        need(directory_identity(child.stat()) == directory_id
             and stat.S_IMODE(child.stat().st_mode) == 0o555, 'Final physical target mode/identity differs')
        final_dir = physical_fd(child, True)
        try:
            need(directory_identity(os.fstat(final_dir)) == directory_id, 'Final target is no longer the same physical directory')
        finally:
            os.close(final_dir)
        for name, raw in dict(members, **{'manifest.json': mraw}).items():
            need(read(child / name, hashlib.sha256(raw).hexdigest()) == raw
                 and meta((child / name).stat()) == saved_ids[name], 'Closed publication member changed')
        return published
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def publish_packet(packet):
    need(sys.flags.optimize == 0 and os.getuid() == 0
         and socket.gethostname().split('.')[0] == 'h100-10', 'Only own live h100-10 publication allowed')
    mraw, members = validate_packet(packet)
    path = ROOT / 'tools/final-acceptance-f5c72c942641/checkpoint_overlap.py'
    c = compiled_module('exact_fixed_overlap_for_publication', path, OVERLAP)
    before = source_anchor(c)
    published = publish_files(ROOT / 'tools', mraw, members)
    after = source_anchor(c)
    need(before['source_files'] == after['source_files'], 'Source anchors changed across publication')
    for rec in published:
        path = ROOT / 'tools' / TARGET_NAME / rec['name']
        read(path, rec['sha256'])
        current = path.stat()
        need((current.st_dev, current.st_ino, current.st_uid) == (rec['device'], rec['inode'], rec['uid'])
             and stat.S_IMODE(current.st_mode) == 0o444, 'Publication member changed during final allocation read')
    return {'scope': 'NEW_FIXED_INTEGRITY_SOFTWARE_PUBLICATION_ONLY_NOT_ASSET_OR_TERMINAL_INSPECTION',
            'source_commit': SOURCE, 'manifest_sha256': PIN, 'directory': str(ROOT / 'tools' / TARGET_NAME),
            'manifest_last': True, 'published': published, 'directory_mode': '0o555',
            'allocation_source_before': before, 'allocation_source_after': after,
            'integrity_runtime_executed': False, 'training_runtime_modified': False}


def remote_command(raw, pin):
    need(hashlib.sha256(raw).hexdigest() == pin, 'Publisher source bytes differ before transport')
    encoded = base64.b64encode(raw).decode()
    wrapper = ('import base64,hashlib,json,sys,types\n'
       + 'raw=base64.b64decode(' + repr(encoded) + ',validate=True)\n'
       + 'if hashlib.sha256(raw).hexdigest()!=' + repr(pin) + ': raise ValueError("Publisher source SHA differs before compile")\n'
       + 'm=types.ModuleType("same_exact_reviewed_publisher_remote")\n'
       + 'm.__file__="<same exact reviewed publisher bytes carried by transport>"\n'
       + 'exec(compile(raw,m.__file__,"exec",dont_inherit=True),m.__dict__)\n'
       + 'print(json.dumps(m.publish_packet(m.unique(sys.stdin.buffer.read())),sort_keys=True))\n')
    return 'python3 -c ' + shlex.quote(wrapper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-reviewed', action='store_true')
    parser.add_argument('--publisher-sha256', required=True)
    args = parser.parse_args()
    need(args.deploy_reviewed, 'Root must choose the reviewed one-time publication explicitly')
    raw = read(Path(__file__).absolute(), args.publisher_sha256)
    packet = prepare_packet()
    transport = compiled_module('same_exact_approved_transport_a64', W / 'transport.py', TRANSPORT)
    command = remote_command(raw, args.publisher_sha256)
    out = W / 'post-terminal-integrity-deployment-b8d8ff319cec-originals'
    parent = physical_fd(W, True)
    try:
        need(os.fstat(parent).st_uid == os.getuid(), 'Own physical original-output directory required')
        os.mkdir(out.name, 0o700, dir_fd=parent)  # A second attempt is forbidden even after an unknown remote outcome.
        os.fsync(parent)
    finally:
        os.close(parent)
    intent = {'manifest_sha256': PIN, 'publisher_source_sha256': args.publisher_sha256,
              'packaging_review_sha256': PEER, 'transport_source_sha256': TRANSPORT,
              'actual_argv': sys.argv, 'remote_command_sha256': hashlib.sha256(command.encode()).hexdigest(),
              'unknown_outcome_rule': 'Inspect originals and exact remote target; never remove, overwrite, clean up or repeat this publisher'}
    exclusive(out / 'publish-once-intent.json', json.dumps(intent, sort_keys=True).encode())
    exclusive(out / 'remote-command-original.txt', command.encode())
    try:
        result = transport.remote(command, json.dumps(packet).encode(), timeout=180)
    except Exception as error:
        for name, attr in [('publish-failed-stdout.original', 'stdout'), ('publish-failed-stderr.original', 'stderr')]:
            value = getattr(error, attr, None)
            if isinstance(value, bytes):
                exclusive(out / name, value)
        exit_code = getattr(error, 'returncode', None)
        exclusive(out / 'publish-failure-or-unknown-original.json', json.dumps({'error': repr(error),
            'actual_returncode': exit_code if type(exit_code) is int else None,
            'remote_outcome_unknown': True, 'must_not_retry': True,
            'note': 'A known SSH transport exit is preserved, but partial remote publication still requires read-only inspection.'}, sort_keys=True).encode())
        raise
    exclusive(out / 'publish-stdout.original.json', result.stdout)
    exclusive(out / 'publish-stderr.original', result.stderr)
    exclusive(out / 'publish-exit-original.json', json.dumps({'actual_returncode': result.returncode}, sort_keys=True).encode())
    need(type(result.returncode) is int and result.returncode == 0, 'Real publication exit failed')
    observed = unique(result.stdout)
    need(observed['manifest_sha256'] == PIN and observed['manifest_last'] is True
         and observed['integrity_runtime_executed'] is False and len(observed['published']) == 6,
         'Real fixed publication result differs')
    print(json.dumps({'scope': observed['scope'], 'manifest_sha256': PIN,
                      'original_stdout_sha256': hashlib.sha256(result.stdout).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
