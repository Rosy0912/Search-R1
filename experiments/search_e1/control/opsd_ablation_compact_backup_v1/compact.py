"""Mac-only lossless pinned checkpoint compression; no Node mutation or signals."""
import ast
import fcntl
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
PREFIX = 'experiments/search_e1/control/opsd_ablation_compact_backup_v1/'
DICT_BYTES = 128 << 20
MEMLIMIT = 256 << 20


def legacy_modules(root, manifest_sha):
    """Compile the exact independently published V2 bytes, never timestamp pyc."""
    root = Path(root).absolute()
    raw = (root / 'source-manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha:
        raise RuntimeError('V2 source manifest SHA differs')
    manifest = json.loads(raw)
    if {p.name for p in root.iterdir()} != set(manifest['files']) | {'source-manifest.json'}:
        raise RuntimeError('V2 source inventory differs')
    snapshots = {}
    for name, row in manifest['files'].items():
        p = root / name
        if p.is_symlink() or not p.is_file():
            raise RuntimeError('V2 source is not regular')
        source = p.read_bytes()
        if len(source) != row['bytes'] or hashlib.sha256(source).hexdigest() != row['sha256']:
            raise RuntimeError('V2 source payload SHA differs: ' + name)
        snapshots[name] = source
    b = types.ModuleType('pinned_v2_backup')
    b.__file__ = str(root / 'backup.py')
    exec(compile(snapshots['backup.py'], b.__file__, 'exec'), b.__dict__)
    q = types.ModuleType('pinned_v2_queue')
    q.__file__ = str(root / 'queue.py')
    q.b = b
    tree = ast.parse(snapshots['queue.py'], q.__file__)
    # Replace only the published module's three loader statements with the exact
    # pre-hashed source module. Every reused function body remains unchanged.
    def loader(node):
        return (isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'b' for x in node.targets)
            or isinstance(node, ast.Assign) and any(isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
                and x.value.id == 'b' for x in node.targets)
            or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'exec')
    removed = [n for n in tree.body if loader(n)]
    if len(removed) != 3: raise RuntimeError('Published V2 loader shape changed')
    tree.body = [n for n in tree.body if not loader(n)]
    exec(compile(tree, q.__file__, 'exec'), q.__dict__)
    for name, row in manifest['files'].items():
        b.fileproof(root / name, dict(path=str(root / name), **row))
    b.checked_json(root / 'source-manifest.json', manifest_sha)
    return b, q


def publication(b, args, run):
    old = b.source_publication(Path(args.legacy_helper_root) / 'source-manifest.json',
        args.source_manifest_sha256, args.publication, args.publication_sha256,
        args.source_commit, run)
    manifest, mp = b.checked_json(HERE / 'source-manifest.json', args.compact_manifest_sha256)
    b.require({p.name for p in HERE.iterdir()} == set(manifest['files']) | {'source-manifest.json'},
              'Compact source inventory differs')
    proofs = [b.fileproof(HERE / n, dict(path=str(HERE / n), **r)) for n, r in manifest['files'].items()]
    pub, pp = b.checked_json(args.compact_publication, args.compact_publication_sha256)
    b.require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
        and pub.get('source_commit') == pub.get('remote_readback_sha') == args.compact_source_commit
        and re.fullmatch('[0-9a-f]{40}', args.compact_source_commit)
        and pub.get('source_manifest_sha256') == mp['sha256']
        and (pub.get('run_id') == run or run in pub.get('run_ids', {}).values()), 'Compact publication differs')
    for fp in proofs + [mp]:
        hits = [r for r in pub['files'] if r['path'] == PREFIX + Path(fp['path']).name]
        b.require(len(hits) == 1 and hits[0]['equal'] is True and
                  hits[0]['sha256'] == fp['sha256'] and hits[0]['bytes'] == fp['bytes'], 'Compact readback differs')
    for fp in proofs + [mp, pp]:
        b.stable(fp)
    return old, dict(manifest=mp, publication=pp, source_commit=args.compact_source_commit, files=proofs)


class QueueLease:
    """Exactly the old Mac queue lock; never a GPU/experiment lock."""
    def __init__(self, b, root):
        self.b = b
        self.path = b.checked_path(Path(root).absolute(), directory=True) / '.backup-queue.lock'
        self.before = b.metadata(self.path)
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            self.b.require(self.b.meta_from_stat(self.path, os.fstat(self.fd)) == self.before, 'Old queue lock inode changed')
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.b.stable(self.before)
        except BaseException:
            os.close(self.fd); self.fd = None
            raise
        return self

    def __exit__(self, *_):
        os.close(self.fd)


def raw_upper(inventory):
    return sum(512 + ((r['bytes'] + 511) // 512) * 512 for r in inventory['files']) + 10240


def compressed_ceiling(inventory):
    # This is an enforced writer ceiling, not an assumed measured compression ratio.
    size = raw_upper(inventory)
    return size + size // 100 + (1 << 20)


def write_private(path, data):
    with Path(path).open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())


def space_gate(b, destination, inventory, margin_gib=5):
    b.require(margin_gib >= 5, 'Five GiB margin is mandatory')
    need = compressed_ceiling(inventory) + (margin_gib << 30)
    b.require(shutil.disk_usage(destination).free >= need, 'Local space BACKLOG; no training dependency')
    return need


def compress_stream(b, source, destination, inventory):
    """Compress exact stream bytes; output never exceeds the pre-reserved ceiling."""
    h = hashlib.sha256(); count = 0; written = 0
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, check=lzma.CHECK_SHA256,
        filters=[dict(id=lzma.FILTER_LZMA2, dict_size=DICT_BYTES)])
    ceiling = compressed_ceiling(inventory)
    def emit(data):
        nonlocal written
        b.require(written + len(data) <= ceiling, 'XZ writer ceiling exceeded; partial retained')
        destination.write(data); written += len(data)
    for block in iter(lambda: source.read(4 << 20), b''):
        count += len(block); h.update(block)
        b.require(count <= raw_upper(inventory), 'Raw stream exceeds pinned USTAR bound')
        emit(compressor.compress(block))
    emit(compressor.flush())
    return dict(bytes=count, sha256=h.hexdigest(), compressed_bytes=written)


class XZReader:
    """Single XZ stream, bounded dictionary/output, no concatenated/trailing data."""
    def __init__(self, source, maximum):
        self.source = source; self.maximum = maximum; self.buffer = b''; self.done = False
        self.decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=MEMLIMIT)
        self.digest = hashlib.sha256(); self.total = 0

    def read(self, size):
        while len(self.buffer) < size and not self.done:
            data = self.source.read(64 << 10) if self.decoder.needs_input else b''
            if self.decoder.needs_input and not data:
                raise RuntimeError('Truncated XZ stream')
            output = self.decoder.decompress(data, max_length=4 << 20)
            self.total += len(output)
            if self.total > self.maximum:
                raise RuntimeError('Decompressed stream exceeds pinned upper bound')
            self.digest.update(output); self.buffer += output
            if self.decoder.eof:
                if self.decoder.unused_data or self.source.read(1):
                    raise RuntimeError('Concatenated/trailing XZ data refused')
                self.done = True
        value = self.buffer[:size]; self.buffer = self.buffer[size:]
        return value


def verify_xz(b, path, inventory, raw_identity):
    """Independent full disk read/decompression, whole-tar SHA and all member SHA."""
    expected = {r['relative']: r for r in inventory['files']}
    b.require(len(expected) == len(inventory['files']), 'Duplicate expected members')
    before = b.fileproof(path); seen = {}; offset = 0
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)), 'rb') as source:
        b.require(b.meta_from_stat(path, os.fstat(source.fileno())) == {k: before[k] for k in b.META}, 'XZ handle differs')
        stream = XZReader(source, raw_upper(inventory))
        while True:
            header = stream.read(512)
            b.require(len(header) == 512, 'Truncated USTAR header/end')
            if header == bytes(512):
                b.require(stream.read(512) == bytes(512), 'Two zero USTAR end blocks required')
                while True:
                    tail = stream.read(4 << 20)
                    if not tail: break
                    b.require(not any(tail), 'Trailing nonzero/second tar refused')
                break
            b.require(header[257:263] == b'ustar\0' and header[263:265] == b'00', 'Only original USTAR accepted')
            member = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
            name = b.safe_name(member.name)
            b.require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and member.isfile()
                and not member.pax_headers and name in expected and name not in seen,
                'Special/duplicate/extra/escaping member refused')
            row = expected[name]; b.require(member.size == row['bytes'], 'Member size differs')
            h = hashlib.sha256(); remaining = member.size
            while remaining:
                block = stream.read(min(4 << 20, remaining))
                b.require(block, 'Truncated member'); remaining -= len(block); h.update(block)
            b.require(h.hexdigest() == row['sha256'], 'Member content SHA differs')
            padding = (-member.size) % 512
            b.require(stream.read(padding) == bytes(padding), 'Noncanonical/truncated USTAR padding')
            seen[name] = dict(bytes=member.size, sha256=h.hexdigest())
            offset += 512 + member.size + padding
        b.require(stream.done and not stream.buffer and stream.total % 512 == 0 and stream.total >= offset + 1024,
                  'Truncated USTAR/XZ end')
        b.require(stream.total == raw_identity['bytes'] and stream.digest.hexdigest() == raw_identity['sha256'],
                  'Lossless original whole-tar identity differs')
        b.require(b.meta_from_stat(path, os.fstat(source.fileno())) == {k: before[k] for k in b.META}, 'XZ changed during read')
    b.require(set(seen) == set(expected), 'Missing member')
    b.stable(before)
    return dict(accepted=True, compressed=before, raw_tar=dict(raw_identity), members=seen,
                member_count=len(seen), total_payload_bytes=sum(r['bytes'] for r in seen.values()),
                independent_disk_decompression=True, no_extraction_required=True)


def validate_inventory(b, inv, job, args):
    return b.validate_inventory(inv, job['config']['path'], job['config']['sha256'],
        job['checkpoint']['path'], job['checkpoint']['sha256'], job['run_id'], job['optimizer_updates'],
        job['outer_index'], dict(path=args.remote_goal_identity, sha256=args.goal_identity_sha256), job['outer_complete'])


def save_complete(b, destination, inventory, ip, job, raw, verification, publications, legacy=None):
    final = destination / 'checkpoint.tar.xz'; os.rename(destination / 'checkpoint.tar.xz.partial', final); b.fsync_dir(destination)
    cp = b.fileproof(final, dict(path=str(final), sha256=verification['compressed']['sha256'], bytes=verification['compressed']['bytes']))
    value = dict(schema='opsd-ablation-compact-checkpoint-complete-v1', accepted=True, job=job,
        compressed=cp, raw_tar=raw, node_inventory=ip, goal_identity=inventory['goal_identity'],
        source_publication=publications, actual_transfer_returncode=0,
        independent_lossless_member_readback=True, member_verification=verification['members'],
        historical_v2=legacy, completed_unix=time.time(), node_outputs_retained=True,
        scope='Recoverable byte identity only; no numerical Adam/science/effect acceptance',
        format=dict(container='single-XZ', check='SHA256', dictionary_bytes=DICT_BYTES))
    return b.write_json(destination / 'compact-complete.json', value)


def import_v2(b, q, item, args, publications, destination):
    """Old proof is fresh before import; never relabel its post-reclaim metadata."""
    q.verified_complete(item)
    old, op = b.checked_json(item['complete_receipt']['path'], item['complete_receipt']['sha256'])
    inv, ip0 = b.checked_json(old['node_inventory']['path'], old['node_inventory']['sha256'])
    validate_inventory(b, inv, item['job'], args)
    b.checked_path(destination.parent, directory=True); b.require(not destination.exists(), 'New independent import destination required')
    destination.mkdir(mode=0o700); b.fsync_dir(destination.parent)
    ip = b.write_json(destination / 'node-inventory.json', inv)
    space_gate(b, destination, inv, args.margin_gib)
    ap = b.fileproof(old['archive']['path'], old['archive'])
    with Path(ap['path']).open('rb') as source, (destination / 'checkpoint.tar.xz.partial').open('xb') as target:
        b.require(b.meta_from_stat(ap['path'], os.fstat(source.fileno())) == {k: ap[k] for k in b.META}, 'Legacy tar handle differs')
        raw = compress_stream(b, source, target, inv); target.flush(); os.fsync(target.fileno())
    b.stable(ap); b.require(raw['sha256'] == ap['sha256'] and raw['bytes'] == ap['bytes'], 'Imported original tar differs')
    q.verified_complete(item)
    result = verify_xz(b, destination / 'checkpoint.tar.xz.partial', inv, raw)
    legacy = dict(complete_receipt=op, node_inventory=ip0, archive=ap, files=old['files'],
        metadata_scope='Fresh at import only. Old receipt stays immutable after explicitly recorded reclamation.')
    return save_complete(b, destination, inv, ip, item['job'], raw, result, publications, legacy)


def transfer(b, job, args, publications, destination):
    """Future remote stream is compressed directly, without raw tar/extract copies."""
    b.checked_path(destination.parent, directory=True); b.require(not destination.exists(), 'New transfer destination required')
    destination.mkdir(mode=0o700); b.fsync_dir(destination.parent)
    options = types.SimpleNamespace(**vars(args))
    for name, value in dict(config=job['config']['path'], config_sha256=job['config']['sha256'],
        checkpoint=job['checkpoint']['path'], checkpoint_sha256=job['checkpoint']['sha256'],
        expected_run=job['run_id'], expected_updates=job['optimizer_updates'], expected_outer=job['outer_index'],
        expected_completion=job['outer_complete']['path'], expected_completion_sha256=job['outer_complete']['sha256']).items(): setattr(options, name, value)
    result = subprocess.run(b.remote_command(options, 'inventory'), capture_output=True, check=False)
    write_private(destination / 'inventory.stderr', result.stderr)
    b.write_json(destination / 'inventory-execution.json', dict(actual_returncode=result.returncode,
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
    b.require(result.returncode == 0, 'Remote inventory failed')
    inv = json.loads(result.stdout); validate_inventory(b, inv, job, args)
    goal, _ = b.checked_json(args.goal_identity, args.goal_identity_sha256)
    b.require(inv['goal_identity']['value'] == goal and inv['source_publication']['manifest']['sha256'] == publications['legacy']['manifest']['sha256']
        and inv['source_publication']['publication']['sha256'] == publications['legacy']['publication']['sha256']
        and inv['source_publication']['source_commit'] == args.source_commit, 'Remote publication/goal differs')
    ip = b.write_json(destination / 'node-inventory.json', inv); space_gate(b, destination, inv, args.margin_gib)
    with (destination / 'checkpoint.tar.xz.partial').open('xb') as target, (destination / 'transfer.stderr').open('xb') as err:
        process = subprocess.Popen(b.remote_command(options, 'stream'), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        try:
            b.write_json(destination / 'transfer-child.json', dict(pid=process.pid, accepted=False, started_unix=time.time()))
            process.stdin.write(b.canonical(inv)); process.stdin.close()
            raw = compress_stream(b, process.stdout, target, inv)
        finally:
            try:
                if process.stdin and not process.stdin.closed: process.stdin.close()
            finally:
                try: process.stdout.close()
                finally: rc = process.wait()
            b.write_json(destination / 'transfer-wait.json', dict(actual_returncode=rc, ended_unix=time.time(), no_signal_sent=True))
        target.flush(); os.fsync(target.fileno()); err.flush(); os.fsync(err.fileno())
    b.require(rc == 0, 'Actual transfer nonzero; partial retained')
    verified = verify_xz(b, destination / 'checkpoint.tar.xz.partial', inv, raw)
    return save_complete(b, destination, inv, ip, job, raw, verified, publications)


def verified_compact(b, reference, job, args, full=False):
    value, _ = b.checked_json(reference['path'], reference['sha256'])
    b.require(value.get('schema') == 'opsd-ablation-compact-checkpoint-complete-v1' and value.get('accepted') is True
        and value['job'] == job and value['actual_transfer_returncode'] == 0
        and value['independent_lossless_member_readback'] is True, 'Compact completion differs from pinned job')
    inv, _ = b.checked_json(value['node_inventory']['path'], value['node_inventory']['sha256'])
    expected = validate_inventory(b, inv, job, args)
    b.require(value['member_verification'] == {n:dict(bytes=r['bytes'], sha256=r['sha256']) for n,r in expected.items()}, 'Compact member proof differs')
    bound = value['source_publication']
    b.require(bound['legacy']['manifest']['sha256'] == args.source_manifest_sha256
        and bound['legacy']['publication']['sha256'] == args.publication_sha256
        and bound['legacy']['source_commit'] == args.source_commit
        and bound['compact']['manifest']['sha256'] == args.compact_manifest_sha256
        and bound['compact']['publication']['sha256'] == args.compact_publication_sha256
        and bound['compact']['source_commit'] == args.compact_source_commit, 'Compact publication provenance differs')
    b.stable(value['compressed'])
    if full: verify_xz(b, value['compressed']['path'], inv, value['raw_tar'])
    return value


def reclaim_legacy(b, reference, job, args):
    """Explicit only, under the old queue lease, recoverable exact local duplicates."""
    value = verified_compact(b, reference, job, args, full=True)
    old = value['historical_v2']; b.require(old is not None, 'No legacy files belong to this compact package')
    original, _ = b.checked_json(old['complete_receipt']['path'], old['complete_receipt']['sha256'])
    b.require(original['archive'] == old['archive'] and original['files'] == old['files'], 'Historical legacy proof changed')
    package = Path(old['complete_receipt']['path']).parent
    queue_root = b.checked_path(Path(args.legacy_queue_root).absolute(), directory=True)
    b.require(package.parent == queue_root and re.fullmatch('[0-9a-f]{64}-attempt-[0-9a-f]{32}', package.name), 'Not an owned V2 package')
    extract = package / 'verified-files'; archive = package / 'checkpoint.tar'
    b.require(Path(old['archive']['path']) == archive, 'Legacy tar escapes package')
    inv, _ = b.checked_json(value['node_inventory']['path'], value['node_inventory']['sha256'])
    b.require(set(old['files']) == {r['relative'] for r in inv['files']}, 'Legacy extraction declaration differs')
    for name, fp in old['files'].items(): b.require(Path(fp['path']) == extract / b.safe_name(name), 'Legacy member escapes extraction')
    planned = [old['archive']] + list(old['files'].values())
    scope = Path(reference['path']).parent; plan_path = scope / 'legacy-reclaim-plan.json'
    plan = dict(schema='compact-legacy-local-reclaim-plan-v1', compact=reference, job=job, files=planned,
        preserve_original_receipt_inventory=True, node_files_touched=False)
    resumed = plan_path.exists()
    if resumed:
        saved, _ = b.checked_json(plan_path, b.fileproof(plan_path)['sha256']); b.require(saved == plan, 'Existing reclamation plan differs')
    else:
        # Inventory must be exact before the first mutation. No unrelated file is removed.
        actual = set()
        for parent, dirs, files in os.walk(extract, followlinks=False):
            for name in dirs: b.checked_path(Path(parent) / name, directory=True)
            for name in files: actual.add(str(b.checked_path(Path(parent) / name)))
        b.require(actual == {r['path'] for r in old['files'].values()}, 'Unexpected/missing extraction file before reclamation')
        for fp in planned: b.stable(fp)
        b.write_json(plan_path, plan)
    for fp in planned:
        path = Path(fp['path'])
        b.require(not path.is_symlink(), 'Legacy symlink replacement refused')
        if not path.exists(): b.require(resumed, 'Unrecorded missing legacy payload'); continue
        b.stable(fp); b.require(path.stat().st_nlink == 1, 'Legacy hardlink alias refused')
        path.unlink(); b.fsync_dir(path.parent)
    if extract.exists():
        for parent, dirs, files in os.walk(extract, topdown=False, followlinks=False):
            b.require(not files, 'Unexpected residual file; never remove it')
            for name in dirs: b.checked_path(Path(parent) / name, directory=True)
            Path(parent).rmdir(); b.fsync_dir(Path(parent).parent)
    done = scope / 'legacy-reclaimed.json'
    if not done.exists(): b.write_json(done, dict(accepted=True, plan=b.fileproof(plan_path),
        recovered_byte_identity=reference, logical_duplicate_bytes=sum(r['bytes'] for r in planned),
        missing_on_resume_is_recorded_intent_not_fabricated_unlink=True, historical_receipts_unchanged=True, node_files_touched=False))
    return b.fileproof(done)
