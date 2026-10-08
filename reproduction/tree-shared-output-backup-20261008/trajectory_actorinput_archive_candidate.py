#!/usr/bin/env python3
"""UNAPPROVED LOCAL CANDIDATE: lossless own3341 actor-input evidence archive.

Nothing has been deployed or executed on the training node. The default is a
metadata-only plan. Any non-plan call requires explicit human authorization in
the controlling conversation. This entry never removes/replaces an original.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008')
RUN = ROOT / 'runs/tree-current200-continue-3341'
TRAJECTORIES = RUN / 'output/trajectories'
ARCHIVES = RUN / '.trajectory-actorinput-archives-20261009'
CODE = 'ebef786697b78abafc2248c0c794827606c7214a'
CONFIG = '72502a1e48e757540f33fd848bacf681ba2a8796fc8b448e8b18d2f9cdf4ba47'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
CHUNK = 1024 * 1024
RATE = 8 * 1024 * 1024


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class OwnPaths:
    """Pin literal ROOT/RUN once; resolve every child via no-follow directory FDs."""
    def __init__(self):
        self.root_fd = self.run_fd = None
        current = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for name in ROOT.parts[1:]:
                following = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                os.close(current); current = following
            self.root_fd = current; current = None
            self.device = os.fstat(self.root_fd).st_dev
            self.run_fd = self._walk(self.root_fd, RUN.relative_to(ROOT).parts)
        except BaseException:
            if current is not None:
                os.close(current)
            self.close()
            raise

    def __enter__(self):
        return self

    def close(self):
        for name in ('run_fd', 'root_fd'):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd); setattr(self, name, None)

    def __exit__(self, *_):
        self.close()

    def _walk(self, base, parts):
        current = os.dup(base)
        try:
            for name in parts:
                require(name not in ('', '.', '..'), 'Invalid own path component')
                following = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                try:
                    require(os.fstat(following).st_dev == self.device, 'Own directory changed device')
                except BaseException:
                    os.close(following); raise
                os.close(current); current = following
            return current
        except BaseException:
            os.close(current); raise

    def parts(self, path):
        require(path.is_absolute() and path.is_relative_to(RUN), 'Path escaped exact own RUN namespace')
        parts = path.relative_to(RUN).parts
        require(all(x not in ('', '.', '..') for x in parts), 'Invalid own path component')
        return parts

    @contextmanager
    def parent(self, path):
        parts = self.parts(path)
        require(bool(parts), 'Leaf path required')
        fd = self._walk(self.run_fd, parts[:-1])
        try:
            yield fd, parts[-1]
        finally:
            os.close(fd)

    def open(self, path, flags, mode=0o600):
        with self.parent(path) as (parent, leaf):
            fd = os.open(leaf, flags | os.O_NOFOLLOW, mode, dir_fd=parent)
        try:
            s = os.fstat(fd)
            require(stat.S_ISREG(s.st_mode) and s.st_dev == self.device, 'Expected regular own-device file')
            return fd
        except BaseException:
            os.close(fd); raise

    def signature(self, path):
        fd = self.open(path, os.O_RDONLY)
        try:
            return signature_fd(fd)
        finally:
            os.close(fd)

    def mkdir(self, path, exist_ok=False):
        with self.parent(path) as (parent, leaf):
            try:
                os.mkdir(leaf, 0o700, dir_fd=parent)
            except FileExistsError:
                require(exist_ok, 'Existing archive/intent: inspect, never overwrite or duplicate')
            fd = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                require(os.fstat(fd).st_dev == self.device, 'Own directory changed device')
            finally:
                os.close(fd)
            os.fsync(parent)

    def write_json_new(self, path, value):
        with self.parent(path) as (parent, leaf):
            pending = '.' + leaf + '.pending-%d-%d' % (os.getpid(), time.time_ns())
            fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            with os.fdopen(fd, 'w') as f:
                json.dump(value, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
            os.link(pending, leaf, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            os.unlink(pending, dir_fd=parent)
            os.fsync(parent)

    def publish_archive(self, partial, final):
        require(partial.parent == final.parent, 'Archive publication must stay in one directory')
        with self.parent(partial) as (parent, leaf):
            s = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISREG(s.st_mode) and s.st_dev == self.device, 'Archive must be a regular own file')
            os.link(leaf, final.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            os.unlink(leaf, dir_fd=parent); os.fsync(parent)


def signature_fd(fd):
    s = os.fstat(fd)
    return dict(inode=s.st_ino, device=s.st_dev, bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def read_json(path, paths):
    fd = paths.open(path, os.O_RDONLY)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        require(before.st_size < 1024*1024, 'Bounded metadata required')
        raw = f.read(before.st_size+1)
        after = os.fstat(f.fileno())
        require(len(raw) == before.st_size and (after.st_size, after.st_mtime_ns, after.st_ino) == (before.st_size, before.st_mtime_ns, before.st_ino), 'Metadata changed during read')
    return json.loads(raw), raw


class LimitedReader:
    def __init__(self, stream):
        self.stream = stream
        self.start = time.monotonic()
        self.bytes = 0
        self.sha = hashlib.sha256()

    def read(self, length=CHUNK):
        block = self.stream.read(min(length, CHUNK))
        self.bytes += len(block); self.sha.update(block)
        delay = self.bytes / RATE - (time.monotonic() - self.start)
        if delay > 0:
            time.sleep(delay)
        return block


def hash_file(path, paths):
    fd = paths.open(path, os.O_RDONLY)
    with os.fdopen(fd, 'rb') as f:
        reader = LimitedReader(f)
        while reader.read():
            pass
    return reader.sha.hexdigest(), reader.bytes


def completed_directory(name, paths):
    match = re.fullmatch(r'outer-(\d{6})-attempt-(\d{6})', name)
    require(match is not None, 'Only exact existing outer/attempt trajectory names')
    outer, attempt = map(int, match.groups())
    directory = TRAJECTORIES / name
    manifest, raw = read_json(directory / 'manifest.json', paths)
    identity = manifest['identity']
    require(manifest['status'] == 'ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED', 'Trajectory payload publication incomplete')
    require(manifest['rows'] == 2560, 'Frozen five-leaf trajectory row count differs')
    require(identity == dict(outer_step=outer, attempt_id=attempt, job_id='3341', source=CODE, config=CONFIG, data=DATA), 'Trajectory identity mismatch')
    require(set(manifest['payloads']) == {'actor-inputs.pt', 'trajectories.jsonl.gz'}, 'Frozen trajectory format differs')
    require(all(re.fullmatch('[a-f0-9]{64}', v) for v in manifest['payloads'].values()), 'Original payload SHA missing')
    result, proof_raw = read_json(RUN / 'checkpoints/training_state/attempt-results' / ('attempt-%06d.json' % attempt), paths)
    require(result['status'] == 'COMPLETED_OPPORTUNITIES_OUTER_CREDITED' and result['accepted_outer'] == outer and result['attempt_id'] == attempt,
            'No credited completed attempt proof')
    ranks = result['all_rank_result']['ranks']
    gathered = result['all_rank_result']
    expected_identity = dict(job_id='3341', run_id=RUN.name, code_commit=CODE, config_sha256=CONFIG, data_identity=DATA, model_identity=MODEL)
    gathered_identity = gathered['identity']
    require(all(gathered_identity.get(k) == v for k, v in expected_identity.items()),
            'Completed attempt source/config identity differs')
    require(result['protocol'] == gathered['protocol'] == 'tree_fresh_actor_result_v1' and gathered['decision'] == 'COMPLETED_OUTER'
            and gathered['accepted_outer'] == outer and gathered['attempt_id'] == attempt and gathered['completed_outer_before'] == outer-1,
            'Completed attempt protocol/counters differ')
    require(result.get('automatic_rollback') is False, 'Rollback/refusal evidence is excluded')
    require(len(ranks) == 8 and sorted(r['rank'] for r in ranks) == list(range(8)), 'Attempt must prove eight ranks')
    require(all(r['scheduler_after'] == outer and r['accepted_outer'] == outer and r['attempt_id'] == attempt for r in ranks), 'Rank completion mismatch')
    direct_proofs = {}
    for row in ranks:
        descriptor = row['direct_rank_record']
        direct_path = RUN / 'output/fresh-state' / ('outer-%06d-attempt-%06d-rank%02d.json' % (outer, attempt, row['rank']))
        described = Path(descriptor['path'])
        alias = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
        if described.is_relative_to(alias):
            described = ROOT / described.relative_to(alias)
        require(described == direct_path, 'Direct-rank path differs from own exact outer')
        physical, direct_raw = read_json(direct_path, paths)
        require(digest(direct_raw) == descriptor['sha256'] and all(physical.get(k) == v for k, v in expected_identity.items()), 'Direct-rank SHA/source identity differs')
        require(physical['rank'] == row['rank'] and physical['world_size'] == 8 and physical['attempt_id'] == attempt
                and physical['observed_outer'] == outer and physical['scheduler_step_before'] == outer-1 and physical['scheduler_step_after'] == outer,
                'Physical direct-rank completion differs')
        direct_proofs[str(direct_path)] = digest(direct_raw)
    latest, latest_raw = read_json(RUN / 'checkpoints/training_state/rolling-completed/latest.json', paths)
    i = latest['identity']
    require(latest['status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED' and i['job_id'] == '3341' and i['code_commit'] == CODE and i['config_sha256'] == CONFIG and i['data_identity'] == DATA,
            'Latest complete native identity differs')
    native_path = Path(latest['manifest_path'])
    alias = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
    if native_path.is_relative_to(alias):
        native_path = ROOT / native_path.relative_to(alias)
    native, native_raw = read_json(native_path, paths)
    require(digest(native_raw) == latest['manifest_sha256'] and native['identity'] == i and native['status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED',
            'Latest pointer does not bind a complete native manifest')
    require(native_path == RUN / 'checkpoints/training_state/rolling-completed' / ('global_step_' + str(i['completed_step'])) / 'manifest.json', 'Latest complete native path differs')
    native_workers = native['worker_states']
    require(len(native_workers) == 8 and sorted(w['identity']['rank'] for w in native_workers) == list(range(8)), 'Native eight-rank coverage differs')
    require(all(all(w['identity'].get(k) == v for k,v in expected_identity.items()) and w['identity']['world_size'] == 8
                and w['identity']['completed_step'] == i['completed_step'] for w in native_workers), 'Native worker identities differ')
    require(outer <= i['completed_step'] - 2, 'Recent or running outer is excluded')
    return directory, manifest, dict(outer=outer, attempt=attempt, identity=identity,
           original_manifest_sha256=digest(raw), credited_attempt_proof_sha256=digest(proof_raw),
           direct_rank_proofs=direct_proofs, latest_complete_native_manifest_sha256=digest(native_raw),
           latest_complete_native_pointer_sha256=digest(latest_raw), latest_complete_outer_at_selection=i['completed_step'])


@contextmanager
def own_lock():
    require(socket.gethostname() == 'h100-10', 'Candidate can execute only on h100-10 after approval')
    with OwnPaths() as paths:
        # The path supplied to findmnt is our held RUN FD, never an untrusted
        # string ancestor. pass_fds keeps that FD alive in the child process.
        mount = subprocess.check_output(['findmnt', '-n', '-T', '/proc/self/fd/%d' % paths.run_fd,
                                         '-o', 'SOURCE,FSTYPE'], text=True, timeout=5, pass_fds=(paths.run_fd,)).strip()
        require(mount.split() == ['/dev/mapper/data-data', 'ext4'], 'Only the original same-machine ext4 is allowed')
        paths.mkdir(ARCHIVES, exist_ok=True)
        fd = paths.open(ARCHIVES / '.single-instance.lock', os.O_RDWR | os.O_CREAT)
        with os.fdopen(fd, 'a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
            subprocess.check_call(['ionice', '-c', '3', '-p', str(os.getpid())], timeout=10)
            yield paths


def archive_preserve(name, request, approval_reference):
    with own_lock() as paths:
        directory, manifest, proof = completed_directory(name, paths)
        source = directory / 'actor-inputs.pt'
        before = paths.signature(source)
        capacity = os.fstatvfs(paths.run_fd)
        require(capacity.f_bavail * capacity.f_frsize > before['bytes'] + 64 * 1024**3,
                'Keep 64GiB plus one worst-case source-sized archive reserve')
        target = ARCHIVES / name
        paths.mkdir(target)
        paths.write_json_new(target / 'intent.json', dict(request_id=request, approval_reference=approval_reference,
                       source=str(source), source_signature=before, scope='APPROVED_ARCHIVE_PRESERVE_ONLY', **proof))
        compressed = target / 'actor-inputs.pt.gz.partial'
        with os.fdopen(paths.open(source, os.O_RDONLY), 'rb') as raw, os.fdopen(paths.open(compressed, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'wb') as dest:
            require(signature_fd(raw.fileno()) == before, 'Source identity changed')
            reader = LimitedReader(raw)
            with gzip.GzipFile(filename='', fileobj=dest, mode='wb', compresslevel=1, mtime=0) as output:
                while True:
                    block = reader.read()
                    if not block:
                        break
                    output.write(block)
            dest.flush(); os.fsync(dest.fileno())
        require(paths.signature(source) == before, 'Source changed during archive; preserve all originals')
        require(reader.bytes == before['bytes'] and reader.sha.hexdigest() == manifest['payloads']['actor-inputs.pt'], 'Original source full SHA differs')
        compressed_sha, compressed_bytes = hash_file(compressed, paths)
        require(compressed_bytes < before['bytes'], 'Archive provides no savings; source and own partial retained')
        with os.fdopen(paths.open(compressed, os.O_RDONLY), 'rb') as compressed_stream, gzip.GzipFile(fileobj=compressed_stream, mode='rb') as f:
            restored = LimitedReader(f)
            while restored.read():
                require(restored.bytes <= before['bytes'], 'Decoded content exceeds source size')
        require(restored.bytes == before['bytes'] and restored.sha.hexdigest() == reader.sha.hexdigest(), 'Full decompression SHA mismatch')
        require(paths.signature(source) == before, 'Source changed before completion')
        final = target / 'actor-inputs.pt.gz'
        paths.publish_archive(compressed, final)
        receipt = dict(request_id=request, approval_reference=approval_reference, status='COMPLETE_LOSSLESS_ARCHIVE_ORIGINAL_RETAINED',
                       original_path=str(source), original_bytes=before['bytes'], original_sha256=reader.sha.hexdigest(),
                       archive_path=str(final), archive_bytes=compressed_bytes, archive_sha256=compressed_sha,
                       full_decompression_sha256=restored.sha.hexdigest(), original_signature=before,
                       original_retained=True, reclaimed_source_bytes=0, saved_bytes_if_separately_approved_replacement=before['bytes']-compressed_bytes,
                       manifest_and_existing_jsonl_gzip_unchanged=True, future_exact_original_path_restore_supported=True,
                       training_unchanged=True, **proof)
        paths.write_json_new(target / '.complete.json', receipt)
        return receipt


class VerifiedActorInput:
    """Future audit consumes read_path while its original verified file FD lives."""
    def __init__(self, paths, fd, original, temp_parent=None, temp_leaf=None):
        self.read_path = Path('/proc/self/fd/%d' % fd)
        self.original_present = temp_parent is None
        self._paths, self._fd, self._original = paths, fd, original
        self._temp_parent, self._temp_leaf = temp_parent, temp_leaf

    def restore_preserve(self):
        if self.original_present:
            return False
        held = os.fstat(self._fd)
        s = os.stat(self._temp_leaf, dir_fd=self._temp_parent, follow_symlinks=False)
        require(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) == (held.st_dev, held.st_ino), 'Own verified temp replaced; refuse restore')
        with self._paths.parent(self._original) as (parent, leaf):
            os.link(self._temp_leaf, leaf, src_dir_fd=self._temp_parent, dst_dir_fd=parent, follow_symlinks=False)
            restored = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                s = os.fstat(restored)
                require((s.st_dev, s.st_ino) == (held.st_dev, held.st_ino), 'Restored inode differs')
                os.fchmod(restored, 0o600)
                os.fsync(restored); os.fsync(parent)
            finally:
                os.close(restored)
        return True


@contextmanager
def verified_actor_input(name, request):
    """Future approved CPU audit: one verified FD path, finally own temp cleanup.

Use handle.read_path only during this context. This pinned descriptor protects
the consumer from a swapped ancestor; no caller may hot-change frozen training.
"""
    require(re.fullmatch('[a-z0-9-]{8,80}', request), 'Safe unique request required')
    with own_lock() as paths:
        directory, manifest, proof = completed_directory(name, paths)
        source = directory / 'actor-inputs.pt'
        try:
            source_fd = paths.open(source, os.O_RDONLY)
        except FileNotFoundError:
            source_fd = None
        if source_fd is not None:
            with os.fdopen(source_fd, 'rb') as original:
                before = signature_fd(original.fileno())
                reader = LimitedReader(original)
                while reader.read():
                    pass
                require(signature_fd(original.fileno()) == before and reader.bytes == before['bytes']
                        and reader.sha.hexdigest() == manifest['payloads']['actor-inputs.pt'], 'Existing source SHA or identity differs')
                original.seek(0)
                yield VerifiedActorInput(paths, original.fileno(), source)
            return
        target = ARCHIVES / name
        receipt, _ = read_json(target / '.complete.json', paths)
        require(receipt['identity'] == proof['identity'] and receipt['original_manifest_sha256'] == proof['original_manifest_sha256'], 'Archive metadata binding changed')
        require(receipt['original_sha256'] == receipt['full_decompression_sha256'] == manifest['payloads']['actor-inputs.pt'], 'Archive original SHA binding differs')
        compressed = target / 'actor-inputs.pt.gz'
        with os.fdopen(paths.open(compressed, os.O_RDONLY), 'rb') as compressed_stream:
            compressed_before = signature_fd(compressed_stream.fileno())
            reader = LimitedReader(compressed_stream)
            while reader.read():
                pass
            require(signature_fd(compressed_stream.fileno()) == compressed_before and reader.sha.hexdigest() == receipt['archive_sha256']
                    and reader.bytes == receipt['archive_bytes'], 'Compressed archive SHA or identity differs')
            compressed_stream.seek(0)
            tmp = target / ('audit-temp-' + request + '.actor-inputs.pt')
            # Keep this actual parent FD through creation, consumption and cleanup.
            with paths.parent(tmp) as (parent, leaf):
                owned_temp = None
                try:
                    fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
                    with os.fdopen(fd, 'wb') as out, gzip.GzipFile(fileobj=compressed_stream, mode='rb') as raw:
                        s = os.fstat(out.fileno()); owned_temp = (s.st_dev, s.st_ino)
                        require(s.st_dev == paths.device, 'Own temp changed device')
                        reader = LimitedReader(raw)
                        while True:
                            block = reader.read()
                            if not block:
                                break
                            require(reader.bytes <= receipt['original_bytes'], 'Decompression exceeds recorded original size')
                            out.write(block)
                        out.flush(); os.fsync(out.fileno()); os.fchmod(out.fileno(), 0o400)
                    require(reader.bytes == receipt['original_bytes'] and reader.sha.hexdigest() == receipt['original_sha256'], 'Temporary original full SHA differs')
                    held = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
                    with os.fdopen(held, 'rb') as verified:
                        s = os.fstat(verified.fileno())
                        require(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) == owned_temp, 'Verified temp inode replaced')
                        yield VerifiedActorInput(paths, verified.fileno(), source, parent, leaf)
                finally:
                    if owned_temp is not None:
                        try:
                            s = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                        except FileNotFoundError:
                            s = None
                        if s is not None:
                            require(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) == owned_temp, 'Own temp path replaced; refuse cleanup')
                            os.unlink(leaf, dir_fd=parent); os.fsync(parent)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['plan', 'archive-preserve', 'restore-preserve'], default='plan')
    p.add_argument('--directory', required=True)
    p.add_argument('--request-id')
    p.add_argument('--approval-reference')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    require(socket.gethostname() == 'h100-10', 'No local model/data read; candidate not deployed')
    if args.action == 'plan':
        with OwnPaths() as paths:
            directory, manifest, proof = completed_directory(args.directory, paths)
        print(json.dumps(dict(status='PLAN_ONLY_ORIGINALS_UNCHANGED', source=str(directory/'actor-inputs.pt'),
                              original_sha256=manifest['payloads']['actor-inputs.pt'], **proof), indent=2))
        return
    require(args.execute and args.approval_reference and args.request_id and re.fullmatch('[a-z0-9-]{8,80}', args.request_id),
            'Explicit future human approval, execute and unique request are required')
    if args.action == 'archive-preserve':
        result = archive_preserve(args.directory, args.request_id, args.approval_reference)
        print(json.dumps(result, indent=2))
    else:
        with verified_actor_input(args.directory, args.request_id) as handle:
            if handle.original_present:
                print('Verified original already present; unchanged')
            else:
                handle.restore_preserve()
                print('Verified original path restored; archive retained')


if __name__ == '__main__':
    main()
