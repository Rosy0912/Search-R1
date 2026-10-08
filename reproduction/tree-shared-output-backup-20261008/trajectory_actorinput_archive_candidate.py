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


def safe(path, root):
    require(path.is_relative_to(root) and not path.is_symlink() and path.resolve(strict=True) == path, 'Path escaped exact own namespace')
    return path


def read_json(path):
    safe(path, RUN)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size < 1024*1024 and before.st_dev == RUN.stat().st_dev, 'Bounded own-device metadata required')
        raw = f.read(before.st_size+1)
        after = os.fstat(f.fileno())
        require(len(raw) == before.st_size and (after.st_size, after.st_mtime_ns, after.st_ino) == (before.st_size, before.st_mtime_ns, before.st_ino), 'Metadata changed during read')
    return json.loads(raw), raw


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_new(path, value):
    pending = path.with_name('.' + path.name + '.pending-%d-%d' % (os.getpid(), time.time_ns()))
    with pending.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(pending, path)
    pending.unlink()
    sync_dir(path.parent)


def file_signature(path):
    s = safe(path, RUN).stat()
    require(stat.S_ISREG(s.st_mode) and s.st_dev == RUN.stat().st_dev, 'Expected regular own-device actor evidence')
    return dict(inode=s.st_ino, device=s.st_dev, bytes=s.st_size, mtime_ns=s.st_mtime_ns)


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


def hash_file(path):
    safe(path, RUN)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        reader = LimitedReader(f)
        while reader.read():
            pass
    return reader.sha.hexdigest(), reader.bytes


def completed_directory(name):
    match = re.fullmatch(r'outer-(\d{6})-attempt-(\d{6})', name)
    require(match is not None, 'Only exact existing outer/attempt trajectory names')
    outer, attempt = map(int, match.groups())
    directory = safe(TRAJECTORIES / name, TRAJECTORIES)
    manifest, raw = read_json(directory / 'manifest.json')
    identity = manifest['identity']
    require(manifest['status'] == 'ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED', 'Trajectory payload publication incomplete')
    require(manifest['rows'] == 2560, 'Frozen five-leaf trajectory row count differs')
    require(identity == dict(outer_step=outer, attempt_id=attempt, job_id='3341', source=CODE, config=CONFIG, data=DATA), 'Trajectory identity mismatch')
    require(set(manifest['payloads']) == {'actor-inputs.pt', 'trajectories.jsonl.gz'}, 'Frozen trajectory format differs')
    require(all(re.fullmatch('[a-f0-9]{64}', v) for v in manifest['payloads'].values()), 'Original payload SHA missing')
    result, proof_raw = read_json(RUN / 'checkpoints/training_state/attempt-results' / ('attempt-%06d.json' % attempt))
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
        physical, direct_raw = read_json(direct_path)
        require(digest(direct_raw) == descriptor['sha256'] and all(physical.get(k) == v for k, v in expected_identity.items()), 'Direct-rank SHA/source identity differs')
        require(physical['rank'] == row['rank'] and physical['world_size'] == 8 and physical['attempt_id'] == attempt
                and physical['observed_outer'] == outer and physical['scheduler_step_before'] == outer-1 and physical['scheduler_step_after'] == outer,
                'Physical direct-rank completion differs')
        direct_proofs[str(direct_path)] = digest(direct_raw)
    latest, latest_raw = read_json(RUN / 'checkpoints/training_state/rolling-completed/latest.json')
    i = latest['identity']
    require(latest['status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED' and i['job_id'] == '3341' and i['code_commit'] == CODE and i['config_sha256'] == CONFIG and i['data_identity'] == DATA,
            'Latest complete native identity differs')
    native_path = Path(latest['manifest_path'])
    alias = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
    if native_path.is_relative_to(alias):
        native_path = ROOT / native_path.relative_to(alias)
    native, native_raw = read_json(native_path)
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
    safe(RUN, ROOT)
    require(not ARCHIVES.is_symlink(), 'Archive root cannot be a symlink')
    ARCHIVES.mkdir(mode=0o700, exist_ok=True)
    safe(ARCHIVES, RUN)
    require(ARCHIVES.stat().st_dev == RUN.stat().st_dev, 'Archive directory must stay on source device')
    mount = subprocess.check_output(['findmnt', '-n', '-T', str(ARCHIVES), '-o', 'SOURCE,FSTYPE'], text=True, timeout=5).strip()
    require(mount.split() == ['/dev/mapper/data-data', 'ext4'], 'Only the original same-machine ext4 is allowed')
    fd = os.open(ARCHIVES / '.single-instance.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a+') as lock:
        require(stat.S_ISREG(os.fstat(lock.fileno()).st_mode), 'Own lock must be regular')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
        subprocess.check_call(['ionice', '-c', '3', '-p', str(os.getpid())], timeout=10)
        yield


def archive_preserve(name, request, approval_reference):
    with own_lock():
        directory, manifest, proof = completed_directory(name)
        source = directory / 'actor-inputs.pt'
        before = file_signature(source)
        require(os.statvfs(ARCHIVES).f_bavail * os.statvfs(ARCHIVES).f_frsize > before['bytes'] + 64 * 1024**3,
                'Keep 64GiB plus one worst-case source-sized archive reserve')
        target = ARCHIVES / name
        require(not target.exists(), 'Existing archive/intent: inspect, never overwrite or duplicate')
        target.mkdir(mode=0o700)
        sync_dir(ARCHIVES)
        write_json_new(target / 'intent.json', dict(request_id=request, approval_reference=approval_reference,
                       source=str(source), source_signature=before, scope='APPROVED_ARCHIVE_PRESERVE_ONLY', **proof))
        compressed = target / 'actor-inputs.pt.gz.partial'
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as raw, compressed.open('xb') as dest:
            require(os.fstat(raw.fileno()).st_ino == before['inode'], 'Source inode changed')
            reader = LimitedReader(raw)
            with gzip.GzipFile(filename='', fileobj=dest, mode='wb', compresslevel=1, mtime=0) as output:
                while True:
                    block = reader.read()
                    if not block:
                        break
                    output.write(block)
            dest.flush(); os.fsync(dest.fileno())
        require(file_signature(source) == before, 'Source changed during archive; preserve all originals')
        require(reader.bytes == before['bytes'] and reader.sha.hexdigest() == manifest['payloads']['actor-inputs.pt'], 'Original source full SHA differs')
        compressed_sha, compressed_bytes = hash_file(compressed)
        require(compressed_bytes < before['bytes'], 'Archive provides no savings; source and own partial retained')
        with gzip.open(compressed, 'rb') as f:
            restored = LimitedReader(f)
            while restored.read():
                require(restored.bytes <= before['bytes'], 'Decoded content exceeds source size')
        require(restored.bytes == before['bytes'] and restored.sha.hexdigest() == reader.sha.hexdigest(), 'Full decompression SHA mismatch')
        require(file_signature(source) == before, 'Source changed before completion')
        final = target / 'actor-inputs.pt.gz'
        os.link(compressed, final); compressed.unlink(); sync_dir(target)
        receipt = dict(request_id=request, approval_reference=approval_reference, status='COMPLETE_LOSSLESS_ARCHIVE_ORIGINAL_RETAINED',
                       original_path=str(source), original_bytes=before['bytes'], original_sha256=reader.sha.hexdigest(),
                       archive_path=str(final), archive_bytes=compressed_bytes, archive_sha256=compressed_sha,
                       full_decompression_sha256=restored.sha.hexdigest(), original_signature=before,
                       original_retained=True, reclaimed_source_bytes=0, saved_bytes_if_separately_approved_replacement=before['bytes']-compressed_bytes,
                       manifest_and_existing_jsonl_gzip_unchanged=True, future_exact_original_path_restore_supported=True,
                       training_unchanged=True, **proof)
        write_json_new(target / '.complete.json', receipt)
        return receipt


@contextmanager
def verified_actor_input(name, request):
    """Future separately approved CPU audit: one verified temp, finally own cleanup.

No caller is allowed to change the frozen trainer/checker on the running node.
An independent archive-aware audit may use this path after human approval.
"""
    require(re.fullmatch('[a-z0-9-]{8,80}', request), 'Safe unique request required')
    with own_lock():
        directory, manifest, proof = completed_directory(name)
        source = directory / 'actor-inputs.pt'
        if source.exists():
            actual, _ = hash_file(source)
            require(actual == manifest['payloads']['actor-inputs.pt'], 'Existing source SHA differs')
            yield source
            return
        target = safe(ARCHIVES / name, ARCHIVES)
        receipt, _ = read_json(target / '.complete.json')
        require(receipt['identity'] == proof['identity'] and receipt['original_manifest_sha256'] == proof['original_manifest_sha256'], 'Archive metadata binding changed')
        require(receipt['original_sha256'] == receipt['full_decompression_sha256'] == manifest['payloads']['actor-inputs.pt'], 'Archive original SHA binding differs')
        compressed = safe(target / 'actor-inputs.pt.gz', ARCHIVES)
        actual, size = hash_file(compressed)
        require(actual == receipt['archive_sha256'] and size == receipt['archive_bytes'], 'Compressed archive SHA differs')
        tmp = target / ('audit-temp-' + request + '.actor-inputs.pt')
        require(not tmp.exists(), 'Own temp request exists: inspect, never overwrite')
        owned_temp = None
        try:
            with gzip.open(compressed, 'rb') as raw, tmp.open('xb') as out:
                st = os.fstat(out.fileno())
                owned_temp = (st.st_dev, st.st_ino)
                reader = LimitedReader(raw)
                while True:
                    block = reader.read()
                    if not block:
                        break
                    require(reader.bytes <= receipt['original_bytes'], 'Decompression exceeds recorded original size')
                    out.write(block)
                out.flush(); os.fsync(out.fileno())
            require(reader.bytes == receipt['original_bytes'] and reader.sha.hexdigest() == receipt['original_sha256'], 'Temporary original full SHA differs')
            os.chmod(tmp, 0o400)
            yield tmp
        finally:
            # Only the unique temp this context created; never a source/archive file.
            if owned_temp is not None and tmp.exists():
                st = tmp.lstat()
                require(stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) == owned_temp, 'Own temp path replaced; refuse cleanup')
                tmp.unlink(); sync_dir(target)


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
        directory, manifest, proof = completed_directory(args.directory)
        print(json.dumps(dict(status='PLAN_ONLY_ORIGINALS_UNCHANGED', source=str(directory/'actor-inputs.pt'),
                              original_sha256=manifest['payloads']['actor-inputs.pt'], **proof), indent=2))
        return
    require(args.execute and args.approval_reference and args.request_id and re.fullmatch('[a-z0-9-]{8,80}', args.request_id),
            'Explicit future human approval, execute and unique request are required')
    if args.action == 'archive-preserve':
        result = archive_preserve(args.directory, args.request_id, args.approval_reference)
        print(json.dumps(result, indent=2))
    else:
        with verified_actor_input(args.directory, args.request_id) as path:
            original = TRAJECTORIES / args.directory / 'actor-inputs.pt'
            if path == original:
                print('Verified original already present; unchanged')
            else:
                # Same ext4 hardlink publishes the verified bytes without overwriting.
                os.link(path, original); os.chmod(original, 0o600); sync_dir(original.parent)
                print('Verified original path restored; archive retained')


if __name__ == '__main__':
    main()
