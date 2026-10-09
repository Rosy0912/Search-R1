"""Mac-only exact-byte XZ compression of one retained accepted GNU tar."""
import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
PREFIX = 'experiments/search_e1/control/opsd_local_archive_recompress_v1/'
ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-online-launch-20261008/checkpoint-backup/protected-0-1-5-11-20261008')
ARCHIVE = ROOT/'protected-checkpoints.tar'
RAW_SHA = '0646f2da8b338108f753f880b30859a96831681cc99767cc6c47a50da93e9dd1'
RAW_BYTES = 6239805440
RECEIPT_SHA = '0169e6b4f9be5429e3fb0d39211cec16cf6a234e208eb32f6af6ad9a6718db65'
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
CHUNK = 4 << 20
MARGIN = 5 << 30
FILTERS = [dict(id=lzma.FILTER_LZMA2, preset=6, dict_size=128 << 20)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def path_checked(path, directory=False):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts, 'Canonical absolute path required')
    for item in [*reversed(p.parents), p]:
        s = item.lstat()
        require(not stat.S_ISLNK(s.st_mode), 'Symlink refused')
        require(stat.S_ISDIR(s.st_mode) if item != p or directory else stat.S_ISREG(s.st_mode), 'Unexpected file type')
    return p


def metadata(path):
    p = path_checked(path); s = p.stat()
    require(s.st_uid == os.getuid() and s.st_nlink == 1, 'Foreign owner or hardlink refused')
    return dict(path=str(p), device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mode=s.st_mode, uid=s.st_uid, nlink=s.st_nlink,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def stable(proof):
    require(metadata(proof['path']) == {k: proof[k] for k in metadata(proof['path'])}, 'File metadata changed')


def opened_same(handle, proof):
    s = os.fstat(handle.fileno())
    require((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode, s.st_uid, s.st_nlink) ==
            tuple(proof[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns', 'mode', 'uid', 'nlink')), 'Fixed handle changed')


def fileproof(path):
    before = metadata(path); h = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        opened_same(f, before)
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
        opened_same(f, before)
    stable(before)
    return dict(**before, sha256=h.hexdigest())


def checked_json(path, sha):
    proof = fileproof(path)
    require(proof['bytes'] <= 1 << 20 and proof['sha256'] == sha, 'Small JSON SHA differs')
    raw = Path(path).read_bytes(); stable(proof)
    require(hashlib.sha256(raw).hexdigest() == sha, 'JSON changed')
    return json.loads(raw), proof


def sync_dir(path):
    fd = os.open(path_checked(path, directory=True), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    with Path(path).open('xb') as f:
        f.write((json.dumps(value, indent=2, sort_keys=True)+'\n').encode())
        f.flush(); os.fsync(f.fileno())
    sync_dir(Path(path).parent)
    return fileproof(path)


def compress_exact(source, output, expected_sha, expected_bytes):
    before = metadata(source); require(before['bytes'] == expected_bytes, 'Input bytes differ')
    ceiling = expected_bytes + expected_bytes//100 + (1 << 20)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, filters=FILTERS)
    original = hashlib.sha256(); written = read = 0
    with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as src, Path(output).open('xb') as dst:
        opened_same(src, before)
        for block in iter(lambda: src.read(CHUNK), b''):
            read += len(block); original.update(block)
            value = compressor.compress(block); written += len(value)
            require(read <= expected_bytes and written <= ceiling, 'Bound exceeded')
            dst.write(value)
        value = compressor.flush(); written += len(value)
        require(written <= ceiling, 'Compressed writer ceiling exceeded')
        dst.write(value); dst.flush(); os.fsync(dst.fileno()); opened_same(src, before)
    stable(before)
    require(read == expected_bytes and original.hexdigest() == expected_sha, 'Original whole-tar SHA differs')
    sync_dir(Path(output).parent)
    return dict(input=before, original_sha256=original.hexdigest(), original_bytes=read,
                writer_ceiling_bytes=ceiling, compressed_bytes=written)


def verify_disk_xz(path, expected_sha, expected_bytes):
    """Separate disk read, bounded decode; exactly one XZ stream and no suffix."""
    before = metadata(path); compressed = hashlib.sha256(); whole = hashlib.sha256(); count = 0
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=256 << 20)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        opened_same(f, before)
        for block in iter(lambda: f.read(CHUNK), b''):
            require(not decoder.eof, 'Trailing bytes or concatenated XZ stream refused')
            compressed.update(block); data = block
            while True:
                value = decoder.decompress(data, max_length=CHUNK); data = b''
                count += len(value); require(count <= expected_bytes, 'Decoded bytes exceed original bound')
                whole.update(value)
                if decoder.eof:
                    require(not decoder.unused_data, 'Trailing bytes or concatenated stream refused')
                    break
                if decoder.needs_input:
                    break
        opened_same(f, before)
    stable(before)
    require(decoder.eof and decoder.check == lzma.CHECK_CRC64, 'Truncated stream or wrong integrity check')
    require(count == expected_bytes and whole.hexdigest() == expected_sha, 'Independent decoded whole-tar identity differs')
    return dict(compressed=dict(**before, sha256=compressed.hexdigest()), decoded_bytes=count,
                decoded_sha256=whole.hexdigest(), check='CRC64', single_stream=True,
                independent_disk_read=True, no_extracted_payload_written=True)


def publication(args):
    m, mp = checked_json(HERE/'source-manifest.json', args.source_manifest_sha256)
    require({p.name for p in HERE.iterdir()} == set(m['files'])|{'source-manifest.json'}, 'Source inventory differs')
    proofs = []
    for name, row in m['files'].items():
        require(Path(name).name == name and name not in ('', '.', '..'), 'Unsafe source name')
        fp = fileproof(HERE/name); require(all(fp[k] == row[k] for k in ('bytes', 'sha256')), 'Source bytes differ')
        proofs.append(fp)
    pub, pp = checked_json(args.publication, args.publication_sha256)
    require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
            and re.fullmatch('[0-9a-f]{40}', args.source_commit or '')
            and pub.get('source_commit') == pub.get('remote_readback_sha') == args.source_commit
            and pub.get('source_manifest_sha256') == mp['sha256'], 'Publication/readback differs')
    for fp in proofs+[mp]:
        rows = [x for x in pub['files'] if x['path'] == PREFIX+Path(fp['path']).name]
        require(len(rows) == 1 and rows[0].get('equal') is True and all(rows[0][k] == fp[k] for k in ('bytes', 'sha256')), 'Published bytes differ')
    return dict(manifest=mp, publication=pp, source_commit=args.source_commit, files=proofs)


def original_acceptance():
    receipt, proof = checked_json(ROOT/'complete-sha-verified.json', RECEIPT_SHA)
    require(receipt.get('accepted') is True and receipt['run_id'] == RUN, 'Historical acceptance/run differs')
    archive = receipt['archive']
    require((archive['path'] if isinstance(archive, dict) else archive) == str(ARCHIVE), 'Historical archive path differs')
    require((archive['sha256'] if isinstance(archive, dict) else receipt['archive_sha256']) == RAW_SHA
            and (archive['bytes'] if isinstance(archive, dict) else receipt['archive_bytes']) == RAW_BYTES, 'Historical archive identity differs')
    return proof


def worker(args, pub, receipt, out):
    started = time.time(); compression = compress_exact(ARCHIVE, out/'archive.tar.xz.partial', RAW_SHA, RAW_BYTES)
    recovered = verify_disk_xz(out/'archive.tar.xz.partial', RAW_SHA, RAW_BYTES)
    stable(compression['input']); stable(receipt)
    for fp in pub['files']+[pub['manifest'], pub['publication']]:
        stable(fp)
    return write_json(out/'worker-complete.json', dict(schema='local-exact-tar-xz-worker-v1', accepted=True,
        actual_pid=os.getpid(), uid=os.getuid(), started_unix=started, ended_unix=time.time(),
        compression=compression, recovery=recovered, historical_receipt=receipt,
        source_publication=pub, original_tar_retained=True, member_hashes_repeated=False,
        verification_scope='All original GNU bytes, including headers and padding, match accepted whole-tar SHA; inherited historical member proof. No new tensor/numerical acceptance.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest-sha256', 'publication', 'publication-sha256', 'source-commit', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--worker-only', action='store_true'); args = p.parse_args()
    require(sys.platform == 'darwin' and os.getuid() == os.geteuid() == 501, 'Only actual Mac UID501 production operation')
    pub = publication(args); receipt = original_acceptance(); before = metadata(ARCHIVE)
    require(before['bytes'] == RAW_BYTES, 'Original archive size differs')
    out = Path(args.output); require(out.is_absolute() and '..' not in out.parts, 'Absolute new destination required')
    path_checked(out.parent, directory=True)
    if args.worker_only:
        path_checked(out, directory=True)
        worker(args, pub, receipt, out); return
    require(not out.exists() and not out.is_symlink(), 'Existing destination refused; failed partials remain')
    ceiling = RAW_BYTES + RAW_BYTES//100 + (1 << 20); free = shutil.disk_usage(out.parent).free
    require(free >= ceiling + MARGIN, 'Storage BACKLOG; not a GPU hold')
    out.mkdir(mode=0o700); sync_dir(out.parent)
    command = [sys.executable, '-B', str(HERE/'recompress.py'), *sys.argv[1:], '--worker-only']
    try:
        write_json(out/'start.json', dict(accepted=False, started_unix=time.time(), input=before,
            historical_receipt=receipt, source_publication=pub, actual_free_bytes=free,
            maximum_new_output_bytes=ceiling, margin_bytes=MARGIN, argv=command, no_deletion=True))
        with (out/'worker.stdout').open('xb') as stdout, (out/'worker.stderr').open('xb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            rc = child.wait(); stdout.flush(); os.fsync(stdout.fileno()); stderr.flush(); os.fsync(stderr.fileno())
        terminal = write_json(out/'actual-child-terminal.json', dict(actual_pid=child.pid,
            actual_returncode=rc, ended_unix=time.time(), status_not_inferred_from_absence=True,
            stdout=fileproof(out/'worker.stdout'), stderr=fileproof(out/'worker.stderr')))
        require(rc == 0, 'Compression/recovery child failed; partial preserved')
        value, wp = checked_json(out/'worker-complete.json', fileproof(out/'worker-complete.json')['sha256'])
        require(value['accepted'] is True and value['actual_pid'] == child.pid
                and value['recovery']['decoded_sha256'] == RAW_SHA and value['recovery']['decoded_bytes'] == RAW_BYTES,
                'Actual worker whole-byte recovery proof absent')
        stable(before); stable(value['recovery']['compressed']); stable(receipt)
        final = out/'archive.tar.xz'; require(not final.exists(), 'Final exists')
        os.rename(out/'archive.tar.xz.partial', final); sync_dir(out)
        compressed = dict(**metadata(final), sha256=value['recovery']['compressed']['sha256'])
        result = dict(schema='local-accepted-exact-gnu-tar-xz-v1', accepted=True, completed_unix=time.time(),
            actual_child_returncode=rc, actual_terminal=terminal, worker=wp, compressed=compressed,
            original=dict(**before, sha256=RAW_SHA), historical_receipt=receipt, source_publication=pub,
            original_tar_retained=True, no_deletion=True, numerical_state_acceptance=False,
            member_acceptance='Historical member proofs preserved via independently verified exact whole-tar bytes',
            no_node_model_torch_gpu_http_or_signals=True, backup_not_gpu_dependency=True)
        proof = write_json(out/'complete-xz-verified.json', result)
        print(json.dumps(dict(accepted=True,complete=proof,compressed_bytes=compressed['bytes']), sort_keys=True))
    except BaseException as error:
        write_json(out/'failed.json', dict(accepted=False, reason=type(error).__name__+': '+str(error),
            partial_retained=(out/'archive.tar.xz.partial').exists(), original_tar_retained=ARCHIVE.exists(),
            no_deletion=True, failed_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
