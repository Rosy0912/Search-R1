#!/usr/bin/env python3
"""Reclaim only exact Mac text duplicates of the accepted lossless capsule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import types


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args()
    ep = Path(args.entry)
    if sha(ep) != args.entry_sha256:
        raise ValueError('entry SHA mismatch')
    e = json.loads(ep.read_text())
    # Source verifier is the already published, actually executed immutable version.
    hp = Path(e['archive_helper']['path'])
    helper_bytes = hp.read_bytes()
    if hashlib.sha256(helper_bytes).hexdigest() != e['archive_helper']['sha256']:
        raise ValueError('archive verifier source changed')
    namespace = {'__file__': str(hp), '__name__': 'closed_capsule'}
    exec(compile(helper_bytes, str(hp), 'exec'), namespace)
    helper = types.SimpleNamespace(**namespace)
    cp = Path(e['capsule_complete']['path'])
    if sha(cp) != e['capsule_complete']['sha256']:
        raise ValueError('capsule receipt SHA mismatch')
    c = json.loads(cp.read_text())
    if not c.get('accepted') or c['text_members'] != 6547 or c['text_bytes'] != 22419590028:
        raise ValueError('wrong capsule scope')
    ip = Path(c['accepted_inventory']['path'])
    if sha(ip) != c['accepted_inventory']['sha256']:
        raise ValueError('original accepted inventory changed')
    inventory = json.loads(ip.read_text())
    if not inventory['accepted'] or inventory['file_count'] != 6603:
        raise ValueError('wrong complete inventory')
    rows = {x['relative_path']: x for x in inventory['files'] if Path(x['relative_path']).suffix in ('.json', '.jsonl')}
    npy = [x for x in inventory['files'] if Path(x['relative_path']).suffix == '.npy']
    if len(rows) != 6547 or len(npy) != 56:
        raise ValueError('wrong partition')
    archive = Path(c['archive']['path'])
    if not archive.is_file() or archive.is_symlink() or sha(archive) != c['archive']['sha256']:
        raise ValueError('whole capsule SHA mismatch')
    helper.verify(archive, rows)  # A second complete, independent decompressed read.
    sync_directory(cp.parent)  # Persist capsule and completion-receipt rename names.
    mp = cp.parent / 'source-metadata.json'
    if sha(mp) != c['source_metadata_sha256']:
        raise ValueError('fixed source identity manifest changed')
    metadata = json.loads(mp.read_text())
    if set(metadata) != set(rows):
        raise ValueError('wrong exact deletion set')
    root = Path(e['source_root']).resolve(strict=True)
    if root.name != 'debug-894021c8f55f436f96bb3cb146ab05bb':
        raise ValueError('unexpected source root')
    files = {}
    for name in rows:
        p = helper.checked_file(root, name)
        if helper.identity(p) != metadata[name] or p.stat().st_size != rows[name]['bytes']:
            raise ValueError('original text changed; nothing deleted')
        files[name] = p
    npy_before = {x['relative_path']: helper.identity(helper.checked_file(root, x['relative_path'])) for x in npy}
    if any(npy_before[x['relative_path']]['size'] != x['bytes'] for x in npy):
        raise ValueError('NPY size changed independently; no text deleted')
    output = Path(e['output']).resolve()
    output.mkdir(exist_ok=False)
    sync_directory(output.parent)
    helper.record(output / 'reclaim-plan.json', {'pid': os.getpid(), 'uid': os.getuid(), 'started_unix': time.time(), 'entry_sha256': args.entry_sha256, 'capsule_complete': e['capsule_complete'], 'archive': c['archive'], 'source_root': str(root), 'exact_file_count': len(rows), 'exact_bytes': 22419590028, 'source_metadata_sha256': c['source_metadata_sha256'], 'delete_Node_NPY_directories_or_capsule': False})
    sync_directory(output)  # Must complete before the first exact duplicate unlink.
    removed = []
    # No recursive tree deletion. Every unlink rechecks the accepted exact inode.
    for name, path in sorted(files.items()):
        if helper.identity(path) != metadata[name] or path.is_symlink():
            raise ValueError('source changed before exact unlink; capsule retained')
        path.unlink()
        sync_directory(path.parent)
        removed.append(name)
        if len(removed) % 100 == 0:
            helper.record(output / 'progress.json', {'removed_count': len(removed), 'observed_unix': time.time()})
            sync_directory(output)
    for name, previous in npy_before.items():
        if helper.identity(helper.checked_file(root, name)) != previous:
            raise ValueError('NPY changed independently; capsule retained')
    helper.record(output / 'complete.json', {'schema': 'full192-local-hybrid-lossless-layout-v1', 'accepted': True, 'ended_unix': time.time(), 'entry_sha256': args.entry_sha256, 'capsule_complete': e['capsule_complete'], 'archive': c['archive'], 'archived_members': 6547, 'remaining_original_npy': 56, 'total_recoverable_accepted_files': 6603, 'reclaimed_exact_duplicate_bytes': 22419590028, 'net_smaller_bytes_vs_original_layout': 18402917541, 'second_decompressed_member_SHA_accepted': True, 'exact_removed_paths': removed, 'Node_originals_and_all_NPY_retained': True, 'no_Node_GPU_HTTP_or_training_lock': True})
    sync_directory(output)
    print(json.dumps({'accepted': True, 'archived': 6547, 'npy_retained': 56, 'reclaimed_bytes': 22419590028}), flush=True)


if __name__ == '__main__':
    main()
