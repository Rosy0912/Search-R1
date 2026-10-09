"""Fixed Mac-only reclamation of one exact-XZ-protected original GNU tar."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
PREFIX = 'experiments/search_e1/control/opsd_local_single_tar_reclaim_v1/'
ANALYSIS = Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output')
ROOT = ANALYSIS/'opsd-online-launch-20261008/checkpoint-backup/protected-0-1-5-11-20261008'
ARCHIVE = ROOT/'protected-checkpoints.tar'
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
RAW_SHA = '0646f2da8b338108f753f880b30859a96831681cc99767cc6c47a50da93e9dd1'
RAW_BYTES = 6239805440
G27 = ANALYSIS/'opsd-ablation-execution-20261008/goal-continuation-20261009-27'
XZ_DIR = G27/'protected-gnu-exact-xz-5ad8a6d-20261009'
COMPLETION = XZ_DIR/'complete-xz-verified.json'
XZ = XZ_DIR/'archive.tar.xz'
RECOMPRESS_COMMIT = '5ad8a6df0bad10f5eb49c7bf95effb5b3af6a913'
RECOMPRESS_MANIFEST_SHA = '51b13e09dbe81daaff8cc1942dc6b5bb118bdb4037e5468ba8006855392d633b'
RECOMPRESS_PUBLICATION = G27/'recompression-publication/publication-verified.private.json'
RECOMPRESS_PUBLICATION_SHA = '361d546e72df01c7c54a92e0446281aae746aad685737ca583ace7a99b927f03'
PARENT_DIR = G27/'recompression-actual-5ad8a6d'
PARENT_TERMINAL = PARENT_DIR/'root-observed-parent-terminal.private.json'
PARENT_START = PARENT_DIR/'start.private.json'
PARENT_START_SHA = '5731582efd0e64e6d62840f657ef614ea7b7240307065f7912b198edae34bb2d'
PRIMITIVE_SHA = '4c4d2518c9fc44f8ef75ef0ef31c3fe51d7cb00f2de0242932587c26f2bd3e6c'
META_KEYS = ('path', 'device', 'inode', 'bytes', 'mode', 'uid', 'nlink', 'mtime_ns', 'ctime_ns')
RENAME_KEYS = ('device', 'inode', 'bytes', 'mode', 'uid', 'nlink', 'mtime_ns', 'sha256')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def primitives():
    """Compile the pinned published low-level helper, never its main/execute."""
    path = HERE.parent/'opsd_backup_duplicate_reclaim_v1/reclaim.py'
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    require(all(getattr(before, k) == getattr(after, k) for k in
                ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_uid', 'st_nlink', 'st_mtime_ns', 'st_ctime_ns'))
            and not path.is_symlink()
            and hashlib.sha256(raw).hexdigest() == PRIMITIVE_SHA, 'Published primitive source differs')
    module = types.ModuleType('_single_tar_fixed_primitives'); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    module.checked_path(path)
    return module


def read_ref(p, ref):
    value, actual = p.checked_json(ref['path'], ref['sha256'])
    require(all(actual[k] == ref[k] for k in META_KEYS), 'Fixed small ref metadata differs')
    return value, actual


def small_ref(p, ref):
    require(p.metadata(ref['path'])['bytes'] <= 1 << 20, 'Oversized actual wait evidence refused')
    actual = p.fileproof(ref['path'])
    require(all(actual[k] == ref[k] for k in (*META_KEYS, 'sha256')), 'Actual wait evidence SHA/stat differs')
    return actual


def validate_parent_terminal(value, start, completion_ref):
    require(value.get('schema') == 'local-recompress-parent-terminal-root-observation-v1'
            and value.get('actual_pid') == start.get('actual_pid') == 41057
            and value.get('uid') == start.get('uid') == 501
            and value.get('actual_returncode') == 0
            and value.get('status_not_inferred_from_absence') is True,
            'Independent actual recompression parent rc0 absent/nonzero')
    require(value['start']['path'] == str(PARENT_START) and value['start']['sha256'] == PARENT_START_SHA
            and value['completion'] == completion_ref, 'Parent start/completion binding differs')


def published_source(p, args):
    manifest, mp = p.checked_json(HERE/'source-manifest.json', args.source_manifest_sha256)
    require({x.name for x in HERE.iterdir()} == set(manifest['files'])|{'source-manifest.json'}, 'Own source inventory differs')
    files = []
    for name, row in manifest['files'].items():
        require(Path(name).name == name and name not in ('', '.', '..'), 'Unsafe source name')
        fp = p.fileproof(HERE/name)
        require(all(fp[k] == row[k] for k in ('bytes', 'sha256')), 'Own source bytes differ')
        files.append(fp)
    pub, pp = p.checked_json(args.publication, args.publication_sha256)
    require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
            and re.fullmatch('[0-9a-f]{40}', args.source_commit or '')
            and pub.get('source_commit') == pub.get('remote_readback_sha') == args.source_commit
            and pub.get('source_manifest_sha256') == mp['sha256'], 'Own publication/readback differs')
    for fp in files+[mp]:
        rows = [r for r in pub['files'] if r['path'] == PREFIX+Path(fp['path']).name]
        require(len(rows) == 1 and rows[0].get('equal') is True
                and all(rows[0][k] == fp[k] for k in ('bytes', 'sha256')), 'Own published file differs')
    return dict(manifest=mp, publication=pp, source_commit=args.source_commit, files=files,
                primitive=p.fileproof(HERE.parent/'opsd_backup_duplicate_reclaim_v1/reclaim.py'))


def validate_recovery(complete, worker, terminal):
    require(complete.get('schema') == 'local-accepted-exact-gnu-tar-xz-v1'
            and complete.get('accepted') is True and complete.get('actual_child_returncode') == 0,
            'Exact XZ completion absent/incomplete')
    require(worker.get('schema') == 'local-exact-tar-xz-worker-v1' and worker.get('accepted') is True
            and terminal.get('actual_returncode') == 0 and terminal.get('status_not_inferred_from_absence') is True
            and terminal['actual_pid'] == worker['actual_pid'] and worker['uid'] == 501, 'Actual child wait identity differs')
    require(complete['original']['path'] == str(ARCHIVE) and complete['original']['bytes'] == RAW_BYTES
            and complete['original']['sha256'] == RAW_SHA, 'Original fixed GNU tar differs')
    recovered = worker['recovery']; compression = worker['compression']
    require(recovered['decoded_sha256'] == RAW_SHA and recovered['decoded_bytes'] == RAW_BYTES
            and recovered.get('check') == 'CRC64' and recovered.get('single_stream') is True
            and recovered.get('independent_disk_read') is True and recovered.get('no_extracted_payload_written') is True,
            'Independent exact whole-GNU-byte decode not accepted')
    require(compression['original_sha256'] == RAW_SHA and compression['original_bytes'] == RAW_BYTES
            and {k: compression['input'][k] for k in META_KEYS}
                == {k: complete['original'][k] for k in META_KEYS}, 'Compression source differs')
    require(complete['compressed']['path'] == str(XZ)
            and recovered['compressed']['path'] == str(XZ_DIR/'archive.tar.xz.partial')
            and all(complete['compressed'][k] == recovered['compressed'][k] for k in RENAME_KEYS),
            'Final XZ differs from independently decoded inode; old partial path is historical')
    require(complete.get('original_tar_retained') is True and complete.get('no_deletion') is True
            and worker.get('original_tar_retained') is True and worker.get('member_hashes_repeated') is False,
            'Unexpected recompression scope')


def completion_chain(p, sha, parent_sha, original_present=True):
    complete, cp = p.checked_json(COMPLETION, sha)
    worker, wp = read_ref(p, complete['worker']); terminal, tp = read_ref(p, complete['actual_terminal'])
    validate_recovery(complete, worker, terminal)
    start, sp = p.checked_json(XZ_DIR/'start.json', p.fileproof(XZ_DIR/'start.json')['sha256'])
    require(start.get('accepted') is False and start.get('no_deletion') is True
            and start['input'] == worker['compression']['input'], 'Compression start/source differs')
    argv = start['argv']
    require(argv[:3] == [argv[0], '-B', str(HERE.parent/'opsd_local_archive_recompress_v1/recompress.py')]
            and argv.count('--worker-only') == 1
            and argv[argv.index('--output')+1] == str(XZ_DIR)
            and argv[argv.index('--source-commit')+1] == RECOMPRESS_COMMIT
            and argv[argv.index('--source-manifest-sha256')+1] == RECOMPRESS_MANIFEST_SHA
            and argv[argv.index('--publication')+1] == str(RECOMPRESS_PUBLICATION)
            and argv[argv.index('--publication-sha256')+1] == RECOMPRESS_PUBLICATION_SHA,
            'Published compression launch differs')
    parent, pp_terminal = p.checked_json(PARENT_TERMINAL, parent_sha)
    root_start, ps = read_ref(p, parent['start'])
    validate_parent_terminal(parent, root_start, cp)
    require(root_start.get('accepted') is False and root_start.get('original_tar_retained') is True
            and root_start['output'] == str(XZ_DIR) and argv == root_start['argv']+['--worker-only'],
            'Actual parent launch differs from child launch')
    wait_evidence = small_ref(p, parent['actual_wait_evidence'])
    source = complete['source_publication']
    require(source == worker['source_publication'] == start['source_publication']
            and source['source_commit'] == RECOMPRESS_COMMIT
            and source['manifest']['sha256'] == RECOMPRESS_MANIFEST_SHA
            and source['publication']['path'] == str(RECOMPRESS_PUBLICATION)
            and source['publication']['sha256'] == RECOMPRESS_PUBLICATION_SHA, 'Compression publication binding differs')
    pub, pp = read_ref(p, source['publication']); manifest, mp = read_ref(p, source['manifest'])
    require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
            and pub.get('source_commit') == pub.get('remote_readback_sha') == RECOMPRESS_COMMIT
            and pub.get('source_manifest_sha256') == RECOMPRESS_MANIFEST_SHA, 'Compression remote readback differs')
    require({Path(x['path']).name for x in source['files']} == set(manifest['files']), 'Compression source inventory differs')
    for fp in source['files']+[mp]:
        p.stable(fp)
        fresh = p.fileproof(fp['path'])
        require(fresh == fp, 'Fresh compression source SHA/metadata differs')
        rows = [r for r in pub['files'] if r['path'] == 'experiments/search_e1/control/opsd_local_archive_recompress_v1/'+Path(fp['path']).name]
        require(len(rows) == 1 and rows[0].get('equal') is True
                and all(rows[0][k] == fp[k] for k in ('bytes', 'sha256')), 'Compression published source differs')
    rows, oldrefs = p.inventory(p.SCOPES[0])
    require(len(rows) == 58 and complete['historical_receipt'] == worker['historical_receipt']
            == start['historical_receipt'] == oldrefs[0], 'Historical 58-member acceptance differs')
    p.stable(complete['compressed'])
    if original_present:
        p.stable(complete['original'])
    for ref in [*oldrefs, cp, wp, tp, sp, pp, mp, pp_terminal, ps, wait_evidence, *source['files']]:
        p.stable(ref)
    return dict(completion=cp, worker=wp, terminal=tp, start=sp,
                parent_terminal=pp_terminal, parent_start=ps, actual_parent_wait_evidence=wait_evidence,
                original=complete['original'], compressed=complete['compressed'],
                preserve=[complete['compressed'], *oldrefs, cp, wp, tp, sp, pp, mp,
                          pp_terminal, ps, wait_evidence, *source['files']],
                historical_member_count=58, member_hashes_repeated=False)


def unlink_one(p, row, pending, intended, preserve):
    for ref in preserve:
        p.stable(ref)
    if pending.exists():
        old, _ = p.checked_json(pending, p.fileproof(pending)['sha256'])
        require(old == intended, 'Durable pending differs')
    else:
        p.stable(row); p.no_open_files([row['path']]); p.write_json(pending, intended)
    path = Path(row['path'])
    if not path.exists():
        return 'absent_after_durable_pending_actual_unlink_not_inferred'
    p.stable(row); p.no_open_files([row['path'], *[r['path'] for r in preserve]])
    parent = p.checked_path(path.parent, directory=True)
    fd = os.open(parent, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        parentstat = parent.stat(); handle = os.fstat(fd)
        require((parentstat.st_dev, parentstat.st_ino) == (handle.st_dev, handle.st_ino), 'Parent handle changed')
        s = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        require((s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_uid, s.st_nlink, s.st_mtime_ns, s.st_ctime_ns)
                == tuple(row[k] for k in ('device', 'inode', 'bytes', 'mode', 'uid', 'nlink', 'mtime_ns', 'ctime_ns')),
                'Immediate exact tar identity changed')
        for ref in preserve:
            p.stable(ref)
        p.stable(row); os.unlink(path.name, dir_fd=fd); os.fsync(fd)
    finally:
        os.close(fd)
    return 'exact_planned_original_tar_unlinked'


def prepare(p, args, source):
    output = Path(args.output); p.checked_path(output.parent, directory=True)
    require(output.is_absolute() and not output.exists() and not output.is_symlink()
            and ROOT not in output.parents and XZ_DIR not in output.parents, 'New independent plan directory required')
    chain = completion_chain(p, args.completion_sha256, args.parent_terminal_sha256)
    p.no_open_files([str(ARCHIVE), str(XZ)])
    output.mkdir(mode=0o700); p.fsync_dir(output.parent)
    return p.write_json(output/'plan.json', dict(schema='local-single-original-gnu-tar-reclaim-plan-v1',
        accepted=True, run_id=RUN, created_unix=time.time(), source=source, chain=chain,
        target=chain['original'], logical_duplicate_bytes=RAW_BYTES,
        one_regular_file_only=True, inherited_historical_member_proof_via_exact_decoded_whole_bytes=True,
        large_hashes_repeated=False, no_gpu_queue_dependency=True))


def execute(p, args, source):
    plan, pref = p.checked_json(args.plan, args.plan_sha256); out = Path(args.plan).parent
    require(plan.get('schema') == 'local-single-original-gnu-tar-reclaim-plan-v1' and plan.get('accepted') is True
            and plan['run_id'] == RUN and plan['source'] == source
            and plan['target']['path'] == str(ARCHIVE) and plan['target']['sha256'] == RAW_SHA
            and plan['target'] == plan['chain']['original']
            and plan['logical_duplicate_bytes'] == plan['target']['bytes'] == RAW_BYTES
            and plan.get('one_regular_file_only') is True, 'Exact immutable plan/source differs')
    require(out.is_absolute() and ROOT not in out.parents and XZ_DIR not in out.parents, 'Plan location overlaps retained package')
    pending = out/'unlink-pending.json'; done = out/'unlink-done.json'; intent = out/'execution-intent.json'
    present = ARCHIVE.exists() or ARCHIVE.is_symlink()
    require(present or pending.exists(), 'Absence without this exact durable pending is not success')
    chain = completion_chain(p, plan['chain']['completion']['sha256'],
                             plan['chain']['parent_terminal']['sha256'], original_present=present)
    require(chain == plan['chain'], 'Accepted completion chain changed')
    expected = dict(plan=pref, source=source, target=plan['target'])
    if intent.exists():
        saved, _ = p.checked_json(intent, p.fileproof(intent)['sha256'])
        require(saved == expected, 'Unique durable execution intent differs')
    else:
        require(present and not pending.exists() and not done.exists(), 'Fresh intent requires original tar')
        p.stable(plan['target']); p.no_open_files([str(ARCHIVE), str(XZ)])
        p.write_json(intent, expected)
    intended = dict(plan=pref, file=plan['target'])
    if done.exists():
        value, _ = p.checked_json(done, p.fileproof(done)['sha256'])
        require(value['intent'] == intended and not present and pending.exists(), 'Finished original tar reappeared/differs')
        saved, _ = p.checked_json(pending, p.fileproof(pending)['sha256'])
        require(saved == intended, 'Finished durable pending differs')
    else:
        outcome = unlink_one(p, plan['target'], pending, intended, chain['preserve'])
        value = dict(intent=intended, outcome=outcome, observed_unix=time.time())
        p.write_json(done, value)
    for ref in chain['preserve']+[pref]:
        p.stable(ref)
    require(not ARCHIVE.exists() and not ARCHIVE.is_symlink(), 'Original tar still exists')
    receipt = out/'reclaim-complete.json'
    result = dict(schema='local-single-original-gnu-tar-reclaim-complete-v1', accepted=True,
        plan=pref, source=source, journal=value, preserve=chain['preserve'], logical_duplicate_bytes=RAW_BYTES,
        actual_apfs_free_gain_not_measured=True, xz_receipts_node_and_directories_preserved=True,
        no_member_or_model_numerical_reverification=True, no_gpu_queue_dependency=True)
    if receipt.exists():
        old, proof = p.checked_json(receipt, p.fileproof(receipt)['sha256'])
        require(old == result, 'Completion differs'); return proof
    return p.write_json(receipt, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=['prepare', 'execute'])
    for name in ('source-manifest-sha256', 'publication', 'publication-sha256', 'source-commit'):
        parser.add_argument('--'+name, required=True)
    for name in ('completion-sha256', 'parent-terminal-sha256', 'output', 'plan', 'plan-sha256'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    require(sys.platform == 'darwin' and os.getuid() == os.geteuid() == 501, 'Only actual local Mac UID501 allowed')
    p = primitives(); source = published_source(p, args)
    p.checked_path(ROOT, directory=True)
    fd = os.open(ROOT/'.loose-duplicate-reclaim.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        s = os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_uid == 501 and s.st_nlink == 1, 'Lease identity differs')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'prepare':
            require(args.output and args.completion_sha256 and args.parent_terminal_sha256
                    and not args.plan and not args.plan_sha256, 'prepare needs completion/actual parent terminal SHA and fresh output only')
            result = prepare(p, args, source)
        else:
            require(args.plan and args.plan_sha256 and not args.output and not args.completion_sha256
                    and not args.parent_terminal_sha256, 'execute needs immutable plan/pathSHA only')
            result = execute(p, args, source)
        print(json.dumps(dict(accepted=True, mode=args.mode, result=result), sort_keys=True))
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
