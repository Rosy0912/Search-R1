#!/usr/bin/env python3
"""UNAPPROVED LOCAL PREPARATION. No deployment or asset action has occurred.

Default plan is metadata-only. Preserve and replacement require future explicit
human authorization. This entry has no training/native/model/data operations.
"""
import argparse
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import types

ARCHIVE_ENTRY_SHA = '79f3eee7a0e9fb4e3307c92f09589ef20484943058df4ba0113d8fe4587cdcab'


def archive_module():
    path = Path(__file__).parent / 'trajectory_actorinput_archive_candidate.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_ENTRY_SHA:
        raise RuntimeError('Reviewed archive entry changed; no execution')
    module = types.ModuleType('reviewed_own3341_archive')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def write_bytes_new(paths, path, raw):
    with paths.parent(path) as (parent, leaf):
        pending = '.' + leaf + '.pending-%d' % os.getpid()
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(pending, leaf, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.unlink(pending, dir_fd=parent); os.fsync(parent)


def verified_stream(a, stream, expected_sha, expected_bytes):
    before = a.signature_fd(stream.fileno())
    stream.seek(0); reader = a.LimitedReader(stream)
    while reader.read():
        a.require(reader.bytes <= expected_bytes, 'Source exceeds bound size')
    a.require(reader.bytes == expected_bytes and reader.sha.hexdigest() == expected_sha
              and a.signature_fd(stream.fileno()) == before, 'Pinned full SHA/bytes or source identity differs')
    stream.seek(0)
    return before


def verify_archive(a, stream, receipt):
    before = verified_stream(a, stream, receipt['archive_sha256'], receipt['archive_bytes'])
    with gzip.GzipFile(fileobj=stream, mode='rb') as decoded:
        reader = a.LimitedReader(decoded)
        while reader.read():
            a.require(reader.bytes <= receipt['original_bytes'], 'Decoded content exceeds original bound')
    a.require(reader.bytes == receipt['original_bytes'] and reader.sha.hexdigest() == receipt['original_sha256']
              and a.signature_fd(stream.fileno()) == before, 'Full decompression SHA/bytes or archive identity differs')
    stream.seek(0)
    return before


def bound_archive(a, paths, name, expected):
    directory, manifest, proof = a.completed_directory(name, paths)
    target = a.ARCHIVES / name
    receipt, raw = a.read_json(target / '.complete.json', paths)
    a.require(a.digest(raw) == expected['receipt'] and receipt['archive_sha256'] == expected['archive']
              and receipt['original_sha256'] == expected['original'] == manifest['payloads']['actor-inputs.pt'],
              'Explicit approved original/archive/receipt pins differ')
    a.require(receipt['status'] == 'COMPLETE_LOSSLESS_ARCHIVE_ORIGINAL_RETAINED'
              and receipt['original_sha256'] == receipt['full_decompression_sha256']
              and receipt['archive_bytes'] < receipt['original_bytes'], 'Complete smaller lossless archive required')
    a.require(receipt['original_path'] == str(directory / 'actor-inputs.pt')
              and receipt['archive_path'] == str(target / 'actor-inputs.pt.gz'), 'Archive fixed own paths differ')
    for key in ('identity', 'original_manifest_sha256', 'credited_attempt_proof_sha256', 'direct_rank_proofs'):
        a.require(receipt[key] == proof[key], 'Original source/manifest/eight-rank proof changed: ' + key)
    return directory, target, manifest, proof, receipt, raw


def leaf_present(paths, path):
    with paths.parent(path) as (parent, leaf):
        try:
            s = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(s.st_mode):
                raise RuntimeError('Expected regular own source, not symlink')
            return True
        except FileNotFoundError:
            return False


def pid_ticks():
    return int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])


def rename_no_replace(source_parent, source_leaf, target_parent, target_leaf):
    """Linux ext4 capture, without replacing any existing staging asset.

No fallback: if unsupported, the source remains and the own intent is inspectable.
"""
    library = ctypes.CDLL(None, use_errno=True)
    operation = library.renameat2
    operation.argtypes = [ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    operation.restype = ctypes.c_int
    if operation(source_parent, os.fsencode(source_leaf), target_parent, os.fsencode(target_leaf), 1) != 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))


@contextmanager
def verified_archive_input(name, request, expected_original):
    """Prefer archived bytes even while source remains, for pre-release CPU audit."""
    a = archive_module()
    with a.own_lock() as paths:
        receipt, raw = a.read_json(a.ARCHIVES / name / '.complete.json', paths)
        expected = dict(original=expected_original, archive=receipt['archive_sha256'], receipt=a.digest(raw))
        directory, target, _, _, receipt, _ = bound_archive(a, paths, name, expected)
        tmp = target / ('audit-temp-' + request + '.actor-inputs.pt')
        a.require(re.fullmatch('[a-z0-9-]{8,80}', request), 'Safe unique owned audit request required')
        with paths.parent(tmp) as (parent, leaf):
            owned = None
            try:
                with os.fdopen(paths.open(target / 'actor-inputs.pt.gz', os.O_RDONLY), 'rb') as compressed:
                    compressed_signature = verified_stream(a, compressed, expected['archive'], receipt['archive_bytes'])
                    fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
                    with os.fdopen(fd, 'wb') as out, gzip.GzipFile(fileobj=compressed, mode='rb') as decoded:
                        st = os.fstat(out.fileno()); owned = (st.st_dev, st.st_ino)
                        a.require(st.st_dev == paths.device, 'Own audit temp changed device')
                        reader = a.LimitedReader(decoded)
                        while True:
                            block = reader.read()
                            if not block:
                                break
                            a.require(reader.bytes <= receipt['original_bytes'], 'Archive decompression exceeds original size')
                            out.write(block)
                        out.flush(); os.fsync(out.fileno()); os.fchmod(out.fileno(), 0o400)
                    a.require(reader.bytes == receipt['original_bytes'] and reader.sha.hexdigest() == expected_original
                              and a.signature_fd(compressed.fileno()) == compressed_signature, 'Verified archive/original decoded SHA or identity differs')
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
                with os.fdopen(fd, 'rb') as held:
                    st = os.fstat(held.fileno())
                    a.require(stat.S_ISREG(st.st_mode) and (st.st_dev,st.st_ino) == owned, 'Owned audit temp replaced')
                    handle = a.VerifiedActorInput(paths, held.fileno(), directory/'actor-inputs.pt', parent, leaf)
                    handle.archive_sha256 = expected['archive']; handle.archive_receipt_sha256 = expected['receipt']
                    yield handle
            finally:
                if owned is not None:
                    try:
                        st = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                    except FileNotFoundError:
                        st = None
                    if st is not None:
                        a.require(stat.S_ISREG(st.st_mode) and (st.st_dev,st.st_ino) == owned, 'Refuse cleanup of a replaced audit temp')
                        os.unlink(leaf, dir_fd=parent); os.fsync(parent)


def bound_cpu_receipt(a, paths, name, expected, proof):
    cpu_path = a.ARCHIVES / 'cpu-audits' / expected['cpu_audit_request'] / '.complete.json'
    # Bounded output metadata may exceed the old helper's 1MiB JSON limit.
    with os.fdopen(paths.open(cpu_path, os.O_RDONLY), 'rb') as stream:
        before = a.signature_fd(stream.fileno()); a.require(before['bytes'] <= 32*1024**2, 'Bounded CPU audit receipt required')
        raw = stream.read(before['bytes']+1)
        a.require(len(raw) == before['bytes'] and a.signature_fd(stream.fileno()) == before and a.digest(raw) == expected['cpu'], 'Specific approved CPU receipt SHA changed')
    value = json.loads(raw); adapter = value['archive_adapter']
    expected_identity = dict(job_id='3341', run_id=a.RUN.name, code_commit=a.CODE, config_sha256=a.CONFIG,
                             data_identity=a.DATA, model_identity=a.MODEL)
    a.require(value['status'] == 'FRESH200_REAL_BEHAVIOR_MU_CPU_BINDING_VERIFIED' and value['cuda_initialized'] is False
              and value['identity'] == expected_identity and value['optimizer_writes'] == value['model_forwards'] == 0
              and adapter['status'] == 'INDEPENDENT_ARCHIVE_AWARE_CPU_ACCESS_COMPLETED'
              and adapter['original_verification_logic_unchanged'] is True
              and adapter['original_checker_sha256'] == 'c56e3c79959c501d6256ba51f0cd455b2de0a7540ea0c1ff9231fae066afe5bc'
              and adapter['replacement_entry_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
              and adapter['archive_entry_sha256'] == ARCHIVE_ENTRY_SHA, 'Actual original-logic archive-aware CPU acceptance missing')
    matching = [r for r in adapter['actor_inputs'] if r['directory'] == name and r['archive_used'] is True
                and r['actor_input_sha256'] == expected['original'] and r['archive_sha256'] == expected['archive']
                and r['archive_receipt_sha256'] == expected['receipt']]
    original = [r for r in value['outer_records'] if r['candidate_outer'] == proof['outer'] and r['attempt_id'] == proof['attempt']
                and r['credited_outer'] is True and r['manifest_sha256'] == proof['original_manifest_sha256'] and r['actor_input_sha256'] == expected['original']]
    a.require(len(matching) == len(original) == 1, 'CPU checker must have consumed this exact archived source before release')
    return a.digest(raw)


def replace_existing(name, request, approval, expected):
    a = archive_module()
    with a.own_lock() as paths:
        directory, target, manifest, proof, receipt, receipt_raw = bound_archive(a, paths, name, expected)
        cpu_sha = bound_cpu_receipt(a, paths, name, expected, proof)
        source = directory / 'actor-inputs.pt'
        staging = target / ('.replacement-staging-' + request)
        quarantine = staging / 'actor-inputs.pt'
        # One permanent intent per directory. A prior intent is UNKNOWN until
        # separately inspected/reconciled; neither duplicate nor blind retry.
        intent = dict(status='REPLACEMENT_INTENT_ONLY', request_id=request, approval_reference=approval,
                      pid=os.getpid(), pid_startticks=pid_ticks(), original_path=str(source),
                      owned_staging_path=str(quarantine),
                      archive_entry_sha256=ARCHIVE_ENTRY_SHA, replacement_entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      expected_pins=expected, **proof)
        paths.write_json_new(target / '.replacement.intent.json', intent)
        paths.mkdir(staging)
        source_parent_context = paths.parent(source)
        with source_parent_context as (parent, leaf):
            source_fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(source_fd, 'rb') as original, os.fdopen(paths.open(target / 'actor-inputs.pt.gz', os.O_RDONLY), 'rb') as compressed:
                source_stat = os.fstat(original.fileno())
                a.require(stat.S_ISREG(source_stat.st_mode) and source_stat.st_dev == paths.device
                          and source_stat.st_nlink == 1, 'Require one-link regular own source for release')
                source_signature = verified_stream(a, original, expected['original'], receipt['original_bytes'])
                a.require(source_signature == receipt['original_signature'], 'Source inode/size/mtime differs from preserved archive')
                archive_signature = verify_archive(a, compressed, receipt)
                archive_stat = os.fstat(compressed.fileno())
                source_allocated, archive_allocated = source_stat.st_blocks * 512, archive_stat.st_blocks * 512
                a.require(source_allocated > archive_allocated, 'No allocated-block savings; preserve source')
                # Original manifest and archive-complete raw bytes remain
                # recoverable with their actual original hashes, not reserialized.
                _, manifest_raw = a.read_json(directory / 'manifest.json', paths)
                a.require(a.digest(manifest_raw) == proof['original_manifest_sha256'], 'Original manifest changed')
                write_bytes_new(paths, target / '.replacement.original-manifest.json', manifest_raw)
                write_bytes_new(paths, target / '.replacement.archive-complete.json', receipt_raw)
                for path, sha in [(target / '.replacement.original-manifest.json', proof['original_manifest_sha256']),
                                  (target / '.replacement.archive-complete.json', expected['receipt'])]:
                    _, readback = a.read_json(path, paths); a.require(a.digest(readback) == sha, 'Small original metadata readback differs')
                ready = dict(status='DURABLE_VERIFIED_ARCHIVE_READY_BEFORE_SOURCE_UNLINK', request_id=request,
                             expected_pins=expected, source_signature=source_signature, archive_signature=archive_signature,
                             original_allocated_bytes=source_allocated, archive_allocated_bytes=archive_allocated,
                             allocated_savings_if_no_other_open_reader=source_allocated-archive_allocated,
                             original_manifest_sha256=proof['original_manifest_sha256'], full_decompression_verified=True)
                ready['archive_aware_cpu_receipt_sha256'] = cpu_sha
                ready['owned_staging_path'] = str(quarantine)
                paths.write_json_new(target / '.replacement.ready.json', ready)
                # Re-pin/recheck source immediately before the only destructive
                # operation. Both actual parent FD and original FD stay held.
                check_fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
                try:
                    a.require(a.signature_fd(check_fd) == source_signature and a.signature_fd(original.fileno()) == source_signature
                              and a.signature_fd(compressed.fileno()) == archive_signature
                              and paths.signature(target / 'actor-inputs.pt.gz') == archive_signature,
                              'Source/archive inode changed before unlink; no release')
                finally:
                    os.close(check_fd)
                # Atomically capture the actual current leaf in our exclusive
                # private staging namespace. Then validate the captured inode
                # before removing it. A swapped source is preserved, never
                # deleted on a stale check-before-unlink identity assertion.
                with paths.parent(quarantine) as (stage_parent, stage_leaf):
                    rename_no_replace(parent, leaf, stage_parent, stage_leaf)
                    os.fsync(parent); os.fsync(stage_parent)
                    captured = os.open(stage_leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=stage_parent)
                    try:
                        a.require(a.signature_fd(captured) == source_signature, 'Captured source leaf differs; preserve staging for inspection')
                        s = os.stat(stage_leaf, dir_fd=stage_parent, follow_symlinks=False)
                        a.require(stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino) == (source_signature['device'],source_signature['inode']), 'Own staging inode replaced; refuse removal')
                        os.unlink(stage_leaf, dir_fd=stage_parent); os.fsync(stage_parent)
                    finally:
                        os.close(captured)
            # Our original FD is closed here. Another process may still hold its
            # inode: the receipt distinguishes unlink from guaranteed free space.
            a.require(not leaf_present(paths, source), 'Source pathname unexpectedly recreated')
        final = dict(ready)
        final.update(status='SOURCE_UNLINKED_VERIFIED_ARCHIVE_RETAINED', request_id=request,
                     approval_reference=approval, expected_pins=expected,
                     original_unlinked=True, archive_retained=True, source_manifest_and_jsonl_unchanged=True,
                     guaranteed_filesystem_free_bytes_increase=False,
                     free_space_limit='Other open readers and global filesystem changes may delay/obscure reclamation.',
                     training_unchanged=True, no_native_model_data_operations=True)
        paths.write_json_new(target / '.replacement.complete.json', final)
        return final


def reconcile(name, request, approval, expected):
    a = archive_module()
    with a.own_lock() as paths:
        directory, target, _, proof, receipt, _ = bound_archive(a, paths, name, expected)
        intent, _ = a.read_json(target / '.replacement.intent.json', paths)
        ready, ready_raw = a.read_json(target / '.replacement.ready.json', paths)
        a.require(intent['expected_pins'] == ready['expected_pins'] == expected
                  and ready['request_id'] == intent['request_id'] and ready['original_manifest_sha256'] == proof['original_manifest_sha256'],
                  'No fixed durable pre-unlink ready proof for this original request')
        if leaf_present(paths, directory / 'actor-inputs.pt'):
            return dict(status='SOURCE_PRESENT_NO_UNLINK_OR_RETRY_PERFORMED', original_request_id=intent['request_id'],
                        message='Inspect source/intent. This reconciliation never deletes a present source.')
        with os.fdopen(paths.open(target / 'actor-inputs.pt.gz', os.O_RDONLY), 'rb') as compressed:
            actual = verify_archive(a, compressed, receipt)
            a.require(actual == ready['archive_signature'], 'Archive changed since durable pre-unlink ready proof')
        paths.write_json_new(target / ('.reconcile-intent-' + request + '.json'),
                             dict(request_id=request, approval_reference=approval, original_request_id=intent['request_id']))
        expected_staging = target / ('.replacement-staging-' + intent['request_id']) / 'actor-inputs.pt'
        a.require(intent['owned_staging_path'] == ready['owned_staging_path'] == str(expected_staging), 'Owned staging path differs')
        staging_unlinked = False
        if leaf_present(paths, expected_staging):
            with paths.parent(expected_staging) as (parent, leaf):
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
                with os.fdopen(fd, 'rb') as held:
                    a.require(verified_stream(a, held, expected['original'], receipt['original_bytes']) == ready['source_signature'], 'Staging inode/full SHA differs; preserve all files')
                    st = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                    a.require(stat.S_ISREG(st.st_mode) and (st.st_dev,st.st_ino) == (ready['source_signature']['device'],ready['source_signature']['inode']), 'Own staging leaf replaced')
                    os.unlink(leaf, dir_fd=parent); os.fsync(parent); staging_unlinked = True
        final = dict(status='RECONCILED_SOURCE_ABSENT_VERIFIED_ARCHIVE_RETAINED', request_id=request,
                     approval_reference=approval, original_request_id=intent['request_id'], expected_pins=expected,
                     durable_ready_sha256=a.digest(ready_raw), source_unlink_performed_by_reconcile=False,
                     absence_cause_or_time_not_inferred=True, archive_retained=True, full_decompression_verified=True,
                     only_verified_owned_staging_unlinked_by_reconcile=staging_unlinked,
                     training_unchanged=True, original_intent_and_ready_preserved=True)
        paths.write_json_new(target / '.replacement.complete.json', final)
        return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['plan', 'archive-preserve', 'replace-existing', 'reconcile'], default='plan')
    p.add_argument('--directory', required=True); p.add_argument('--request-id'); p.add_argument('--approval-reference')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--cpu-audit-request')
    for name in ('original', 'archive', 'receipt', 'cpu'):
        p.add_argument('--expected-' + name + '-sha256')
    args = p.parse_args(); a = archive_module()
    a.require(a.socket.gethostname() == 'h100-10', 'Only own retained18 host; local preparation is not execution authority')
    if args.action == 'plan':
        with a.OwnPaths() as paths:
            directory, manifest, proof = a.completed_directory(args.directory, paths)
            print(json.dumps(dict(status='PLAN_ONLY_UNAPPROVED_NO_ASSET_CHANGE', source=str(directory/'actor-inputs.pt'),
                                  original_sha256=manifest['payloads']['actor-inputs.pt'], **proof), indent=2))
        return
    a.require(args.execute and args.approval_reference and args.request_id and re.fullmatch('[a-z0-9-]{8,80}', args.request_id),
              'Future explicit human authorization plus execute/reference/unique request required')
    if args.action == 'archive-preserve':
        result = a.archive_preserve(args.directory, args.request_id, args.approval_reference)
    else:
        expected = {name:getattr(args, 'expected_' + name + '_sha256') for name in ('original','archive','receipt','cpu')}
        a.require(all(type(x) is str and re.fullmatch('[a-f0-9]{64}', x) for x in expected.values()), 'Specific approved original/archive/receipt SHA pins required')
        a.require(args.cpu_audit_request and re.fullmatch('[a-z0-9-]{8,40}', args.cpu_audit_request), 'Specific completed CPU audit request required')
        expected['cpu_audit_request'] = args.cpu_audit_request
        action = replace_existing if args.action == 'replace-existing' else reconcile
        result = action(args.directory, args.request_id, args.approval_reference, expected)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
