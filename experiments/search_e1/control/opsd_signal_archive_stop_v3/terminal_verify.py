"""Read-only terminal audit of the one completed K4 stop with exit-race HOLD.

No signals, Node writes, model payload reads, Torch, HTTP, or pidfd probes.
The original failed receipt remains unchanged; this emits a separate observation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

HELPER_SHA = '79f760c6cfed229cf2aad34347dcf62acbce580f6b8650df571fce6f0e582a1f'
RESULT_SHA = '1e58cb985f68e8ff5e772d58f2d0be64c2bd6b1e9f137399aa0858d7954e3c2b'
PRESERVATION_SHA = 'a7c99df18d0e4f83e88caa0bcb07c2d9324c46bd452c46b20c13b715e952a592'
QUIESCED_SHA = '15243436aba89cdd79074b98e4b9bb6d68ae539124ebacbfdebee5d5704746cf'
AUTH_SHA = 'b1b362ede782745c6b37ec7ea9dbebbffabb46223981c5a2d588bfe29e9d59f6'
DESTINATION = 'user-stop-archive-20261009-057b905a73ae4ca2a28c220466d5608a'
COPY_SCOPES = ('source', 'runtime', 'opsd-runtime', 'logs', 'receipts',
               'runs/archive/round-1-grpo/train.log')


class Hold(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Hold(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bound(path, digest):
    raw = Path(path).read_bytes()
    require(sha(raw) == digest, 'Bound input SHA mismatch: ' + str(path))
    return raw


def load_helper(path):
    require(file_sha(path) == HELPER_SHA, 'Published v3 helper SHA mismatch')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('published_archive_stop_v3_readonly', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(record):
    require(all(type(record.get(k)) is int and record[k] > 0
                for k in ('pid', 'uid', 'starttick')), 'Invalid captured identity')
    return record['pid'], record['uid'], record['starttick']


def validate_receipts(result, preservation, quiesced, helper):
    require(result.get('schema') == 'k4-archive-replacement-stop-v1' and
            result.get('status') == 'HOLD_STOP_INCOMPLETE' and
            result.get('accepted') is False and result.get('signals_executed') is True and
            result.get('error_type') == 'Hold' and
            result.get('error') == 'Known PID identity changed: 1434651',
            'Receipt is not the specific exit-race HOLD')
    expected = {'base': str(helper.BASE), 'run_id': helper.RUN, 'uid': helper.UID,
                'hostname': helper.HOST, 'helper_source_sha256': HELPER_SHA,
                'authorization_sha256': AUTH_SHA,
                'preserved_directory': str(helper.BASE / 'preserved' / DESTINATION)}
    for key, value in expected.items():
        require(result.get(key) == value, 'Old receipt binding mismatch: ' + key)
    require(sha(result['authorization_raw_utf8'].encode('utf-8')) == AUTH_SHA,
            'Authorization text digest mismatch')
    initial = result.get('initial_owned', [])
    stopped = result.get('stopped_by_this_helper', [])
    require(len(initial) == len(stopped) == 188, 'Captured identity count mismatch')
    historical = {r['pid']: r for r in initial}
    require(len(historical) == 188 and {identity(r) for r in initial} ==
            {identity(r) for r in stopped}, 'Historical stopped identity mismatch')
    require(all(r['uid'] == helper.UID for r in initial) and
            not (set(historical) & set(helper.PROTECTED)), 'Foreign/protected captured target')
    events = result.get('signal_events', [])
    require(len(events) == 564, 'Original signal count is not 564')
    for index, signal_name in enumerate(('SIGSTOP', 'SIGTERM', 'SIGCONT')):
        block = events[index * 188:(index + 1) * 188]
        require({identity(e) for e in block} == {identity(r) for r in initial} and
                len({e['pid'] for e in block}) == 188 and
                all(e.get('signal') == signal_name for e in block),
                'Original signal identity/order mismatch')
        for event in block:
            require(all(re.fullmatch('[0-9a-f]{64}', event.get(k, ''))
                        for k in ('argv_sha256', 'pidfd_bound_actual_argv_sha256',
                                  'signal_observed_actual_argv_sha256')),
                    'Missing actual signal argv digests')
    require(preservation == result.get('preservation'), 'Preservation body differs from result')
    require(preservation.get('checkpoint_weights_only') is True and
            preservation.get('complete_Adam_resume_state') is False,
            'Preservation overclaims complete resume state')
    require(quiesced.get('status') == 'QUIESCED_PRESERVING' and
            quiesced.get('accepted') is False and quiesced.get('signal_events') == events[:188],
            'Quiesced evidence/prefix mismatch')
    for key in (*expected, 'snapshot_sha256', 'initial_owned', 'protected',
                'authorization_raw_utf8', 'authorization_observed_utc'):
        require(quiesced.get(key) == result.get(key), 'Quiesced binding mismatch: ' + key)
    return historical


def safe_leaf(root, relative):
    part = Path(relative)
    require(isinstance(relative, str) and relative and not part.is_absolute() and
            '..' not in part.parts and '.' not in part.parts, 'Unsafe evidence member')
    target = root / part
    require(not root.is_symlink(), 'Evidence root is a symlink')
    for ancestor in target.parents:
        if ancestor == root:
            break
        require(not ancestor.is_symlink(), 'Evidence member has symlink ancestor')
    return target


def leaves(root):
    require(root.is_dir() and not root.is_symlink(), 'Missing/non-directory evidence scope')
    return {str(p.relative_to(root)) for p in root.rglob('*') if p.is_symlink() or p.is_file()}


def checkpoint_metadata(checkpoints, base):
    require(len(checkpoints) == 3 and {c['step'] for c in checkpoints} == {50, 100, 150},
            'Unexpected original checkpoint inventory')
    observed = []
    for cp in checkpoints:
        root = base / 'runs/archive/round-1-grpo/actor' / ('global_step_' + str(cp['step']))
        require(cp['path'] == str(root), 'Checkpoint path changed')
        records = cp['files']
        require(len(records) == 12 and len({r['relative_path'] for r in records}) == 12 and
                leaves(root) == {r['relative_path'] for r in records}, 'Checkpoint member names changed')
        for record in records:
            p = safe_leaf(root, record['relative_path'])
            require(not p.is_symlink() and p.is_file(), 'Checkpoint member type changed')
            st = p.stat()
            require(st.st_size == record['bytes'] and st.st_mtime_ns == record['mtime_ns'],
                    'Checkpoint size/mtime continuity failed: ' + str(p))
            require(re.fullmatch('[0-9a-f]{64}', record.get('sha256', '')) is not None,
                    'Missing previously recorded checkpoint SHA')
        observed.append({'step': cp['step'], 'members': len(records),
                         'metadata_continuity': True, 'fresh_payload_sha': False})
    return {'checkpoints': observed, 'members': 36, 'prior_payload_sha_bound': True,
            'fresh_payload_sha': False, 'tensor_finite_checked': False,
            'Adam_or_RNG_captured': False}


def copied_evidence(entries, base, destination):
    require(len(entries) == len(COPY_SCOPES) and
            {e['path'] for e in entries} == {str(base / p) for p in COPY_SCOPES},
            'Missing/unexpected copied evidence scopes')
    count = total = symlinks = 0
    for entry in entries:
        relative = Path(entry['path']).relative_to(base)
        root = destination / 'copies' / relative
        require(entry.get('copy') == str(root) and not entry.get('absent'),
                'Missing/wrong preserved copy')
        records = entry['files']
        names = {r['relative_path'] for r in records}
        require(len(names) == len(records), 'Duplicate copy member')
        single = str(relative) == COPY_SCOPES[-1]
        actual = {root.name} if single and (root.is_file() or root.is_symlink()) else leaves(root)
        require(actual == names, 'Copied member names changed')
        for record in records:
            p = safe_leaf(root.parent if single else root, record['relative_path'])
            if 'symlink' in record:
                require(p.is_symlink() and os.readlink(p) == record['symlink'],
                        'Copied symlink changed')
                symlinks += 1
                continue
            require(p.is_file() and not p.is_symlink(), 'Copied file type changed')
            before = p.stat()
            require(before.st_size == record['bytes'] and file_sha(p) == record['sha256'],
                    'Copied file payload mismatch: ' + str(p))
            after = p.stat()
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                    'Copied file changed during fresh SHA read')
            count += 1
            total += after.st_size
    return {'regular_files_fresh_sha': count, 'regular_bytes_fresh_sha': total,
            'symlinks_literal_target_verified': symlinks, 'all_scopes_verified': True}


def terminal_sample(helper, bound_records, historical):
    table = helper.process_table()
    helper.check_protected(table, bound_records)
    remaining, ended = helper.remaining_owned(table, bound_records, historical)
    require(not remaining and len(ended) == len(historical), 'Old captured identities still live')
    gpus, apps, raw = helper.checked_gpu()
    release = helper.check_gpu_ownership(gpus, apps, historical, released=True)
    helper.check_protected(helper.process_table(), bound_records)
    return {'historical_identity_terminal_checks': ended, 'released_gpu': release,
            'gpu_raw': raw, 'protected_pids': sorted(helper.PROTECTED)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', default=str(Path(__file__).with_name('archive_stop.py')))
    parser.add_argument('--result', required=True)
    for flag in ('result-sha256', 'preservation-sha256', 'quiesced-sha256', 'authorization-sha256'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv)
    require((args.result_sha256, args.preservation_sha256, args.quiesced_sha256,
             args.authorization_sha256) == (RESULT_SHA, PRESERVATION_SHA, QUIESCED_SHA, AUTH_SHA),
            'This verifier is bound to the one reviewed stop receipt')
    helper = load_helper(args.helper)
    require(os.geteuid() == helper.UID and os.uname().nodename == helper.HOST, 'Wrong host/effective UID')
    destination = helper.BASE / 'preserved' / DESTINATION
    require(Path(args.result) == destination / 'result.private.json', 'Wrong original result path')
    inputs = {Path(args.result): RESULT_SHA, destination / 'preservation.private.json': PRESERVATION_SHA,
              destination / 'quiesced.private.json': QUIESCED_SHA}
    loaded = {path.name: json.loads(bound(path, digest)) for path, digest in inputs.items()}
    result, preservation, quiesced = (loaded[n] for n in ('result.private.json', 'preservation.private.json',
                                                        'quiesced.private.json'))
    historical = validate_receipts(result, preservation, quiesced, helper)
    snapshot_path = destination / 'input-snapshot.private.json'
    snapshot = json.loads(bound(snapshot_path, result['snapshot_sha256']))
    bound(destination / 'authorization.private.txt', AUTH_SHA)
    require(snapshot.get('base') == str(helper.BASE) and snapshot.get('hostname') == helper.HOST and
            snapshot.get('uid') == helper.UID, 'Original snapshot host/base/UID differs')
    records = {r['pid']: r for r in snapshot['owned_processes']}
    for pid, tick in helper.SUPERVISORS.items():
        require(records.get(pid, {}).get('starttick') == tick and records[pid]['uid'] == helper.UID,
                'Original supervisor identity missing')
    for pid, tick in {**helper.ANCHORS, **helper.PROTECTED}.items():
        require(records.get(pid, {}).get('starttick') == tick and records[pid]['uid'] == helper.UID,
                'Original anchor/protected identity missing')
    require(result['protected'] == [records[pid] for pid in helper.PROTECTED],
            'Protected receipt differs from original snapshot')
    before = terminal_sample(helper, records, historical)
    cp = checkpoint_metadata(preservation['checkpoints'], helper.BASE)
    copied = copied_evidence(preservation['copied_evidence'], helper.BASE, destination)
    after = terminal_sample(helper, records, historical)
    # Metadata again after copy hashing; original model payload is never opened.
    require(checkpoint_metadata(preservation['checkpoints'], helper.BASE) == cp,
            'Checkpoint metadata changed during terminal audit')
    for path, digest in inputs.items():
        bound(path, digest)
    bound(snapshot_path, result['snapshot_sha256'])
    bound(destination / 'authorization.private.txt', AUTH_SHA)
    require(file_sha(args.helper) == HELPER_SHA, 'Helper source changed during audit')
    print(json.dumps({'schema': 'k4-archive-stop-independent-terminal-v1', 'accepted': True,
                      'status': 'INDEPENDENT_STOPPED_PRESERVED_RELEASED', 'observed_unix': time.time(),
                      'verifier_source_sha256': file_sha(__file__), 'helper_source_sha256': HELPER_SHA,
                      'original_result_sha256': RESULT_SHA, 'preservation_sha256': PRESERVATION_SHA,
                      'quiesced_sha256': QUIESCED_SHA, 'authorization_sha256': AUTH_SHA,
                      'original_status_unchanged': result['status'], 'new_signals_sent': 0,
                      'Node_writes': 0, 'before': before, 'after': after,
                      'checkpoint_metadata': cp, 'copied_evidence': copied,
                      'complete_Adam_resume_state': False}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'accepted': False, 'status': 'HOLD_INDEPENDENT_TERMINAL_AUDIT',
                          'error_type': type(exc).__name__, 'error': str(exc),
                          'new_signals_sent': 0, 'Node_writes': 0}, sort_keys=True), flush=True)
        raise SystemExit(2)
