#!/usr/bin/env python3
"""Publish the already verified r2 replica; never recopies or loads native files.

NFS4 rejected renameat2(RENAME_NOREPLACE). An exclusive empty directory reservation
plus the same own-writer flock permits ordinary atomic rename without replacing
any previously existing destination. Original failed status/journal are preserved.
"""
import fcntl
import calendar
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import time

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008')
CONTROL = ROOT / '.shared-output-backup-20261008'
SHARED = Path('/mnt/shared/workspace/h100-11_from_h100-9/shaoshitong/search-r1-rl/tree-grpo')
DEST = SHARED / 'retained18-backup-20261008'
ORIGINAL = 'retained18-native-once-20261008-r2'
RECOVERY = 'retained18-native-finalize-r2-20261008'
ENTRY_SHA = '1721d96d21df5a2f36dfa0793a1c099df0ddd31722ebe10ffeb207ecc9e21c22'
MANIFEST_SHA = 'fb4bdd43440f6452ec50307b387df443880ca6932f1048461462d03f58d708fb'
INVENTORY_SHA = 'a9be94dc8d4de276a6baa9c09851b4dcfecd35b2128ca9929b0ebc50cbe81345'
PARTIAL = DEST / ('.partial-' + ORIGINAL)
FINAL = DEST / 'native-3341-step-13-fb4bdd43440f6452'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def stat_record(path):
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and not path.is_symlink(), 'Native target must be regular')
    return dict(name=path.name, bytes=s.st_size, inode=s.st_ino, device=s.st_dev,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, mode=s.st_mode, uid=s.st_uid)


def main():
    started = time.monotonic()
    started_cpu = time.process_time()
    require(socket.gethostname() == 'h100-10', 'Wrong host')
    for p in [CONTROL, SHARED, DEST, PARTIAL]:
        require(p.is_dir() and not p.is_symlink() and p.resolve(strict=True) == p, 'Exact task directory changed: ' + str(p))
    mount = subprocess.check_output(['findmnt', '-n', '-t', 'nfs4', '-T', str(DEST), '-o', 'SOURCE,FSTYPE,TARGET'], text=True).strip()
    require(mount.split()[:2] == ['10.32.0.11:/dataset/from_h100-9', 'nfs4'], 'NFS11 identity changed')
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    subprocess.check_call(['ionice', '-c', '3', '-p', str(os.getpid())], timeout=10)
    lock = (CONTROL / 'native-once.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_entry = CONTROL / ('backup_native_once-' + ENTRY_SHA[:16] + '.py')
    require(sha(old_entry.read_bytes()) == ENTRY_SHA, 'Original verified copy entry changed')
    spec = importlib.util.spec_from_file_location('original_backup', old_entry)
    backup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backup)
    original_status_raw = backup.read_small(CONTROL / (ORIGINAL + '.status.json'))
    old = json.loads(original_status_raw)
    require(old['request_id'] == ORIGINAL and old['entry_sha256'] == ENTRY_SHA, 'Wrong original request')
    require(old['status'] == 'FAILED_OWN_BACKUP_ONLY' and old['error'] == "OSError(22, 'Invalid argument')", 'Unexpected failure boundary')
    require(not Path('/proc/' + str(old['pid'])).exists(), 'Original copy PID still exists: stop finalize')
    require(old['pinned_bytes'] == 0 and old['manifest_sha256'] == MANIFEST_SHA, 'Original state mismatch')
    require(old['partial'] == str(PARTIAL) and old['final'] == str(FINAL), 'Original publication paths changed')
    original_log_raw = backup.read_small(CONTROL / (ORIGINAL + '.log'))
    traceback = original_log_raw.decode().rsplit('Traceback (most recent call last):', 1)[-1]
    require('publish_no_replace(partial, final)' in traceback and 'Invalid argument' in traceback and 'line 420' in traceback, 'Traceback does not prove renameat2 failure')
    journal_path = CONTROL / (RECOVERY + '.request.json')
    require(not journal_path.exists(), 'Recovery already journalled: inspect existing result; never blindly retry')
    state = dict(request_id=RECOVERY, original_request_id=ORIGINAL, pid=os.getpid(),
                 proc_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 entry_sha256=sha(Path(__file__).read_bytes()), started_at=now(), status='FINALIZE_INTENT',
                 partial=str(PARTIAL), final=str(FINAL), native_manifest_sha256=MANIFEST_SHA,
                 new_native_copy_bytes=0, new_native_hash_read_bytes=0, training_unchanged=True,
                 original_failed_status_preserved=True, original_source_checkpoint_not_read=True)
    with journal_path.open('x') as f:
        json.dump(state, f, indent=2); f.flush(); os.fsync(f.fileno())
    backup.fsync_dir(CONTROL)
    status_path = CONTROL / (RECOVERY + '.status.json')
    def update(**values):
        state.update(values, observed_at=now())
        backup.atomic_json(status_path, state)
        print(json.dumps(state, sort_keys=True), flush=True)
    try:
        require(not FINAL.exists() and not FINAL.is_symlink(), 'Existing destination must never be replaced')
        evidence = CONTROL / (RECOVERY + '.evidence')
        evidence.mkdir()
        original_paths = [CONTROL / (ORIGINAL + n) for n in ['.status.json', '.launch.json', '.request.json', '.log']]
        original_paths += [PARTIAL / n for n in ['.complete', 'manifest.json', 'native-file-inventory.json', 'external-provenance-receipt.json']]
        evidence_files = []
        for p in original_paths:
            raw = backup.read_small(p)
            name = p.name if p.parent == CONTROL else 'partial-' + p.name.lstrip('.')
            backup.write_bytes(evidence / name, raw)
            evidence_files.append(dict(source=str(p), preserved_as=str(evidence / name), bytes=len(raw), sha256=sha(raw)))
        manifest_raw = backup.read_small(PARTIAL / 'manifest.json')
        require(sha(manifest_raw) == MANIFEST_SHA, 'Frozen original manifest hash failed')
        manifest = json.loads(manifest_raw)
        require(backup.identity_check(manifest['identity'], '3341') == 13, 'Native13 frozen identity failed')
        require(manifest['identity']['guard_continue_accounting']['completed_optimizer_calls_per_rank'] == {str(i):260 for i in range(8)}, 'Wrong native accounting')
        require(manifest['status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED', 'Original native manifest incomplete')
        inventory_raw = backup.read_small(PARTIAL / 'native-file-inventory.json')
        require(sha(inventory_raw) == INVENTORY_SHA, 'Frozen independently inspected native inventory hash failed')
        inventory = json.loads(inventory_raw)
        receipt_raw = backup.read_small(PARTIAL / '.complete')
        receipt = json.loads(receipt_raw)
        require(inventory['manifest_sha256'] == MANIFEST_SHA and inventory['original_manifest_preserved'] is True, 'Native inventory manifest mismatch')
        require(receipt['request_id'] == ORIGINAL and receipt['entry_sha256'] == ENTRY_SHA and receipt['manifest_sha256'] == MANIFEST_SHA,
                'Original prepublication verification receipt mismatch')
        require(receipt['all_8_ranks_and_driver_stream_and_readback_sha256_verified'] is True and receipt['native_identity'] == manifest['identity'],
                'Original full verification receipt incomplete')
        workers = manifest['worker_states']
        require(len(workers) == 8 and sorted(w['identity']['rank'] for w in workers) == list(range(8)), 'Eight distinct native workers required')
        expected = {}
        for w in workers:
            rank = w['identity']['rank']
            require(backup.identity_check(w['identity'], '3341') == 13 and w['identity']['world_size'] == 8 and w['scheduler_last_epoch'] == 13,
                    'Rank/scheduler identity mismatch')
            name = 'rank-%02d.pt' % rank
            require(Path(w['path']).name == name, 'Original rank filename mismatch')
            expected[name] = dict(sha256=w['sha256'], bytes=w['bytes'])
        require(manifest['driver']['next_step'] == 14 and Path(manifest['driver']['path']).name == 'driver.pt', 'Driver identity mismatch')
        expected['driver.pt'] = dict(sha256=manifest['driver']['sha256'])
        require(len(inventory['files']) == 9 and {f['name'] for f in inventory['files']} == set(expected), 'Inventory must cover exactly nine native files')
        completed_ns = calendar.timegm(time.strptime(receipt['completed_at'], '%Y-%m-%dT%H:%M:%SZ')) * 10**9
        target_stats = []
        for item in inventory['files']:
            e = expected[item['name']]
            require(item['full_destination_readback_verified'] is True and item['sha256'] == e['sha256'], 'Full destination verification not proven')
            if 'bytes' in e:
                require(item['bytes'] == e['bytes'], 'Native inventory size mismatch')
            actual = stat_record(PARTIAL / item['name'])
            require(actual['bytes'] == item['bytes'], 'Native destination size changed')
            require(actual['mtime_ns'] <= completed_ns and actual['ctime_ns'] <= completed_ns, 'Target changed after the original successful full SHA pass')
            target_stats.append(actual)
        total = sum(i['bytes'] for i in inventory['files'])
        require(total == 49376545288 and old['bytes_copied'] == total and old['bytes_readback'] == total and receipt['bytes_readback'] == total,
                'Original copy/readback byte accounting mismatch')
        require({p.name for p in PARTIAL.glob('*.pt')} == set(expected), 'Unexpected native asset in partial')
        proof = dict(request_id=RECOVERY, observed_at=now(), original_failure=old, original_traceback=traceback,
                     evidence_files=evidence_files, native_target_stats=target_stats, original_readback_bytes=total,
                     source_stream_and_full_target_readback_receipt_reused=True, new_native_hash_pass_performed=False,
                     original_verification_completed_at=receipt['completed_at'],
                     original_copy_and_readback_wall_seconds=calendar.timegm(time.strptime(receipt['completed_at'], '%Y-%m-%dT%H:%M:%SZ'))-calendar.timegm(time.strptime(old['started_at'], '%Y-%m-%dT%H:%M:%SZ')),
                     renameat2_noreplace_supported=False, directory_fsync_failure_observed=False,
                     directory_fsync_supported_by_original_success_and_recovery_calls=True,
                     original_inventory_sha256=sha(inventory_raw), prepublication_receipt_sha256=sha(receipt_raw))
        backup.atomic_json(evidence / 'recovery-proof.json', proof)
        backup.atomic_json(PARTIAL / 'recovery-publication-proof.json', proof)
        require(backup.read_small(CONTROL / (ORIGINAL + '.status.json')) == original_status_raw, 'Original failed status changed')
        for previous in target_stats:
            require(stat_record(PARTIAL / previous['name']) == previous, 'Native target changed during finalize validation')
        # Preserve the prepublication marker; completion will be written only after rename.
        archived = PARTIAL / '.prepublication-r2.complete.json'
        require(not archived.exists(), 'Prepublication marker archive already exists')
        os.rename(PARTIAL / '.complete', archived)
        backup.fsync_dir(PARTIAL)
        # Exclusive mkdir rejects every previously existing final path, including an empty one.
        FINAL.mkdir(mode=0o700)
        reserved = FINAL.lstat()
        require(stat.S_ISDIR(reserved.st_mode) and not FINAL.is_symlink() and not any(FINAL.iterdir()), 'Final reservation not empty/exclusive')
        backup.fsync_dir(DEST)
        update(status='DESTINATION_EXCLUSIVELY_RESERVED', reserved_final_inode=reserved.st_ino,
               reserved_final_device=reserved.st_dev, full_existing_verification_reused=True, verified_native_bytes=total)
        current = FINAL.lstat()
        require((current.st_ino, current.st_dev) == (reserved.st_ino, reserved.st_dev) and not any(FINAL.iterdir()), 'Reserved final path changed')
        # Under the same own-writer flock, replace only the empty directory made above.
        # NFS ordinary rename is atomic; no existing checkpoint/asset can be overwritten.
        os.rename(PARTIAL, FINAL)
        backup.fsync_dir(DEST)
        backup.fsync_dir(FINAL)
        update(status='ATOMIC_DIRECTORY_RENAME_SUCCEEDED', destination=str(FINAL))
        new_receipt = dict(receipt, status='COMPLETE_SHARED_NATIVE_BACKUP', publication_recovered=True,
                           original_backup_request_id=ORIGINAL, recovery_request_id=RECOVERY,
                           original_verification_completed_at=receipt['completed_at'], completed_at=now(),
                           original_failed_status_preserved=True, recovery_entry_sha256=state['entry_sha256'],
                           publication_strategy='same-own-writer-flock plus exclusive-empty-final-mkdir then ordinary atomic NFS rename',
                           original_full_native_sha_verification_reused=True, target_stat_revalidated=True,
                           new_native_copy_bytes=0, new_native_hash_read_bytes=0, source_checkpoint_not_reopened=True,
                           finalize_wall_seconds=time.monotonic()-started, finalize_cpu_seconds=time.process_time()-started_cpu,
                           directory_fsync_calls_succeeded=True, file_fsync_calls_succeeded=True,
                           server_power_failure_persistence_guaranteed=False, provider_retention_after_2026_10_21_guaranteed=False,
                           destination=str(FINAL), recovery_proof=str(FINAL / 'recovery-publication-proof.json'))
        backup.atomic_json(FINAL / '.complete', new_receipt)
        require(json.loads(backup.read_small(FINAL / '.complete')) == new_receipt, 'Published new completion receipt readback failed')
        backup.atomic_json(DEST / (RECOVERY + '.complete.json'), new_receipt)
        require(backup.read_small(CONTROL / (ORIGINAL + '.status.json')) == original_status_raw, 'Original failed history changed')
        update(status='COMPLETE_SHARED_NATIVE_BACKUP', completed_at=new_receipt['completed_at'],
               completion_receipt=str(FINAL / '.complete'), original_failure_status='FAILED_OWN_BACKUP_ONLY',
               directory_fsync_calls_succeeded=True, original_full_native_sha_verification_reused=True,
               verified_native_bytes=total, new_native_copy_bytes=0, new_native_hash_read_bytes=0)
    except BaseException as exc:
        update(status='FAILED_FINALIZE_ONLY', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
