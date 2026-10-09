#!/usr/bin/env python3
"""Read-only, final-only original Node closure/NPY inventory; stdlib only."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

STUDY_ROOT = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
NODE_UID = 1005
VARIANTS = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
META = ('device', 'inode', 'bytes', 'mode', 'nlink', 'mtime_ns', 'ctime_ns')
SOURCE_META = ('device', 'inode', 'bytes', 'mode', 'mtime_ns', 'ctime_ns')
SMALL_LIMIT = 2 << 20
LEDGER_LIMIT = 32 << 20
REQUEST_LIMIT = 64 << 20


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def zero(value):
    return type(value) is int and value == 0


def json_value(raw):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=unique)


def canonical(value):
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts and str(p) == value, 'Noncanonical path')
    return p


def node_path(value):
    p = canonical(value)
    require(p == STUDY_ROOT or STUDY_ROOT in p.parents, 'Path outside fixed study root')
    for q in (p, *p.parents):
        require(not q.is_symlink(), 'Symlink in original path: ' + str(q))
    return p


def metadata(s):
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mode=s.st_mode,
                nlink=s.st_nlink, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def current(path):
    p = node_path(str(path)); s = p.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink > 0, 'Regular original file required')
    return metadata(s)


def reference(ref):
    canonical(ref['path'])
    require(type(ref['bytes']) is int and ref['bytes'] > 0 and
            isinstance(ref['sha256'], str) and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Invalid SHA/byte pin')
    return ref


def same_ref(a, b):
    reference(a); reference(b)
    return all(a[k] == b[k] for k in ('path', 'bytes', 'sha256'))


def read_regular(path, limit):
    p = node_path(str(path)); before = current(p)
    require(before['bytes'] <= limit, 'Small read exceeds explicit limit')
    fd = os.open(str(p), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        require(metadata(os.fstat(f.fileno())) == before, 'Opened file metadata changed')
        raw = f.read(limit + 1)
        require(metadata(os.fstat(f.fileno())) == before, 'File changed during read')
    require(len(raw) == before['bytes'] and current(p) == before, 'Path changed during read')
    return raw, before


class Documents:
    def __init__(self):
        self.items = {}

    def read(self, ref):
        reference(ref); path = ref['path']
        if path in self.items:
            require(same_ref(ref, self.items[path]['original']), 'Conflicting original document pins')
            return self.items[path]['value']
        raw, before = read_regular(path, SMALL_LIMIT)
        require(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Closure SHA/bytes differ')
        require(all(ref[k] == before[k] for k in META if k in ref), 'Pinned closure metadata differs')
        original = dict(before, path=path, sha256=ref['sha256'])
        self.items[path] = dict(original=original, content_base64=base64.b64encode(raw).decode(), value=json_value(raw))
        return self.items[path]['value']

    def stable(self):
        require(all(current(p) == {k: d['original'][k] for k in META} for p, d in self.items.items()), 'Closure changed during scan')


def owner_identity(x):
    require(all(type(x[k]) is int and x[k] > 0 for k in ('pid', 'starttick')) and
            x['uid'] == NODE_UID and isinstance(x['run_id'], str) and
            isinstance(x['argv'], list) and x['argv'] and not x.get('capture_status'), 'Unknown original controller identity')
    return tuple(x[k] for k in ('pid', 'uid', 'starttick', 'run_id')) + (tuple(x['argv']),)


def release_audit(value, run, owner=None):
    require(value['accepted'] is True and value['no_signals_sent'] is True, 'Release not accepted/no-signal')
    a = value['audit']
    require(a['schema'] == 'same-uid-owned-process-audit-v2' and a['run_id'] == run and a['uid'] == NODE_UID and
            a['accepted_no_owned_or_opaque'] is True and a['blockers'] == [] and
            a['no_exit_code_inferred_from_absence'] is True, 'Original release audit is not clean')
    if owner is not None:
        require(owner_identity(owner) in {owner_identity(x) for x in a['known_owned']}, 'Original controller release anchor missing')
    require(type(a['ended_unix']) in (float, int) and 0 < a['ended_unix'] <= time.time(), 'Invalid release time')
    return a['ended_unix']


def closed_group(g, docs, study_id):
    variant, run = g['variant'], g['run_id']; root = STUDY_ROOT / ('group-' + variant) / 'runs'
    require(g['status']['path'] == str(root / 'status.json'), 'Wrong original status path')
    c = docs.read(g['config'])
    require(c['schema'] == 'opsd-ablation-config-v1' and c['variant'] == variant and c['run_id'] == run and
            c['study_id'] == study_id and c['scientific']['outer_iterations'] == 10 and
            c['paths']['output_root'] == str(root), 'Original ten-outer run/config differs')
    contract = dict(direction='student_to_target_reverse', target_version='opsd-ablation-reverse-v1',
                    variant=variant, precision=c['precision']['mode'], mixture_alpha=.1 if variant in ('M', 'J') else 0.)
    require(len(g['outers']) == 10, 'Exactly ten original completion refs required')
    previous = None
    for index, ref in enumerate(g['outers']):
        require(ref['path'] == str(root / 'outers' / ('outer-%05d' % index) / 'training/outer-complete.json'), 'Wrong ordered outer path')
        row = docs.read(ref)
        require(row['schema'] == 'opsd-online-outer-complete-v1' and row['accepted'] is True and
                row['run_id'] == run and row['outer_index'] == index and row['contract'] == contract, 'Original outer is not complete')
        node_path(row['checkpoint']['path']); reference(row['checkpoint'])
        require(root / 'checkpoints' in Path(row['checkpoint']['path']).parents, 'Checkpoint ref outside this run')
        if previous is not None:
            require(same_ref(row['outer_old_adapter'], previous), 'Original outer-old checkpoint chain differs')
        previous = row['checkpoint']
    s = docs.read(g['status']); t = docs.read(g['terminal']); owner = s['owner']
    require(owner['run_id'] == run, 'Status owner run differs'); owner_identity(owner)
    require(s['schema'] == 'opsd-ablation-controller-status-v1' and s['phase'] == 'TRAIN_COMPLETE' and
            s['outer_iterations'] == 10 and s['variant'] == variant and s['run_id'] == run and
            s['contract'] == contract and s['config_sha256'] == g['config']['sha256'] and s['actual_optimizer_updates'] == row['optimizer_updates'] and
            same_ref(s['latest_checkpoint'], previous), 'Final original TRAIN_COMPLETE10 missing')
    require(t['schema'] == 'opsd-ablation-group-foreground-terminal-v1' and t['accepted'] is True and
            zero(t['actual_exit_code']) and t['no_signals'] is True and t['signal_termination_not_exit0'] is False and
            t['run_id'] == run and t['config_sha256'] == g['config']['sha256'] and same_ref(t['status'], g['status']) and
            owner_identity(t['controller']) == owner_identity(owner), 'Original foreground terminal rc0 missing')
    wait = docs.read(t['actual_child_exit']); release = docs.read(t['actual_release'])
    require(zero(wait['actual_wait_returncode']) and wait['run_id'] == run and wait['no_signals'] is True and
            wait['signal_termination_not_exit0'] is False and owner_identity(wait['identity']) == owner_identity(owner), 'Actual original parent wait0 missing')
    own = s['resource_release']; release_audit(own['all_owned_release'], run)
    require(own['all_owned_release']['audit']['controller_starttick'] == owner['starttick'] and
            own['resource_admission']['owned_gpu_processes'] == [], 'Native original release differs')
    require(release['accepted'] is True and release['after']['accepted'] is True and release['after']['phase'] == 'cpu-merge' and
            release['after']['run_id'] == run and release['after']['owned_gpu_processes'] == [], 'External original release missing')
    ended = max(release_audit(release['owned'], run, owner), release['after']['checked_unix'])
    require(type(ended) in (int, float) and 0 < ended <= time.time(), 'Invalid external release time')
    return c, dict(variant=variant, run_id=run, outers=10, released_unix=ended, controller=owner)


def accepted_sources(items, roots, groups):
    sources = {}
    for item in items:
        ref = reference(item['reference']); raw = base64.b64decode(item['content_base64'], validate=True)
        require(len(raw) <= REQUEST_LIMIT and len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Accepted inventory embedded SHA/bytes differ')
        inv = json_value(raw)
        require(isinstance(inv['files'], list), 'Source inventory files required')
        g = next((g for g in groups if g['variant'] == item['variant']), None)
        require(g is not None and item['run_id'] == g['run_id'] and
                item['config_sha256'] == g['config']['sha256'], 'Accepted inventory original group identity differs')
        for k in ('variant', 'run_id'):
            require(k not in inv or inv[k] == item[k], 'Source inventory run/variant differs')
        require('config' not in inv or inv['config']['sha256'] == item['config_sha256'], 'Source inventory config differs')
        acceptance = None
        if inv.get('accepted') is not True:
            # Published generic raw inventories are accepted by a separate
            # lossless-backup receipt, not an invented inventory accepted flag.
            a = item['acceptance']; ar = reference(a['reference'])
            ab = base64.b64decode(a['content_base64'], validate=True)
            require(len(ab) == ar['bytes'] and hashlib.sha256(ab).hexdigest() == ar['sha256'], 'Acceptance receipt embedded SHA/bytes differ')
            av = json_value(ab)
            require(inv['schema'] == 'opsd-ablation-closed-raw-inventory-v1' and
                    av['schema'] == 'opsd-ablation-closed-raw-compressed-local-backup-v1' and av['accepted'] is True and
                    av['inventory_sha256'] == ref['sha256'] and zero(av['actual_ssh_returncode']) and
                    av['all_local_decompressed_member_SHA_match'] is True and av['original_NPY_bytes_not_reserialized'] is True and
                    av['run_id'] == item['run_id'] and av['variant'] == item['variant'] and
                    av['config']['sha256'] == item['config_sha256'], 'Source inventory separate acceptance is missing/different')
            acceptance = ar
        for f in inv['files']:
            if not f['path'].endswith('.npy'):
                continue
            p = canonical(f['path']); group_root = roots[VARIANTS.index(item['variant'])]
            require(group_root in p.parents, 'Accepted NPY outside its original group root')
            reference(f)
            require(all(type(f.get(k)) is int for k in SOURCE_META), 'Unknown accepted source metadata')
            origin = dict(kind='accepted_source_inventory', inventory=ref, acceptance=acceptance,
                          variant=item['variant'], run_id=item['run_id'], config_sha256=item['config_sha256'], source_file=f)
            if f['path'] in sources:
                require(all(sources[f['path']]['file'][k] == f[k] for k in (*SOURCE_META, 'sha256')), 'Conflicting accepted source inventories')
            else:
                sources[f['path']] = dict(file=f, origin=origin)
    return sources


def enumerate_npy(roots):
    files, directories = {}, {}
    for root in roots:
        stack = [root]
        while stack:
            p = node_path(str(stack.pop())); st = p.lstat()
            require(stat.S_ISDIR(st.st_mode), 'Original scan root/directory missing')
            directories[str(p)] = metadata(st)
            with os.scandir(str(p)) as entries:
                for entry in sorted(entries, key=lambda e: e.name):
                    path = p / entry.name; s = entry.stat(follow_symlinks=False)
                    require(not stat.S_ISLNK(s.st_mode), 'Symlink in scan tree: ' + str(path))
                    if stat.S_ISDIR(s.st_mode):
                        stack.append(path)
                    else:
                        require(stat.S_ISREG(s.st_mode), 'Unknown file type in scan tree')
                        if entry.name.endswith('.npy'):
                            files[str(path)] = metadata(s)
    return files, directories


def hash_npy(path, before):
    require(current(path) == before, 'NPY changed before fresh hash')
    h = hashlib.sha256(); count = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        require(metadata(os.fstat(f.fileno())) == before, 'NPY opened metadata differs')
        while True:
            raw = f.read(1 << 20)
            if not raw:
                break
            h.update(raw); count += len(raw)
        require(metadata(os.fstat(f.fileno())) == before, 'NPY changed while hashing')
    require(count == before['bytes'] and current(path) == before, 'NPY changed after fresh hash')
    return h.hexdigest()


def collect(request):
    require(os.getuid() == NODE_UID, 'Read-only collector requires original Node UID1005')
    require(request['schema'] == 'opsd-ablation-final-vocab-collection-inputs-v1' and
            request['study_root'] == str(STUDY_ROOT), 'Wrong fixed study collection scope')
    groups = request['groups']; require(tuple(g['variant'] for g in groups) == VARIANTS, 'Seven ordered groups required')
    require(len({g['run_id'] for g in groups}) == 7, 'Seven distinct original run IDs required')
    docs = Documents(); configs, closed = [], []
    for g in groups:
        c, closure = closed_group(g, docs, request['study_id']); configs.append(c); closed.append(closure)
    for field in ('source_model_manifest', 'runtime_manifest', 'common_initial_adapter_manifest', 'common_cohort_manifest', 'helper_source_manifest'):
        require(len({c[field]['sha256'] for c in configs}) == 1, 'Different original common identity: ' + field)
    shared = {c['logging']['full_vocab_study_root'] for c in configs}
    require(len(shared) == 1, 'Different original shared ledger roots')
    ledger_path = node_path(str(Path(shared.pop()) / 'ledger.json'))
    roots = [node_path(c['paths']['output_root'] + '/outers') for c in configs]
    sources = accepted_sources(request['accepted_source_inventories'], roots, groups)
    started = time.time(); require(started >= max(c['released_unix'] for c in closed), 'Collection precedes actual release')
    ledger_raw, ledger_before = read_regular(ledger_path, LEDGER_LIMIT); json_value(ledger_raw)
    ledger_ref = dict(ledger_before, path=str(ledger_path), sha256=hashlib.sha256(ledger_raw).hexdigest())
    first_files, first_dirs = enumerate_npy(roots); files = []
    require(set(sources).issubset(first_files), 'Accepted original NPY missing from final scan')
    for path, before in sorted(first_files.items()):
        old = sources.get(path)
        if old is not None:
            require(all(old['file'][k] == before[k] for k in SOURCE_META) and
                    ('nlink' not in old['file'] or old['file']['nlink'] == before['nlink']), 'Accepted original NPY metadata changed')
            sha, origin = old['file']['sha256'], old['origin']
        else:
            sha, origin = hash_npy(path, before), dict(kind='fresh_sha256_readback')
        files.append(dict(before, path=path, sha256=sha, sha256_origin=origin))
    second_files, second_dirs = enumerate_npy(roots)
    require(first_files == second_files and first_dirs == second_dirs, 'NPY/tree changed between enumerations')
    ledger_after_raw, ledger_after = read_regular(ledger_path, LEDGER_LIMIT)
    require(ledger_raw == ledger_after_raw and ledger_before == ledger_after, 'Shared ledger bytes/stat changed')
    docs.stable()
    return dict(schema='opsd-ablation-final-vocab-physical-inventory-v1', accepted=True,
                actual_read_returncode=0, started_unix=started, ended_unix=time.time(),
                source_metadata_stable=True, scanned_roots=[str(r) for r in roots],
                run_ids=[g['run_id'] for g in groups], ledger=ledger_ref,
                ledger_source_before=ledger_before, ledger_source_after=ledger_after,
                ledger_content_base64=base64.b64encode(ledger_raw).decode(), files=files,
                closures=closed, collection_documents=[{k: v for k, v in d.items() if k != 'value'} for d in docs.items.values()],
                sha_reuse_count=sum(f['sha256_origin']['kind'] == 'accepted_source_inventory' for f in files),
                fresh_sha_count=sum(f['sha256_origin']['kind'] == 'fresh_sha256_readback' for f in files),
                finite_checked=False, scope='Closed original file inventory only; no finite, numerical, GPU-resume or scientific acceptance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request-sha256', required=True); p.add_argument('--source-sha256', required=True)
    args = p.parse_args()
    try:
        raw = sys.stdin.buffer.read(REQUEST_LIMIT + 1)
        require(len(raw) <= REQUEST_LIMIT and hashlib.sha256(raw).hexdigest() == args.request_sha256, 'Request SHA/size differs')
        source_raw, source_meta = read_regular(Path(__file__).absolute(), SMALL_LIMIT)
        require(hashlib.sha256(source_raw).hexdigest() == args.source_sha256, 'Published collector source SHA differs')
        result = collect(json_value(raw))
        require(current(Path(__file__).absolute()) == source_meta, 'Collector source changed')
        result.update(request_sha256=args.request_sha256, collector_source=dict(source_meta, path=str(Path(__file__).absolute()), sha256=args.source_sha256))
        rc = 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(schema='opsd-ablation-final-vocab-collection-hold-v1', accepted=False,
                      actual_read_returncode=2, status='HOLD', reason=str(error), error_type=type(error).__name__, finite_checked=False)
        rc = 2
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n')
    return rc


if __name__ == '__main__':
    sys.exit(main())
