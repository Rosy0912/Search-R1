"""Same-HKUST independent closed-raw copies; never export payloads to Mac."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time

BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
DRIVER = BASE / 'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs'
ORDER = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
OP_SHA = '34857f4a6d0cfea4da2e2c4b71a5af662aca9a49e7ee7134ab97cd3c88185222'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def metadata(s):
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mode=s.st_mode,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def ref(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and all(not p.is_symlink() for p in (path, *path.parents)), 'Canonical nonsymlink file required')
    before = metadata(path.stat())
    require(stat.S_ISREG(before['mode']), 'Regular file required')
    with path.open('rb') as handle:
        raw = handle.read()
        require(metadata(os.fstat(handle.fileno())) == before, 'Opened small document changed')
    require(metadata(path.stat()) == before, 'Small document changed')
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), **before), raw


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
    with Path(path).open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return ref(path)[0]


def copied(path, destination, relative):
    before = metadata(path.stat())
    require(stat.S_ISREG(before['mode']) and not path.is_symlink(), 'Copy source must be regular')
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as source, destination.open('xb') as output:
        require(metadata(os.fstat(source.fileno())) == before, 'Copy source changed before open')
        while True:
            block = source.read(1 << 20)
            if not block:
                break
            digest.update(block)
            count += len(block)
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
        require(metadata(os.fstat(source.fileno())) == before, 'Copy source changed while reading')
    require(count == before['bytes'] and metadata(path.stat()) == before, 'Copy source changed after reading')
    expected = digest.hexdigest()
    local_before = metadata(destination.stat())
    require(local_before['inode'] != before['inode'] or local_before['device'] != before['device'], 'Independent copy required, not a hardlink')
    digest = hashlib.sha256()
    checked = 0
    with destination.open('rb') as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            checked += len(block)
            digest.update(block)
        require(metadata(os.fstat(handle.fileno())) == local_before, 'Destination changed while verifying')
    require(digest.hexdigest() == expected and checked == count and metadata(destination.stat()) == local_before,
            'Independent destination SHA/metadata differs')
    return (dict(path=str(path), relative=relative, sha256=expected, **before),
            dict(path=str(destination), relative=relative, sha256=expected, bytes=count, metadata=local_before))


def scope(op, variant):
    group = BASE / ('group-' + variant)
    cfg_ref, raw = ref(group / 'receipts/authorized-launch-20261009/config.json')
    config = json.loads(raw)
    status_ref, raw = ref(group / 'runs/status.json')
    status = json.loads(raw)
    require(status['phase'] == 'TRAIN_COMPLETE' and status['outer_iterations'] == 10,
            'Only a genuinely closed ten-outer writer may be copied')
    index = ORDER.index(variant)
    terminal_ref, _ = ref(DRIVER / ('%02d-%s' % (index, variant)) / 'training/group-terminal.json')
    outers = []
    for index in range(10):
        complete_ref, raw = ref(group / 'runs/outers' / ('outer-%05d' % index) / 'training/outer-complete.json')
        complete = json.loads(raw)
        outers.append(dict(outer_index=index, optimizer_updates=complete['optimizer_updates'],
                           completion=complete_ref, checkpoint=complete['checkpoint']))
    fixed = [cfg_ref, status_ref]
    for path in (group / 'runs/child-history.jsonl', group / 'runs/run-identity.json'):
        fixed.append(ref(path)[0])
    for root in (group / 'runs/phase-exits', Path(terminal_ref['path']).parent, group / 'receipts/dtype-recovery'):
        if root.exists():
            for path in sorted(root.rglob('*')):
                require(not path.is_symlink(), 'No symlinks in closure evidence')
                if path.is_file():
                    fixed.append(ref(path)[0])
    fixed = list({r['path']: r for r in fixed}.values())
    value = dict(schema='opsd-ablation-closed-raw-scope-v1', accepted=True, cluster='HKUST', variant=variant,
                 run_id=config['run_id'], output_root=config['paths']['output_root'], config=cfg_ref,
                 closed_outers=list(range(10)), outers=outers, status=status_ref, group_terminal=terminal_ref,
                 global_files=fixed, include_outer_raw=True)
    op['scope_shape'](value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', required=True)
    parser.add_argument('--source-operation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(os.getuid() == os.geteuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'HKUST1 original CPU-only UID required')
    selected = args.groups.split(',')
    require(selected and len(set(selected)) == len(selected) and selected == [g for g in ORDER if g in selected], 'Ordered explicit group subset required')
    op_path = Path(args.source_operation)
    source_ref, raw = ref(op_path)
    require(source_ref['sha256'] == OP_SHA and BASE in op_path.parents, 'Exact already-published original scope helper required')
    op = dict(__name__='_remote_closed_raw_scope', __file__=str(op_path))
    exec(compile(raw, str(op_path), 'exec'), op)
    transport = op['primitives']()
    output = Path(args.output)
    require(output.is_absolute() and output.resolve().is_relative_to(BASE / 'analysis') and not output.exists(),
            'New independent HKUST analysis/snapshot output required')
    for p in (output, *output.parents):
        require(not p.is_symlink(), 'No destination symlink or Mac mount alias')
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    started = time.time()
    for variant in selected:
        original = scope(op, variant)
        closure = op['closure'](transport, original)
        paths, excluded = op['enumerate_files'](transport, original, closure)
        size = sum(path.stat().st_size for path in paths)
        require(shutil.disk_usage(output).free >= size + (10 << 30), 'Remote copy capacity below payload plus 10GiB')
        stage = output / variant
        stage.mkdir(exist_ok=False)
        destination = stage / 'raw'
        destination.mkdir(exist_ok=False)
        source_rows, local_rows = [], []
        for index, path in enumerate(paths):
            relative = path.relative_to(BASE).as_posix()
            original_row, local_row = copied(path, destination / relative, relative)
            source_rows.append(original_row)
            local_rows.append(local_row)
            if index % 500 == 0:
                print(json.dumps(dict(event='remote_copy_progress', variant=variant, files=index + 1, total_files=len(paths))), flush=True)
        require(closure == op['closure'](transport, original) and (paths, excluded) == op['enumerate_files'](transport, original, closure),
                'Original closed scope changed during copy')
        for row in source_rows:
            require(metadata(Path(row['path']).stat()) == {k:row[k] for k in metadata(Path(row['path']).stat())}, 'Source metadata changed after copy')
        total = sum(row['bytes'] for row in source_rows)
        source = dict(accepted=True, all_source_pinned_metadata_stable=True, run_id=original['run_id'], variant=variant,
                      source_root=str(BASE), files=source_rows, file_count=len(source_rows), total_bytes=total,
                      excluded_subtrees=excluded, source_scope=original, original_closure=closure, source_operation=source_ref)
        local = dict(accepted=True, exact_inventory=True, local_metadata_stable=True, no_symlinks_or_special_files=True,
                     source_sha_matched_all_files=True, destination=str(destination), files=local_rows,
                     file_count=len(local_rows), total_bytes=total, storage_cluster='HKUST', mac_payload_copy=False)
        source_proof = save(stage / 'source-inventory.json', source)
        local_proof = save(stage / 'copy-accepted.json', local)
        result = dict(accepted=True, variant=variant, run_id=original['run_id'], config=original['config'],
                      source=source_proof, local=local_proof, backups=[dict(kind='direct', source=source_proof, local=local_proof)],
                      files=len(paths), bytes=total, independent_inode_copy=True, node_originals_retained=True,
                      no_GPU_or_Torch=True, no_training_lock=True, checked_unix=time.time())
        results[variant] = save(stage / 'complete.json', result)
        print(json.dumps(dict(event='remote_copy_complete', variant=variant, files=len(paths), bytes=total, complete=results[variant])), flush=True)
    complete = save(output / 'complete.json', dict(schema='opsd-ablation-remote-raw-copy-v1', accepted=True, groups=results,
                    source_operation=source_ref, started_unix=started, ended_unix=time.time(),
                    Mac_raw_payloads=False, scientific_source_changed=False, original_files_deleted=False))
    print(json.dumps(dict(event='all_requested_remote_copies_complete', accepted=True, groups=list(results), complete=complete)), flush=True)


if __name__ == '__main__':
    main()
