"""Read-only closed seven-group raw inventory/USTAR/source-after entry."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time

sys.dont_write_bytecode = True
BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
HELPER = BASE / 'source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py'
HELPER_SHA = 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
SCHEMA = 'opsd-ablation-closed-raw-scope-v1'
INVENTORY = 'opsd-ablation-closed-raw-inventory-v1'
VARIANTS = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')


def require(value, message):
    if not value:
        raise ValueError(message)


def primitives(path=HELPER):
    """Compile the exact published bytes; never import a timestamp-valid pyc."""
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == HELPER_SHA, 'Transport dependency SHA differs')
    namespace = dict(__name__='_closed_raw_transport', __file__=str(path))
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace


def reference(value):
    require(isinstance(value, dict) and set(('path', 'sha256')) <= set(value), 'Missing exact reference')
    require(isinstance(value['path'], str) and Path(value['path']).is_absolute()
            and str(Path(value['path'])) == value['path'] and '..' not in Path(value['path']).parts,
            'Noncanonical reference path')
    require(re.fullmatch('[0-9a-f]{64}', value['sha256'] or ''), 'Missing reference SHA')


def same_ref(left, right):
    return all(left[k] == right[k] for k in ('path', 'sha256'))


def scope_shape(scope):
    require(scope.get('schema') == SCHEMA and scope.get('accepted') is True
            and scope.get('cluster') == 'HKUST' and scope.get('variant') in VARIANTS,
            'Not an exact authorized seven-group scope')
    require(re.fullmatch('[0-9a-f]{32}', scope.get('run_id', '')), 'Missing run identity')
    group = BASE / ('group-' + scope['variant'])
    require(scope['output_root'] == str(group / 'runs'), 'Wrong group output root')
    reference(scope['config'])
    require(scope['config']['path'] == str(group / 'receipts/authorized-launch-20261009/config.json'),
            'Wrong original configuration path')
    require(scope['closed_outers'] == list(range(10)) and len(scope['outers']) == 10,
            'All ten fixed closures required; partial scope is HOLD')
    require(type(scope.get('include_outer_raw', True)) is bool, 'Invalid outer inclusion flag')
    for index, row in enumerate(scope['outers']):
        require(row['outer_index'] == index and type(row['optimizer_updates']) is int
                and row['optimizer_updates'] >= 0, 'Invalid actual closure counters')
        for key in ('completion', 'checkpoint'):
            reference(row[key])
        require(row['completion']['path'] == str(group / 'runs/outers' / ('outer-%05d' % index)
                                                 / 'training/outer-complete.json'), 'Wrong completion path')
        require(row['checkpoint']['path'] == str(group / 'runs/checkpoints'
                / ('update-%08d-outer-%05d' % (row['optimizer_updates'], index)) / 'manifest.json'),
                'Wrong immutable checkpoint path; earlier-outer K0 reference is HOLD')
    for key in ('status', 'group_terminal'):
        reference(scope[key])
    require(scope['status']['path'] == str(group / 'runs/status.json'), 'Wrong final status path')
    require(Path(scope['group_terminal']['path']).name == 'group-terminal.json'
            and Path(scope['group_terminal']['path']).parent.name == 'training'
            and BASE in Path(scope['group_terminal']['path']).parents, 'Wrong group terminal scope')
    require(isinstance(scope.get('global_files', []), list), 'Explicit global file list required')
    require(not scope.get('external_full_vocab_files'),
            'External full-vocab manifest scope not implemented; retain pending, never archive mutable ledger')
    for item in scope.get('global_files', []):
        reference(item)
        require(type(item.get('bytes')) is int and item['bytes'] >= 0, 'Fixed global bytes required')
    require(len({f['path'] for f in scope.get('global_files', [])}) == len(scope.get('global_files', [])),
            'Duplicate fixed global reference')
    return group


def read_ref(b, ref):
    raw, proof = b['fileproof'](Path(ref['path']), ref, retain=True)
    return json.loads(raw), proof


def inline_scope(encoded, sha):
    raw = base64.b64decode(encoded, validate=True)
    require(hashlib.sha256(raw).hexdigest() == sha, 'Literal scope original-byte SHA differs')
    value = json.loads(raw); scope_shape(value)
    return value, dict(delivery='inline-base64', bytes=len(raw), sha256=sha)


def closure(b, scope):
    group = scope_shape(scope)
    cfg, cfg_ref = read_ref(b, scope['config'])
    require(cfg['schema'] == 'opsd-ablation-config-v1' and cfg['variant'] == scope['variant']
            and cfg['run_id'] == scope['run_id'] and cfg['paths']['output_root'] == scope['output_root']
            and cfg['scientific']['outer_iterations'] == 10, 'Original config/run differs')
    rows = []
    for row in scope['outers']:
        complete, cr = read_ref(b, row['completion'])
        cp, cpr = read_ref(b, row['checkpoint'])
        require(complete['accepted'] is True and complete['run_id'] == scope['run_id']
                and complete['outer_index'] == row['outer_index']
                and complete['optimizer_updates'] == row['optimizer_updates']
                and same_ref(complete['checkpoint'], row['checkpoint']), 'Pinned completion differs')
        require(cp['accepted'] is True and cp['run_id'] == scope['run_id']
                and cp['config_sha256'] == scope['config']['sha256']
                and cp['optimizer_updates'] == row['optimizer_updates']
                and cp['cursor']['outer_index'] == row['outer_index'], 'Pinned checkpoint identity differs')
        rows.append(dict(outer_index=row['outer_index'], optimizer_updates=row['optimizer_updates'],
                         complete=cr, checkpoint=cpr))
    status, sr = read_ref(b, scope['status'])
    terminal, tr = read_ref(b, scope['group_terminal'])
    require(status['phase'] == 'TRAIN_COMPLETE' and status['outer_iterations'] == 10
            and status['run_id'] == scope['run_id'] and status['variant'] == scope['variant']
            and status['config_sha256'] == scope['config']['sha256']
            and status['actual_optimizer_updates'] == scope['outers'][-1]['optimizer_updates']
            and same_ref(status['latest_checkpoint'], scope['outers'][-1]['checkpoint']), 'Final status is not ten closed outers')
    require(terminal['schema'] == 'opsd-ablation-group-foreground-terminal-v1'
            and terminal['accepted'] is True and terminal['run_id'] == scope['run_id']
            and terminal['config_sha256'] == scope['config']['sha256']
            and terminal['actual_exit_code'] == 0 and terminal['signal_termination_not_exit0'] is False
            and terminal['controller'] == status['owner'] and same_ref(terminal['status'], scope['status']),
            'Missing actual original controller rc0')
    owner = terminal['controller']
    require(owner['uid'] == 1005 and owner['run_id'] == scope['run_id']
            and type(owner['pid']) is int and owner['pid'] > 0
            and type(owner['starttick']) is int and owner['starttick'] > 0
            and not owner.get('capture_status'), 'Original controller identity is unknown or differs')
    wait, wr = read_ref(b, terminal['actual_child_exit'])
    release, rr = read_ref(b, terminal['actual_release'])
    require(wait['actual_wait_returncode'] == 0 and wait['identity'] == terminal['controller']
            and wait['run_id'] == scope['run_id'] and wait['signal_termination_not_exit0'] is False,
            'Native parent wait does not prove original rc0')
    require(release['accepted'] is True and release['owned']['accepted'] is True
            and release['after']['accepted'] is True, 'Missing original owned-release proof')
    return dict(config=cfg_ref, outers=rows, status=sr, group_terminal=tr,
                actual_child_exit=wr, actual_release=rr)


def enumerate_files(b, scope, closed):
    group = scope_shape(scope); paths = []; excluded = []
    if scope.get('include_outer_raw', True):
        for index in range(10):
            outer = group / 'runs/outers' / ('outer-%05d' % index)
            for root, dirs, names in os.walk(outer, topdown=True, followlinks=False):
                root = Path(root); b['checked_path'](str(root), directory=True)
                if root == outer and 'merged-model-cache' in dirs:
                    b['checked_path'](str(root / 'merged-model-cache'), directory=True)
                    dirs.remove('merged-model-cache'); excluded.append(str(root / 'merged-model-cache'))
                for name in dirs: b['checked_path'](str(root / name), directory=True)
                for name in names: paths.append(b['checked_path'](str(root / name)))
    terminal_dir = Path(scope['group_terminal']['path']).parent
    fixed = {scope['config']['path'], str(group / 'runs/status.json'), str(group / 'runs/child-history.jsonl'),
             str(group / 'runs/run-identity.json')}
    for item in scope.get('global_files', []):
        path = Path(item['path'])
        allowed = str(path) in fixed or group / 'runs/phase-exits' in path.parents \
            or group / 'receipts/dtype-recovery' in path.parents or terminal_dir in path.parents
        require(allowed, 'Global file is outside closed group training scope; no evaluation/observer/ledger')
        paths.append(b['checked_path'](str(path)))
    paths = sorted(set(paths), key=lambda p: p.relative_to(BASE).as_posix())
    require(paths, 'Empty archive scope')
    for path in paths:
        tarfile.TarInfo(b['safe_name'](path.relative_to(BASE).as_posix())).tobuf(format=tarfile.USTAR_FORMAT)
    return paths, excluded


def validate_inventory(b, scope, scope_sha, value):
    group = scope_shape(scope)
    require(value['schema'] == INVENTORY and value['scope']['sha256'] == scope_sha
            and value['source'] == str(BASE) and value['run_id'] == scope['run_id']
            and value['variant'] == scope['variant'] and value['closed_outers'] == list(range(10))
            and same_ref(value['config'], scope['config']), 'Inventory scope identity differs')
    require(len(value['closures']['outers']) == 10, 'Missing inventory closures')
    for wanted, got in zip(scope['outers'], value['closures']['outers']):
        require(wanted['outer_index'] == got['outer_index'] and wanted['optimizer_updates'] == got['optimizer_updates']
                and same_ref(wanted['completion'], got['complete'])
                and same_ref(wanted['checkpoint'], got['checkpoint']), 'Inventory pinned closure differs')
    require(same_ref(value['closures']['status'], scope['status'])
            and same_ref(value['closures']['group_terminal'], scope['group_terminal']), 'Inventory terminal differs')
    files = value['files']; require(len(files) == value['expected_file_count']
            and sum(f['bytes'] for f in files) == value['bytes'], 'Inventory size differs')
    require(len({f['relative'] for f in files}) == len({f['path'] for f in files}) == len(files), 'Duplicate inventory members')
    for f in files:
        reference(f); b['safe_name'](f['relative'])
        require(f['path'] == str(BASE / f['relative']) and type(f['bytes']) is int and f['bytes'] >= 0,
                'Escaping or invalid inventory payload')
        path = Path(f['path'])
        outer_allowed = scope.get('include_outer_raw', True) and any(
            group / 'runs/outers' / ('outer-%05d' % index) in path.parents
            and group / 'runs/outers' / ('outer-%05d' % index) / 'merged-model-cache' not in path.parents
            for index in range(10))
        fixed = next((r for r in scope.get('global_files', []) if r['path'] == f['path']), None)
        require(outer_allowed or fixed is not None, 'Inventory payload is outside exact authorized raw scope')
        if fixed is not None:
            require(same_ref(f, fixed) and f['bytes'] == fixed['bytes'], 'Fixed global payload identity differs')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('inventory', 'stream', 'after'))
    delivery = p.add_mutually_exclusive_group(required=True)
    delivery.add_argument('--scope'); delivery.add_argument('--scope-base64')
    p.add_argument('--scope-sha256', required=True)
    a = p.parse_args()
    require(os.getuid() == os.geteuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and 'torch' not in sys.modules, 'Only HKUST1 readonly CPU identity allowed')
    b = primitives()
    scope, sp = inline_scope(a.scope_base64, a.scope_sha256) if a.scope_base64 is not None \
        else b['checked_json'](Path(a.scope), a.scope_sha256)
    closed = closure(b, scope); paths, excluded = enumerate_files(b, scope, closed)
    if a.mode == 'inventory':
        files = [dict(relative=q.relative_to(BASE).as_posix(), **b['fileproof'](q,
                 next((r for r in scope.get('global_files', []) if r['path'] == str(q)), None))) for q in paths]
        for f in files: b['stable'](f)
        require(closed == closure(b, scope) and (paths, excluded) == enumerate_files(b, scope, closed), 'Closed source changed')
        value = dict(schema=INVENTORY, source=str(BASE), scope=sp, run_id=scope['run_id'], variant=scope['variant'],
                     config=closed['config'], closed_outers=list(range(10)), closures=closed, files=files,
                     excluded_subtrees=excluded, expected_file_count=len(files), bytes=sum(f['bytes'] for f in files),
                     source_helper_sha256=HELPER_SHA, study_mutable_ledger_excluded=True,
                     node_originals_retained=True, backup_is_training_dependency=False, observed_unix=time.time())
        validate_inventory(b, scope, a.scope_sha256, value)
        if 'path' in sp: b['stable'](sp)
        print(json.dumps(value, sort_keys=True))
    else:
        raw = sys.stdin.buffer.read(); value = validate_inventory(b, scope, a.scope_sha256, json.loads(raw))
        require(value['scope'] == sp and value['closures'] == closed
                and [f['path'] for f in value['files']] == [str(q) for q in paths]
                and value['excluded_subtrees'] == excluded, 'Source inventory continuity differs')
        for f in value['files']: b['stable'](f)
        if a.mode == 'stream': b['stream_archive'](value, sys.stdout.buffer)
        else:
            require(closed == closure(b, scope) and (paths, excluded) == enumerate_files(b, scope, closed), 'After source changed')
            print(json.dumps(dict(schema='opsd-ablation-closed-raw-source-after-v1', accepted=True,
                scope_sha256=a.scope_sha256, inventory_sha256=hashlib.sha256(raw).hexdigest(),
                closures=closed, files=len(paths), source=str(BASE), run_id=scope['run_id'],
                all_source_metadata_stable=True, all_noncache_file_set_unchanged=True, observed_unix=time.time()), sort_keys=True))
        if 'path' in sp: b['stable'](sp)


if __name__ == '__main__':
    main()
