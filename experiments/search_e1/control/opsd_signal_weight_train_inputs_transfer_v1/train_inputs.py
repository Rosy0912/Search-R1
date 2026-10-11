"""Fixed completed G-arm TRAIN inputs; no weights, model, GPU or production edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

S = Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
T = Path('/home/shaoshitong/search-e1-migration-20261011')
CONFIGS = {
    'G_POS': 'e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661',
    'G_NEG': '2c37c851193dbc9a2d2512a72dd49310b9b8bcc14ac903159af4a491ef773a0d',
}
SCHEMA = 'signal-completed-G-training-input-transfer-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and not any(c in str(p) for c in '\n\r\x00'), 'Absolute non-traversing path required')
    relative = p.relative_to(S)
    require(relative.parts and relative.parts[0] in {'group-G_POS', 'group-G_NEG'}, 'G inputs only')
    return p


def fresh_ref(path):
    p = safe(path)
    require(p.is_file() and not p.is_symlink() and p.resolve() == p, 'Regular canonical source only')
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))


def verify_file(path, ref):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Regular file required')
    require(p.stat().st_size == ref['bytes'] and digest(p) == ref['sha256'], 'File bytes/SHA differ: ' + str(p))


def completed_rows(ledger, group, cfg, config_sha):
    require(ledger.get('accepted') is True and ledger['variant'] == group and ledger['run_id'] == cfg['run_id'], 'Ledger identity differs')
    rows = ledger['rows']
    require(len(rows) == 25 and [r['outer_index'] for r in rows] == list(range(25)), 'Complete unique 25 outers required')
    for row in rows:
        require(row.get('accepted') is True and row['variant'] == group and row['run_id'] == cfg['run_id']
                and row['config_sha256'] == config_sha, 'Outer identity differs')
        require(row['train_denominator'] == 2560 and row['train_post_update'] is False
                and type(row['train_correct']) is int and 0 <= row['train_correct'] <= 2560
                and row['train_em'] == row['train_correct'] / 2560, 'Full preupdate TRAIN EM required')
    return rows


def inventory():
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only owner required')
    refs = {}
    groups = []

    def add(ref):
        p = safe(ref['path'])
        if str(p) in refs:
            require(all(ref[k] == refs[str(p)][k] for k in ('path', 'bytes', 'sha256')), 'Conflicting duplicate reference')
            return
        verify_file(p, ref)
        require(p.resolve() == p, 'Source aliases not allowed')
        refs[str(p)] = {k: ref[k] for k in ('path', 'bytes', 'sha256')}

    for group, config_sha in CONFIGS.items():
        root = S / ('group-' + group)
        config_ref = fresh_ref(root / 'receipts/config.json')
        require(config_ref['sha256'] == config_sha, 'Frozen config SHA differs')
        add(config_ref)
        cfg = json.loads(Path(config_ref['path']).read_text())
        ledger_ref = fresh_ref(root / 'runs/training-em-25-complete.json')
        add(ledger_ref)
        ledger = json.loads(Path(ledger_ref['path']).read_text())
        rows = completed_rows(ledger, group, cfg, config_sha)
        updates = []
        pairs = 0
        for row in rows:
            summary_ref = fresh_ref(root / 'runs/outers' / ('outer-%05d' % row['outer_index']) / 'summary.json')
            add(summary_ref)
            require(json.loads(Path(summary_ref['path']).read_text()) == row, 'Ledger/summary differ')
            outer_pairs = 0
            require(len(row['source_update_refs']) == row['optimizer_updates_delta'], 'Update count differs')
            for ref in row['source_update_refs']:
                add(ref)
                update = json.loads(Path(ref['path']).read_text())
                require(update['schema'] == 'opsd-online-real-update-v1' and update['accepted'] is True
                        and update['run_id'] == cfg['run_id'] and update['outer_index'] == row['outer_index']
                        and update['contract']['variant'] == group, 'Uncommitted/foreign update')
                updates.append(update['optimizer_updates'])
                ranks = update['all_rank_actual_token_metrics']
                require(len(ranks) == 4 and {r['rank'] for r in ranks} == set(range(4)), 'Four rank metrics required')
                actual = sum(len(r['pairs']) for r in ranks)
                require(actual == update['global_actual_pairs'], 'Actual pair count differs')
                outer_pairs += actual
                for rank in ranks:
                    for metric in rank['pairs']:
                        require(metric['score_training_match']['accepted'] is True, 'Signal/forward mismatch')
                        add(metric['signal_score'])
                        add(metric['token_log'])
            require(outer_pairs == row['K'], 'Outer selected exposure differs')
            pairs += outer_pairs
        require(updates == list(range(1, rows[-1]['optimizer_updates'] + 1)), 'Committed Adam chain must be continuous and unique')
        require(pairs == rows[-1]['cumulative_pair_exposure'], 'Total exposure differs')
        groups.append(dict(variant=group, completed_outers=25, actual_updates=len(updates), selected_pairs=pairs,
                           config=config_ref, ledger=ledger_ref))
    return dict(schema=SCHEMA, accepted=True, groups=groups, files=sorted(refs.values(), key=lambda r: r['path']),
                member_bytes=sum(r['bytes'] for r in refs.values()), checked_unix=time.time(),
                source='HKUST4', target='HKUST5', no_weights=True, no_gpu=True,
                scope='Committed updates, current signal records and actual token logs; not a full rollout/weights/NPY backup')


def install(value, stage):
    require(value.get('schema') == SCHEMA and value.get('accepted') is True, 'Foreign inventory')
    require({g['variant'] for g in value['groups']} == set(CONFIGS) and len(value['groups']) == 2, 'Both G arms required')
    require(all(g['completed_outers'] == 25 and g['config']['sha256'] == CONFIGS[g['variant']] for g in value['groups']), 'Frozen complete G arms required')
    require(len({r['path'] for r in value['files']}) == len(value['files'])
            and sum(r['bytes'] for r in value['files']) == value['member_bytes'], 'Inventory duplicate/size differs')
    stage = Path(stage).resolve()
    target_root = T.resolve()
    checked = []
    for ref in value['files']:
        p = safe(ref['path'])
        relative = str(p).lstrip('/')
        src, dst = stage / relative, T / relative
        require(src.resolve().is_relative_to(stage) and dst.resolve().is_relative_to(target_root)
                and dst.resolve() == dst, 'Stage/destination escapes or aliases root')
        verify_file(src, ref)
        if dst.exists() or dst.is_symlink():
            verify_file(dst, ref)
        checked.append((ref, src, dst))
    # All staging files and existing destinations pass before any installation.
    for ref, src, dst in checked:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.link(src, dst)
            except FileExistsError:
                pass
        verify_file(dst, ref)
    return dict(accepted=True, schema=SCHEMA, files=len(checked), member_bytes=value['member_bytes'], groups=value['groups'],
                all_fresh_sha=True, originals_preserved=True, existing_differences_rejected=True, large_local_files=0,
                scope=value['scope'], checked_unix=time.time())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['inventory', 'install'])
    p.add_argument('--output', required=True)
    p.add_argument('--inventory')
    p.add_argument('--inventory-sha256')
    p.add_argument('--stage')
    a = p.parse_args()
    output = Path(a.output)
    require(output.is_absolute() and not output.exists(), 'Fresh remote output required')
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU owner required')
    if a.mode == 'inventory':
        require(output.is_relative_to(S / 'analysis'), 'Source inventory stays in independent analysis directory')
        value = inventory()
    else:
        require(output.is_relative_to(T), 'Target receipt stays remote')
        require(digest(a.inventory) == a.inventory_sha256, 'Pinned inventory SHA differs')
        value = install(json.loads(Path(a.inventory).read_text()), a.stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
    if a.mode == 'inventory':
        members = output.with_suffix('.members')
        with members.open('x') as f:
            for ref in value['files']:
                f.write(str(safe(ref['path'])).lstrip('/') + '\n')
            f.write(str(output).lstrip('/') + '\n')
    print(json.dumps(dict(accepted=True, output=str(output), sha256=digest(output), files=len(value.get('files', []))
                         if a.mode == 'inventory' else value['files'], member_bytes=value['member_bytes'],
                         groups=value['groups'], scope=value['scope'])))


if __name__ == '__main__':
    main()
