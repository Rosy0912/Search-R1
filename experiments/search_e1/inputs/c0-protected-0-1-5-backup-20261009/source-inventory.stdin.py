# Fixed operational inventory, reusing the published read-only file primitives.
# This is a backup of existing protected states, not an outer-end checkpoint.
import hashlib, json, pathlib, runpy, time

H = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py')
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
S = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C0/runs/checkpoints')
pins = {
    'update-00000000-outer-00000': '25190db42a9635640ca59c4d9312dcf20974baf7a0d19c2c7a28a9536499f4e7',
    'update-00000001-outer-00000': 'b958d0de4a1c9a818bb462afbc1ac9fe44cc15a50f3a6397a5d7f97f19ff07d6',
    'update-00000005-outer-00000': '347cd199817b20529c7d2413ee83bde72554f7c21eae687b3ef47e6d566d40c2',
}
b['checked_path'](str(S), directory=True)
manifests, paths, declared = [], [], {}
for name, sha in sorted(pins.items()):
    D = S / name
    b['checked_path'](str(D), directory=True)
    cp, proof = b['checked_json'](D / 'manifest.json', sha)
    update = int(name.split('-')[1])
    reason = {0: 'initial', 1: 'update1', 5: 'update5'}[update]
    assert cp['accepted'] is True and cp['run_id'] == 'bfc195746b6d41aaa74db5ac4d7a8be7'
    assert cp['config_sha256'] == '11cdb1841e69df7ca4a452dea5265a7e6b0fbe3da462611b2c349f8df84045c8'
    assert cp['optimizer_updates'] == update and cp['cursor']['outer_index'] == 0
    assert reason in cp['protection_reason'] and cp['complete_rank_count'] == 8
    assert sorted(r['rank'] for r in cp['ranks']) == list(range(8))
    core = [proof, cp['shared_optimizer']] + [r['file'] for r in cp['ranks']] + cp['adapter']['files']
    assert len(core) == 13 and len({r['path'] for r in core}) == 13
    for ref in core:
        assert pathlib.Path(ref['path']).is_relative_to(D)
        declared[ref['path']] = ref
    manifests.append(proof)
    for p in sorted(D.rglob('*')):
        b['require'](not p.is_symlink(), 'symlink refused')
        if p.is_dir():
            continue
        b['checked_path'](str(p))
        b['safe_name'](p.relative_to(S).as_posix())
        paths.append(p)
assert {str(p) for p in paths} == set(declared)
files = [dict(relative=p.relative_to(S).as_posix(), **b['fileproof'](p, declared[str(p)])) for p in paths]
for f in files:
    b['stable'](f)
print(json.dumps(dict(schema='c0-protected-0-1-5-fixed-inventory-v1', source=str(S), run_id='bfc195746b6d41aaa74db5ac4d7a8be7', variant='C0', protected_updates=[0, 1, 5], manifest_pins=pins, manifests=manifests, files=files, bytes=sum(f['bytes'] for f in files), backup_is_training_dependency=False, node_originals_retained=True, observed_unix=time.time()), sort_keys=True))
