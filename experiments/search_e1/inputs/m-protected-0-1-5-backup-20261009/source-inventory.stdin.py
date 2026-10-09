# Fixed operational inventory, reusing the published read-only file primitives.
# This is a backup of existing protected states, not an outer-end checkpoint.
import hashlib, json, pathlib, runpy, time

H = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py')
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
S = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-M/runs/checkpoints')
pins = {
    'update-00000000-outer-00000': '5181cf3dfcdcdb741b3f7fe41ccd0c28fcd10584d24dd62f5ee3df0e4cd5abd2',
    'update-00000001-outer-00000': '017afe7f6f586889186141c2c9c8e1bfe2a17472d0d6aa78847f0b99c3caea21',
    'update-00000005-outer-00000': '6f8ac377840255e5ae5a115b73e7477167b9ea436e99d7d2aba43737ce69d19e',
}
b['checked_path'](str(S), directory=True)
manifests, paths, declared = [], [], {}
for name, sha in sorted(pins.items()):
    D = S / name
    b['checked_path'](str(D), directory=True)
    cp, proof = b['checked_json'](D / 'manifest.json', sha)
    update = int(name.split('-')[1])
    reason = {0: 'initial', 1: 'update1', 5: 'update5'}[update]
    assert cp['accepted'] is True and cp['run_id'] == 'b255c1eb72e84ca0bd6315e57e3527d0'
    assert cp['config_sha256'] == 'd075fc3223e4ee43e20378d6b63cfb661ee3ee5ba9737fd1364f0261a334fd12'
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
print(json.dumps(dict(schema='m-protected-0-1-5-fixed-inventory-v1', source=str(S), run_id='b255c1eb72e84ca0bd6315e57e3527d0', variant='M', protected_updates=[0, 1, 5], manifest_pins=pins, manifests=manifests, files=files, bytes=sum(f['bytes'] for f in files), backup_is_training_dependency=False, node_originals_retained=True, observed_unix=time.time()), sort_keys=True))
