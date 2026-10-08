# Run as Python -c SCRIPT; stdin carries the accepted fixed inventory JSON.
import hashlib, json, pathlib, runpy, sys, time

H = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py')
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
raw = sys.stdin.buffer.read()
x = json.loads(raw)
assert x['schema'] == 'c0-protected-0-1-5-fixed-inventory-v1'
S = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C0/runs/checkpoints')
assert x['source'] == str(S) and x['protected_updates'] == [0, 1, 5]
assert x['manifest_pins'] == {
    'update-00000000-outer-00000': '25190db42a9635640ca59c4d9312dcf20974baf7a0d19c2c7a28a9536499f4e7',
    'update-00000001-outer-00000': 'b958d0de4a1c9a818bb462afbc1ac9fe44cc15a50f3a6397a5d7f97f19ff07d6',
    'update-00000005-outer-00000': '347cd199817b20529c7d2413ee83bde72554f7c21eae687b3ef47e6d566d40c2',
}
paths = []
for name, sha in sorted(x['manifest_pins'].items()):
    b['safe_name'](name)
    b['checked_json'](S / name / 'manifest.json', sha)
    for p in sorted((S / name).rglob('*')):
        assert not p.is_symlink()
        if not p.is_dir():
            b['checked_path'](str(p))
            paths.append(p.relative_to(S).as_posix())
assert paths == [f['relative'] for f in x['files']]
assert len(paths) == 39
for f in x['files']:
    b['stable'](f)
print(json.dumps(dict(schema='c0-protected-0-1-5-source-metadata-after-v1', accepted=True, source=str(S), files=len(paths), inventory_run_id=x['run_id'], inventory_sha256=hashlib.sha256(raw).hexdigest(), all_source_metadata_stable=True, observed_unix=time.time()), sort_keys=True))
