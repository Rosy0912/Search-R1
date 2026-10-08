# Run as Python -c SCRIPT; stdin carries the accepted fixed inventory JSON.
import hashlib, json, pathlib, runpy, sys, time

H = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py')
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
raw = sys.stdin.buffer.read()
x = json.loads(raw)
assert x['schema'] == 'c1-protected-0-1-5-fixed-inventory-v1'
S = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C1/runs/checkpoints')
assert x['source'] == str(S) and x['protected_updates'] == [0, 1, 5]
assert x['manifest_pins'] == {
    'update-00000000-outer-00000': '916f213785f1a65742c8a377602ba9a623323d0167523464677f010596a8185b',
    'update-00000001-outer-00000': '2ac3a2a23119fee980ae623d2caa56bdb66b2f6561ab5e30960c50033dd3e82c',
    'update-00000005-outer-00000': 'd98e1477ceeb21e6198a2d12975a1d3fa4b785b41d7c87d2b49d9bf5557be1b0',
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
print(json.dumps(dict(schema='c1-protected-0-1-5-source-metadata-after-v1', accepted=True, source=str(S), files=len(paths), inventory_run_id=x['run_id'], inventory_sha256=hashlib.sha256(raw).hexdigest(), all_source_metadata_stable=True, observed_unix=time.time()), sort_keys=True))
