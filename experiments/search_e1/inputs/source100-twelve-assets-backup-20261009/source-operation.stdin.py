# Fixed read-only asset operations; uses published file/USTAR stream primitives.
import hashlib, json, pathlib, runpy, sys, time

H = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py')
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
S = pathlib.Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/assets/archive100')
M = pathlib.Path('/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/receipts/source100-model-files.json')
m, mp = b['checked_json'](M, '2ef6070a328357b0f9d3a026010aaed0aa3f8840a241bc1021ed9a183e8d75f7')
assert m['source_grpo_step'] == 100 and len(m['files']) == 12
b['checked_path'](str(S), directory=True)
declared = {r['path']: r for r in m['files']}
assert len(declared) == 12 and sum(r['bytes'] for r in declared.values()) == 13604379671
for r in declared.values():
    p = pathlib.Path(r['path'])
    assert p.parent == S
    b['safe_name'](p.name)
actual = {str(b['checked_path'](str(p))) for p in S.iterdir()}
assert actual == set(declared)
mode = sys.argv[1]
if mode == 'inventory':
    files = [dict(relative=pathlib.Path(path).name, **b['fileproof'](path, ref)) for path, ref in sorted(declared.items())]
    for f in files:
        b['stable'](f)
    print(json.dumps(dict(schema='source100-twelve-fixed-assets-inventory-v1', source=str(S), source_grpo_step=100, source_manifest=mp, files=files, bytes=13604379671, expected_file_count=12, node_originals_retained=True, backup_is_training_dependency=False, observed_unix=time.time()), sort_keys=True))
else:
    raw = sys.stdin.buffer.read()
    x = json.loads(raw)
    assert x['schema'] == 'source100-twelve-fixed-assets-inventory-v1'
    assert x['source'] == str(S) and x['bytes'] == 13604379671 and x['expected_file_count'] == 12
    assert x['source_manifest']['sha256'] == mp['sha256']
    assert [f['path'] for f in x['files']] == sorted(declared)
    for f in x['files']:
        ref = declared[f['path']]
        assert f['relative'] == pathlib.Path(f['path']).name
        assert f['bytes'] == ref['bytes'] and f['sha256'] == ref['sha256']
        b['stable'](f)
    if mode == 'stream':
        # Fixed handles, full SHA while streaming, and before/after metadata.
        b['stream_archive'](x, sys.stdout.buffer)
    else:
        assert mode == 'after'
        print(json.dumps(dict(schema='source100-twelve-source-metadata-after-v1', accepted=True, source=str(S), files=12, source_manifest=mp, inventory_sha256=hashlib.sha256(raw).hexdigest(), all_source_metadata_stable=True, observed_unix=time.time()), sort_keys=True))
