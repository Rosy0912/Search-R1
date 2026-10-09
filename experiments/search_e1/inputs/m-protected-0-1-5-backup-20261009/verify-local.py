# Independent complete local copy check using the original published primitives.
import argparse, hashlib, json, pathlib, runpy, time

p = argparse.ArgumentParser()
for key in ('inventory', 'destination', 'source-helper', 'after-metadata', 'output'):
    p.add_argument('--' + key, required=True)
p.add_argument('--actual-transfer-returncode', type=int, required=True)
a = p.parse_args()
H = pathlib.Path(a.source_helper)
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
I = pathlib.Path(a.inventory)
raw = I.read_bytes()
x = json.loads(raw)
assert x['schema'] == 'm-protected-0-1-5-fixed-inventory-v1'
assert a.actual_transfer_returncode == 0
A = pathlib.Path(a.after_metadata)
after = json.loads(A.read_bytes())
assert after['accepted'] is True and after['all_source_metadata_stable'] is True
assert after['source'] == x['source'] and after['files'] == len(x['files'])
assert after['inventory_sha256'] == hashlib.sha256(raw).hexdigest()
assert after['inventory_run_id'] == x['run_id'] == 'b255c1eb72e84ca0bd6315e57e3527d0'
assert x['protected_updates'] == [0, 1, 5] and len(x['files']) == 39
D = pathlib.Path(a.destination)
b['checked_path'](str(D), directory=True)
names = []
for f in sorted(D.rglob('*')):
    assert not f.is_symlink()
    if not f.is_dir():
        b['checked_path'](str(f))
        names.append(f.relative_to(D).as_posix())
assert names == [f['relative'] for f in x['files']]
for f in x['files']:
    name = b['safe_name'](f['relative'])
    b['fileproof'](D / name, dict(path=str(D / name), sha256=f['sha256'], bytes=f['bytes']))
proof = dict(schema='m-protected-0-1-5-full-state-local-backup-v1', accepted=True, run_id=x['run_id'], variant='M', protected_updates=x['protected_updates'], source=x['source'], destination=str(D), files=len(names), bytes=x['bytes'], actual_rsync_returncode=a.actual_transfer_returncode, manifests=x['manifests'], inventory_sha256=hashlib.sha256(raw).hexdigest(), after_metadata_sha256=hashlib.sha256(A.read_bytes()).hexdigest(), all_source_fresh_SHA_before=True, all_source_metadata_stable_after=True, all_local_full_SHA_match=True, node_originals_retained=True, backup_is_training_dependency=False, no_model_or_GPU=True, numeric_Adam_acceptance_is_separate=True, observed_unix=time.time())
b['write_json'](pathlib.Path(a.output), proof)
print(json.dumps(dict(accepted=True, files=len(names), bytes=x['bytes'], output=a.output)))
