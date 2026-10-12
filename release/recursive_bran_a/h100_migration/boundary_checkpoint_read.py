"""Read small native checkpoint manifests; never read model/Adam payloads."""
import hashlib
import json
import pathlib
import time

base = pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
entry = json.loads((base / 'receipts/train-start.json').read_text())
configuration = json.loads((base / 'receipts/configuration.json').read_text())
assert entry['run_id'] == configuration['run_id']
rows = []
concurrent_reads = []
for path in sorted((base / 'checkpoints/training_state').glob('global_step_*/**/manifest.json')):
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (FileNotFoundError, ValueError):
        concurrent_reads.append(str(path))
        continue
    identity = manifest['identity']
    workers = manifest['worker_states']
    rows.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        status=manifest['status'], identity=identity,
        config_matches=identity['config_sha256'] == configuration['config_sha256'],
        declared_source_matches=identity['code_commit'] == configuration['source_commit'],
        worker_count=len(workers), ranks=[w['identity']['rank'] for w in workers],
        native_exact_same_world_reload=[w.get('exact_same_world_reload_verified') for w in workers],
        driver=manifest['driver'], boundary_phase=manifest.get('boundary_phase'),
        hf_checkpoint=manifest['hf_checkpoint']))
print(json.dumps(dict(time=time.time(), entry=entry, manifest_summaries=rows,
    concurrent_reads=concurrent_reads, scope='Fresh small manifest SHA and native reload receipts; no independent model/Adam payload SHA or tensor readback')))
