"""Preserve frozen archive runtime; prepare staging overlays and explicit env forwarding."""
import hashlib
import json
import shutil
from pathlib import Path
from jit_optimizer import patch_actor_optimizer, patch_worker_optimizer

base = Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
original = base / 'runtime/archive'
destination = base / 'runtime/archive-jit-20261005-env-v2'
manifest = json.loads((base / 'receipts/archive-runtime.sha256.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((original / name).read_bytes()).hexdigest() == digest, name
report_path = base / 'receipts/archive-jit-env-v2-runtime.sha256.json'
if destination.exists():
    saved = json.loads(report_path.read_text())
    for name, digest in saved.items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest, name
    print('EXISTING_SEPARATE_ARCHIVE_JIT_RUNTIME_VERIFIED')
else:
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    changes = {'upstream/Search-R1/verl/workers/actor/dp_actor.py': patch_actor_optimizer,
               'upstream/Search-R1/verl/workers/fsdp_workers.py': patch_worker_optimizer}
    def forward_jit_environment(source):
        old = "for key in ['E1_PHYSICAL_MICRO', 'E1_PHYSICAL_SCORE']:"
        assert source.count(old) == 1
        return source.replace(old, "for key in ['E1_PHYSICAL_MICRO', 'E1_PHYSICAL_SCORE', 'E1_JIT_OPTIMIZER']:")
    changes['search_e1/grpo_main.py'] = forward_jit_environment
    for name, patch in changes.items():
        path = destination / name
        source = patch(path.read_text())
        compile(source, str(path), 'exec')
        path.write_text(source)
    updated = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
               for name in manifest}
    assert {name for name in manifest if manifest[name] != updated[name]} == set(changes)
    report_path.write_text(json.dumps(updated, indent=2))
    (base / 'receipts/archive-jit-env-v2-runtime-overlay.json').write_text(json.dumps(
        dict(original=str(original), destination=str(destination), changed_files=list(changes),
             original_files_unchanged=True, logical_config_unchanged=True,
             equivalence_proof='four-rank FSDP_JIT_ADAMW_BITWISE_EQUIVALENCE_OK'), indent=2))
    print('SEPARATE_ARCHIVE_JIT_RUNTIME_BUILT_TWO_STAGING_FILES_AND_ENV_FORWARDING')
