"""Read only the exact released G_POS failure metadata; never return raw rows."""
from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
assert os.getuid() == 1001
config_path = ROOT / 'group-G_POS/receipts/config.json'
raw = config_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661'
config = json.loads(raw)
paths = [ROOT / 'group-G_POS/runs/status.json',
         ROOT / 'group-G_POS/runs/failure-actual-exits.json',
         ROOT / 'study-status.json',
         Path(config['gates']['scoped_cpu_processes_path'])]
references = []
for path in paths:
    assert path.is_relative_to(ROOT) and not path.is_symlink()
    before = path.stat()
    assert before.st_size < 300000
    contents = path.read_bytes()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    assert all(getattr(before, name) == getattr(after, name) for name in fields)
    json.loads(contents)
    references.append(dict(path=str(path.resolve()), bytes=len(contents),
                           sha256=hashlib.sha256(contents).hexdigest()))
print(json.dumps(dict(schema='opsd-signal-first15-mutable-metadata-refs-v1',
                     checked_unix=time.time(), references=references,
                     no_Node_writes_or_GPU_or_model_or_signals=True), sort_keys=True))
