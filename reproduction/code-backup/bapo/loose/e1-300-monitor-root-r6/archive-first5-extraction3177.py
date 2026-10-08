import hashlib
import json
import tarfile
from pathlib import Path

logs = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177/logs')
out = logs / 'independent-step5-extraction-firstactual-20261004'
record = json.loads((out / 'evidence.json').read_text())
assert record['extraction_complete'] and not record['issues']
paths = sorted(out.rglob('*'))
assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in paths)
archive = logs / 'independent-step5-extraction-firstactual-20261004.tar.gz'
with archive.open('xb') as handle:
    with tarfile.open(fileobj=handle, mode='w:gz') as tar:
        tar.add(out, arcname=out.name)
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'files': sum(p.is_file() for p in paths)}))
