"""Prepare Agent 6's immutable assets with the frozen downloader, without editing it."""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import time

root = Path('/dataset/shaoshitong/search-r1-rl/cosearch')
source = root / 'repo/reproduction/h20/download_assets.py'
spec = importlib.util.spec_from_file_location('frozen_asset_downloader', source)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)
base = root / 'assets-root'
downloader.BASE = base
for part in ('assets', 'logs'):
    (base / part).mkdir(parents=True, exist_ok=True)
manifest_source = root / 'repo/reproduction/h20/assets.json'
manifest = manifest_source.read_bytes()
(base / 'assets.json').write_bytes(manifest)
assets = json.loads(manifest)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(downloader.fetch, assets))
record = dict(time=time.time(), complete=all(results),
              source_commit='1620c2edb13809d19cc3f5af4a298ccd75488104',
              source_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
              downloader_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              failed=[a['path'] for a, ok in zip(assets, results) if not ok])
(base / 'logs/assets-status.json').write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(0 if all(results) else 1)
