"""Exact immutable-input layout; never authorizes old logs/runs/receipts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097')
OLD = Path('/dataset/shaoshitong/search-r1-rl/bapo')
SOURCE = 'b09736d9d1b487a5c134e7a358761e04613cea32'
ASSETS_SHA = 'a033b9d6e5b4453374c25d8d9a33d70254c1a79ea0d71677f21afb083d94b3fd'
# Historical derived-record identity (data, never an authorization receipt).
DERIVED_SHA = '9e473466c7d8cd8c623465a45c5006211df7cde395f9fc3d7811cfb07c87916d'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def owned(path, root=ROOT):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or not path.is_file():
        raise ValueError('Invalid owned file: ' + str(path))
    if root.resolve() not in path.resolve().parents:
        raise ValueError('Evidence outside new owned root: ' + str(path))
    return path

def snapshot(root=ROOT, old=OLD):
    for name in ('', 'repo', 'tools', 'logs', 'cache', 'prepared', 'runs', 'assets-root', 'assets-root/data', 'assets-root/data/paper-grpo'):
        p = root / name
        if not p.is_dir() or p.is_symlink() or p.resolve() != p.absolute():
            raise ValueError('Mutable directory must be physical: ' + str(p))
    targets = {name: str((old / name).resolve()) for name in ('env', 'retriever-gpu-lib', 'lib')}
    targets['assets-root/assets'] = str((old / 'assets-root/assets').resolve())
    for name, target in targets.items():
        p = root / name
        if not p.is_symlink() or str(p.resolve()) != target:
            raise ValueError('Shared link target changed: ' + name)
    manifest = owned(root / 'assets-root/assets.json', root)
    derived = owned(root / 'logs/derived-assets.sha256', root)
    if digest(manifest) != ASSETS_SHA or digest(derived) != DERIVED_SHA:
        raise ValueError('Pinned immutable manifest/derived record changed')
    records = json.loads(manifest.read_text())
    asset_root = (old / 'assets-root/assets').resolve()
    paths = set()
    for item in records:
        rel = Path(item['path'])
        p = root / 'assets-root/assets' / rel
        if rel.is_absolute() or '..' in rel.parts or str(rel) in paths or not p.is_file() or asset_root not in p.resolve().parents:
            raise ValueError('Invalid finite asset manifest path')
        if item.get('size') and p.stat().st_size != item['size']:
            raise ValueError('Asset size changed: ' + str(rel))
        paths.add(str(rel))
    expected_derived = {str(old / 'assets-root/assets/wiki/e5_Flat.index'), str(old / 'assets-root/assets/wiki/wiki-18.jsonl')}
    derived_files = {}
    for line in derived.read_text().splitlines():
        sha, path = line.split(None, 1)
        path = path.strip()
        p = Path(path)
        if path not in expected_derived or path in derived_files or len(sha) != 64 or not p.is_file() or asset_root not in p.resolve().parents:
            raise ValueError('Derived asset record outside exact allowlist')
        int(sha, 16)
        derived_files[path] = {'sha256': sha, 'size': p.stat().st_size}
    if set(derived_files) != expected_derived:
        raise ValueError('Incomplete derived asset allowlist')
    setup = json.loads(owned(root / 'logs/isolation-setup.json', root).read_text())
    if setup.get('source_commit') != SOURCE or setup.get('isolated_root') != str(root) or setup.get('shared_read_only_targets') != {k: targets[k] for k in ('env', 'retriever-gpu-lib', 'lib')} or setup.get('shared_assets_only') != targets['assets-root/assets']:
        raise ValueError('Isolation identity mismatch')
    paper = {}
    for p in sorted((root / 'assets-root/data/paper-grpo').rglob('*')):
        if p.is_symlink():
            raise ValueError('Copied paper inputs cannot be symlinked')
        if p.is_file():
            owned(p, root)
            paper[str(p.relative_to(root / 'assets-root'))] = digest(p)
    if not paper or paper != setup.get('copied_paper_data_files'):
        raise ValueError('Copied immutable paper inputs changed')
    return {'version': 1, 'source_commit': SOURCE, 'isolated_root': str(root),
            'exact_shared_targets': targets, 'assets_manifest_sha256': ASSETS_SHA,
            'derived_record_sha256': DERIVED_SHA, 'derived_files': derived_files,
            'paper_files': paper, 'isolation_setup_sha256': digest(root / 'logs/isolation-setup.json'),
            'large_content_validation': 'hardware parity --hash-large-assets; this layout check verifies pinned manifest and exact paths/sizes'}

def verify(root=ROOT, old=OLD):
    observed = snapshot(root, old)
    recorded = json.loads(owned(root / 'logs/shared-inputs.json', root).read_text())
    if recorded != observed:
        raise ValueError('Bound shared inputs record differs from current layout')
    return observed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    if args.record:
        value = snapshot()
        path = ROOT / 'logs/shared-inputs.json'
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError('Refusing to overwrite changed shared-input identity')
        else:
            path.write_text(json.dumps(value, indent=2))
    else:
        verify()
    print('SHARED_INPUTS_EXACT_IMMUTABLE_LAYOUT_OK')
