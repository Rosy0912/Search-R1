#!/usr/bin/env python3
"""Small additive provenance receipt for the already-running r2 native backup.

Run on h100-10 at nice19/ionice3. Never changes the running copy or training.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008')
CONTROL = ROOT / '.shared-output-backup-20261008'
REQUEST = 'retained18-native-once-20261008-r2'
ENTRY_SHA = '1721d96d21df5a2f36dfa0793a1c099df0ddd31722ebe10ffeb207ecc9e21c22'
PACKAGE_SHA = '9774e8854889ace8e618335d5f09c8cfbe229bd265c9b6f454d46ebc8cdfbc88'


def main():
    entry = CONTROL / ('backup_native_once-' + ENTRY_SHA[:16] + '.py')
    assert hashlib.sha256(entry.read_bytes()).hexdigest() == ENTRY_SHA
    spec = importlib.util.spec_from_file_location('backup_entry', entry)
    backup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backup)
    status = json.loads(backup.read_small(CONTROL / (REQUEST + '.status.json')))
    assert status['request_id'] == REQUEST and status['entry_sha256'] == ENTRY_SHA
    assert status['native_job_id'] == '3341'
    package = ROOT / 'instruct300/candidates/current200-retained18-speed-v14-ebef786/manifest.json'
    raw = backup.read_small(package)
    assert backup.sha(raw) == PACKAGE_SHA
    manifest = json.loads(raw)
    data = []
    # Validate and hold every external source byte before writing the supplement.
    for name, digest in sorted(manifest['external_files'].items()):
        path = backup.physical(name)
        assert path.suffix == '.py'
        content = backup.read_small(path)
        assert backup.sha(content) == digest, 'External source SHA mismatch: ' + name
        data.append((path, content, digest))
    target = Path(status['partial'])
    if not target.exists():
        target = Path(status['final'])
    assert target.is_dir() and not target.is_symlink() and target.resolve(strict=True) == target
    assert target.parent == backup.DEST
    inventory = []
    for path, content, digest in data:
        dest = target / 'external-provenance' / path.relative_to(ROOT)
        if dest.exists():
            assert backup.sha(backup.read_small(dest)) == digest
        else:
            backup.write_bytes(dest, content)
        inventory.append(dict(source=str(path), destination=str(dest.relative_to(target)), bytes=len(content), sha256=digest, full_readback_sha256_verified=True))
    receipt = dict(request_id=REQUEST, observed_at=backup.now(), package_manifest_sha256=PACKAGE_SHA,
                   original_manifest_source_identity_verified=True, external_files=inventory,
                   native_job_id=status['native_job_id'], native_step=status['native_step'],
                   native_manifest_sha256=status['manifest_sha256'], training_unchanged=True,
                   native_copy_not_restarted=True, status='COMPLETE_SMALL_EXTERNAL_PROVENANCE_SUPPLEMENT')
    backup.atomic_json(target / 'external-provenance-receipt.json', receipt)
    backup.fsync_dir(target / 'external-provenance')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
