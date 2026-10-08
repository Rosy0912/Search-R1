"""Pure physical evidence closure. No torch/Ray, acceptance creation or thresholds."""
import hashlib
import json
from pathlib import Path

from formal_probability_v1 import require


def file_record(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Missing original evidence file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def verify_record(record, directory=None):
    require(isinstance(record, dict), 'Missing evidence record')
    path = Path(record['path'])
    if directory is not None:
        root = Path(directory).resolve()
        require(path.resolve().is_relative_to(root), 'Evidence path escaped actual outer directory')
    actual = file_record(path)
    require(all(actual[key] == record[key] for key in ('path', 'bytes', 'sha256')),
            'Original evidence bytes/SHA differ')
    return actual


def read_json_record(record, directory=None):
    verify_record(record, directory)
    return json.loads(Path(record['path']).read_text())


def manifest_records(manifest):
    parts = [manifest['observed_turns'], manifest['actions'], manifest['scored_global_batch']]
    parts += manifest['raw_files'] + manifest['raw_receipts']
    for record in manifest['reference_records']:
        parts += [record['raw'], record['receipt']]
    require(len({row['path'] for row in parts}) == len(parts), 'Duplicate evidence declarations')
    return parts


def validate_manifest_closure(manifest, directory, verify_bytes=True):
    """Declared closure cannot replace immutable candidate bytes or omit a receipt."""
    declared, physical = manifest_records(manifest), manifest['evidence_files']
    require(len({row['path'] for row in physical}) == len(physical), 'Duplicate physical evidence paths')
    keyed = {row['path']: row for row in physical}
    require(set(keyed) == {row['path'] for row in declared}, 'Physical evidence closure incomplete/extra')
    for row in declared:
        require(all(row[key] == keyed[row['path']][key] for key in ('path', 'bytes', 'sha256')),
                'Physical evidence overwrote a different declared SHA/bytes')
        if verify_bytes: verify_record(row, directory)
    raw = manifest['raw_files']; receipts = manifest['raw_receipts']
    require(len(raw) == len(receipts) and len(raw) > 0 and len(raw) % 8 == 0,
            'Missing eight-rank turn receipts')
    count = len(raw) // 8
    require(count <= 5 and {(r['serial'], r['rank']) for r in raw} ==
            {(serial, rank) for serial in range(count) for rank in range(8)} and
            {(r['serial'], r['rank']) for r in receipts} ==
            {(serial, rank) for serial in range(count) for rank in range(8)},
            'Missing/duplicate actual rank/turn evidence')
    require(len(manifest['reference_records']) == 8 and
            {r['rank'] for r in manifest['reference_records']} == set(range(8)),
            'Missing/duplicate actual fixed-reference records')
    return physical
