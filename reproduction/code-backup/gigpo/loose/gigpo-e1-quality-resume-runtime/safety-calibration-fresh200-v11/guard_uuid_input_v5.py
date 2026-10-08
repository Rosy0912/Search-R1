"""Bind the unchanged guard's actual read-only legacy full-eight UUID input."""
import hashlib
import json
import re
from pathlib import Path

ACTUAL_GUARD_UUID_PATH = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/tools/expected-gpu-uuids.json'
ACTUAL_GUARD_UUID_SHA256 = '0a5a2f204b4f7b22758c7c297837581e0992b06f2b8e4c72d7890ad23b78a0ed'
ACTUAL_GUARD_UUID_BYTES = 352
ACTUAL_GUARD_UUID_SCHEMA = 'PLAIN_CANONICAL_DISTINCT_NVML_UUIDS_FULL8_V1'
GUARD_SHA256 = '9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'
OWNERSHIP_SHA256 = '7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def validate_guard_uuid_binding(binding):
    item = binding['canonical_uuid_inventory_record']
    require(item['path'] == ACTUAL_GUARD_UUID_PATH and item['sha256'] == ACTUAL_GUARD_UUID_SHA256
        and type(item['bytes']) is int and item['bytes'] == ACTUAL_GUARD_UUID_BYTES, 'Actual guard UUID input identity differs')
    covered = [row for row in binding['files'] if row['path'] == ACTUAL_GUARD_UUID_PATH]
    require(len(covered) == 1 and covered[0]['sha256'] == ACTUAL_GUARD_UUID_SHA256
        and type(covered[0]['bytes']) is int and covered[0]['bytes'] == ACTUAL_GUARD_UUID_BYTES,
        'Actual guard UUID input must occur exactly once with exact physical bytes in files')
    inventory = binding['expected_gpu_uuid_inventory']
    canonical_full_eight(inventory, 'Bound canonical expected GPU UUID inventory')
    inventory_raw = json.dumps(inventory).encode('utf-8')
    require(len(inventory_raw) == ACTUAL_GUARD_UUID_BYTES
        and hashlib.sha256(inventory_raw).hexdigest() == ACTUAL_GUARD_UUID_SHA256,
        'Bound expected GPU inventory differs from exact canonical raw input')
    return dict(path=ACTUAL_GUARD_UUID_PATH, sha256=ACTUAL_GUARD_UUID_SHA256,
                bytes=ACTUAL_GUARD_UUID_BYTES, uuid_schema=ACTUAL_GUARD_UUID_SCHEMA)


def canonical_full_eight(values, label):
    require(type(values) is list and len(values) == 8,
            label + ' must contain eight distinct canonical UUIDs')
    require(all(type(value) is str and re.fullmatch(
        r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', value) for value in values),
        label + ' contains noncanonical UUIDs')
    require(len(set(values)) == 8, label + ' contains duplicate UUIDs')
    return set(values)


def read_actual_guard_input(path):
    actual = Path(path)
    require(actual.resolve() == actual and actual.is_file(), 'Actual guard UUID path is missing/aliased')
    return actual.read_bytes()


def validate_actual_guard_uuid_input(binding, selected, inherited_leases, *, _read_bytes=None):
    """The caller first proves actual inherited OFD ownership, then takes live GPUs.

    _read_bytes is a CPU-fixture seam; production reads only the bound actual path.
    This function never writes the legacy input or acquires/releases a lease.
    """
    identity = validate_guard_uuid_binding(binding)
    raw = (read_actual_guard_input if _read_bytes is None else _read_bytes)(identity['path'])
    require(type(raw) is bytes and len(raw) == ACTUAL_GUARD_UUID_BYTES
        and hashlib.sha256(raw).hexdigest() == ACTUAL_GUARD_UUID_SHA256,
        'Actual read-only guard UUID bytes changed')
    expected = json.loads(raw.decode('utf-8'))
    canonical = canonical_full_eight(expected, 'Actual guard UUID input')
    require(expected == binding['expected_gpu_uuid_inventory'],
            'Actual guard UUID list differs from bound canonical inventory')
    selected_uuids = [row['uuid'] for row in selected]
    lease_uuids = [row['uuid'] for row in inherited_leases]
    require(canonical == canonical_full_eight(selected_uuids, 'Actual allocated/visible GPU set')
        == canonical_full_eight(lease_uuids, 'Validated inherited OFD lease set'),
        'Guard input differs from actual allocated/visible/OFD full-eight UUIDs')
    return dict(identity=identity, actual_guard_read_path=identity['path'], raw_utf8=raw.decode('utf-8'),
        canonical_uuids=expected, actual_selected_uuids=selected_uuids,
        validated_inherited_lease_uuids=lease_uuids, count=8,
        legacy_input_read_only=True, new_run_copy_is_not_guard_input=True)
