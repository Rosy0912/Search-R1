"""UNEXECUTED CPU guard-input fixtures. No NVML, OFD lock or remote reads.

Canonical values are copied from prior read-only metadata evidence, key
old_canonical_expected_gpu_uuids_json, metadata SHA
 d272c4de420927f079130d48d3ca30e5ef7c7bf460048c1cb07d9c9a35021d7a.
json.dumps(list).encode() reproduces the original unchanged352-byte file SHA.
"""
import copy
import hashlib
import json
import pytest
from guard_uuid_input_v5 import (ACTUAL_GUARD_UUID_PATH, ACTUAL_GUARD_UUID_SHA256,
    ACTUAL_GUARD_UUID_BYTES, validate_guard_uuid_binding, validate_actual_guard_uuid_input,
    canonical_full_eight)


def canonical_uuids():
    return ['GPU-53210089-2b52-bf37-b574-f91e8892a012',
        'GPU-9b68e8d5-643e-25d0-99fa-6cdcd810082c',
        'GPU-85473c01-4753-bf6f-080f-9c89017440dd',
        'GPU-9f380cfa-4239-a891-2e73-4bc48f32f06a',
        'GPU-56b1a759-2d63-8c03-fd6f-97883cd6a369',
        'GPU-898426c9-149c-ecf1-1cd6-450c8d36a548',
        'GPU-fc12be2b-bff3-6df7-9eca-656755fdf7c1',
        'GPU-cef6b05a-4e86-cab0-b75e-c417c30cf67e']


def fixture_binding():
    record = dict(path=ACTUAL_GUARD_UUID_PATH, bytes=ACTUAL_GUARD_UUID_BYTES,
                  sha256=ACTUAL_GUARD_UUID_SHA256)
    return dict(canonical_uuid_inventory_record=record,
                expected_gpu_uuid_inventory=canonical_uuids(), files=[dict(record)])


def fixture_records():
    return [dict(uuid=value) for value in canonical_uuids()]


def test_exact_actual_legacy_raw_input_full8_selected_and_inherited_set():
    raw = json.dumps(canonical_uuids()).encode()
    assert len(raw) == 352 and hashlib.sha256(raw).hexdigest() == ACTUAL_GUARD_UUID_SHA256
    reads = []
    def reader(path):
        reads.append(path)
        return raw
    proof = validate_actual_guard_uuid_input(fixture_binding(), fixture_records(),
        list(reversed(fixture_records())), _read_bytes=reader)
    assert reads == [ACTUAL_GUARD_UUID_PATH] and proof['count'] == 8
    assert proof['legacy_input_read_only'] and proof['new_run_copy_is_not_guard_input']


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'alternate_path', 'changed_sha', 'changed_bytes', 'bound_subset'])
def test_missing_duplicate_alternate_or_changed_guard_input_binding_refused(case):
    binding = fixture_binding()
    if case == 'missing': binding['files'] = []
    if case == 'duplicate': binding['files'].append(dict(binding['files'][0]))
    if case == 'alternate_path': binding['canonical_uuid_inventory_record']['path'] = '/new-run/expected-gpu-uuids.json'
    if case == 'changed_sha': binding['canonical_uuid_inventory_record']['sha256'] = 'a' * 64
    if case == 'changed_bytes': binding['files'][0]['bytes'] = 353
    if case == 'bound_subset': binding['expected_gpu_uuid_inventory'] = canonical_uuids()[:7]
    with pytest.raises(ValueError): validate_guard_uuid_binding(binding)


@pytest.mark.parametrize('case', ['empty', 'subset', 'duplicate', 'stale', 'changed_format'])
def test_actual_legacy_input_mutation_is_refused_before_guard(case):
    values = canonical_uuids()
    if case == 'empty': values = []
    if case == 'subset': values = values[:7]
    if case == 'duplicate': values[-1] = values[0]
    if case == 'stale': values[0] = 'GPU-00000000-0000-0000-0000-000000000000'
    raw = json.dumps(values).encode() + (b'\n' if case == 'changed_format' else b'')
    with pytest.raises(ValueError):
        validate_actual_guard_uuid_input(fixture_binding(), fixture_records(), fixture_records(), _read_bytes=lambda path: raw)


@pytest.mark.parametrize('which,case', [('selected','subset'),('selected','duplicate'),('selected','stale'),
    ('leases','subset'),('leases','duplicate'),('leases','stale')])
def test_live_selected_or_inherited_ofd_uuid_set_mismatch_refused(which, case):
    selected, leases = fixture_records(), fixture_records()
    rows = selected if which == 'selected' else leases
    if case == 'subset': rows.pop()
    if case == 'duplicate': rows[-1]['uuid'] = rows[0]['uuid']
    if case == 'stale': rows[0]['uuid'] = 'GPU-00000000-0000-0000-0000-000000000000'
    raw = json.dumps(canonical_uuids()).encode()
    with pytest.raises(ValueError):
        validate_actual_guard_uuid_input(fixture_binding(), selected, leases, _read_bytes=lambda path: raw)


def test_missing_actual_legacy_file_propagates_without_new_run_fallback():
    reads = []
    def missing(path):
        reads.append(path)
        raise FileNotFoundError(path)
    with pytest.raises(FileNotFoundError):
        validate_actual_guard_uuid_input(fixture_binding(), fixture_records(), fixture_records(), _read_bytes=missing)
    assert reads == [ACTUAL_GUARD_UUID_PATH]


@pytest.mark.parametrize('values', [[], ['GPU-bad'] * 8, canonical_uuids()[:7], canonical_uuids()[:7]+[canonical_uuids()[0]]])
def test_plain_full8_canonical_schema_refuses_empty_subset_duplicate_or_noncanonical(values):
    with pytest.raises(ValueError): canonical_full_eight(values, 'fixture')
