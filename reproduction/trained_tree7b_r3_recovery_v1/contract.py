"""Bind one failed allocation to a new namespace, preserving every science field."""
import copy
import json
from pathlib import Path
import re
from reproduction.trained_tree7b_v2 import contract as original
from reproduction.trained_tree7b_v2.contract import BASE, UID, ALLOCATION, RUNTIME, sha

ROOT = BASE / 'evaluations/trained-tree-r3-observation-recovery-v1'
JOB = 'zhangjunming-tg7-v2-r3-rec1'
OLD_UID = '75ad118b-a05f-4954-b041-185d64afd18f'
OLD_JOB = 'zhangjunming-tg7-v2-r3'
OLD_PLAN = original.ROOT / 'plans/r3.json'
OLD_PLAN_SHA = 'e5add9fd4a080764ebae6401753838307d7d219505e35608a0c9652ca13d3e64'
SCIENCE_SOURCE = '1ace724201f75ddf839f942f0d8d4abce94f4b1c'
FAILED_RECEIPT_SHA = '963a68fb6719a749e1f6014f3bac7986a5b466c8bfad9a818e36d5edff4a8688'
FAILED_ARCHIVE_SHA = '53b9c0fd09c222fdb958a1151b95c2ee2af32f550e145c886773f887ec0c0d69'


def validate_recovery(plan, reference):
    assert reference['repeat'] == 3 and reference['repeat_count'] == 3
    assert reference['job'] == OLD_JOB and reference['source_commit'] == SCIENCE_SOURCE
    assert reference['root'] == str(original.ROOT / 'r3')
    value = copy.deepcopy(plan)
    recovery = value.pop('recovery')
    assert value['job'] == JOB and value['root'] == str(ROOT / 'r3')
    value.update(job=reference['job'], root=reference['root'])
    assert value == reference, 'Only allocation job/root and separate recovery provenance may differ'
    assert set(recovery) == {'superseded_uid', 'original_plan_sha256', 'failed_receipt_sha256',
                             'entry_source_commit', 'entry_source', 'entry_source_files'}
    assert recovery['superseded_uid'] == OLD_UID and recovery['original_plan_sha256'] == OLD_PLAN_SHA
    assert recovery['failed_receipt_sha256'] == FAILED_RECEIPT_SHA
    assert re.fullmatch('[a-f0-9]{40}', recovery['entry_source_commit'])
    assert recovery['entry_source_commit'] != SCIENCE_SOURCE
    assert recovery['entry_source'] == str(BASE / 'sources' / recovery['entry_source_commit'])
    files = recovery['entry_source_files']
    assert files and all(re.fullmatch('[a-f0-9]{64}', h) for h in files.values())
    for name, expected in reference['source_files'].items():
        assert files[name] == expected, name
    for name in files:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
    return Path(plan['root'])


def verify(plan):
    assert sha(OLD_PLAN) == OLD_PLAN_SHA
    reference = json.loads(OLD_PLAN.read_text())
    root = validate_recovery(plan, reference)
    rows, _ = original.verify(reference)
    receipt_path = ROOT / 'failed-r3-preservation.json'
    assert sha(receipt_path) == FAILED_RECEIPT_SHA
    receipt = json.loads(receipt_path.read_text())
    assert receipt['uid'] == OLD_UID and receipt['scheduler']['state'] == 'FAILED'
    assert receipt['file_count'] == 55 and receipt['archive_sha256'] == FAILED_ARCHIVE_SHA
    assert receipt['no_scientific_acceptance'] is True
    preserved = BASE / 'receipts/tree7-r3-failure-preservation-v1'
    assert sha(preserved / 'manifest.json') == '1828dce9bb52dcfc0ab0aa5e410b259765221ac350c06b0cab262f2eb789c8cc'
    assert sha(preserved / 'failed-r3.tar.gz') == FAILED_ARCHIVE_SHA
    recovery = plan['recovery']
    for name, expected in recovery['entry_source_files'].items():
        assert sha(Path(recovery['entry_source']) / name) == expected, name
    return rows, root


def validate_gate(plan, submission, gate):
    actual = original.validate_gate(plan, submission, gate)
    assert plan['job'] == JOB and plan['repeat'] == 3
    assert submission['superseded_uid'] == gate['superseded_uid'] == OLD_UID
    assert submission['entry_source_commit'] == gate['entry_source_commit'] == plan['recovery']['entry_source_commit']
    return actual
