"""Bind one failed allocation to a new namespace, preserving every science field."""
import copy
import json
from pathlib import Path
import re
from reproduction.trained_tree7b_v2 import contract as original
from reproduction.trained_tree7b_v2.contract import BASE, UID, ALLOCATION, RUNTIME, sha

ROOT = BASE / 'evaluations/trained-tree-r2-observation-recovery-v1'
JOB = 'zhangjunming-tg7-v2-r2-rec1'
OLD_UID = '8bd97881-b194-4da4-8d56-ba394e377f8f'
OLD_JOB = 'zhangjunming-tg7-v2-r2'
OLD_PLAN = original.ROOT / 'plans/r2.json'
OLD_PLAN_SHA = '3065ff8c8b4aa554323a651e5bdad69f69b9008f1bb17089bb0792026dc70d76'
SCIENCE_SOURCE = '1ace724201f75ddf839f942f0d8d4abce94f4b1c'
FAILED_RECEIPT_SHA = 'aefad1266f7ac7c1cfe963bc5f7e06ad1836a2f8808191b56e598b86f7d7b938'
FAILED_ARCHIVE_SHA = '7835e2e0e58ce90adf5efd801c8832aaef86015e1580fd0e9eaa2116489613c3'


def validate_recovery(plan, reference):
    assert reference['repeat'] == 2 and reference['repeat_count'] == 3
    assert reference['job'] == OLD_JOB and reference['source_commit'] == SCIENCE_SOURCE
    assert reference['root'] == str(original.ROOT / 'r2')
    value = copy.deepcopy(plan)
    recovery = value.pop('recovery')
    assert value['job'] == JOB and value['root'] == str(ROOT / 'r2')
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
    receipt_path = ROOT / 'failed-r2-preservation.json'
    assert sha(receipt_path) == FAILED_RECEIPT_SHA
    receipt = json.loads(receipt_path.read_text())
    assert receipt['uid'] == OLD_UID and receipt['scheduler']['state'] == 'FAILED'
    assert receipt['file_count'] == 55 and receipt['archive_sha256'] == FAILED_ARCHIVE_SHA
    assert receipt['no_scientific_acceptance'] is True
    preserved = BASE / 'receipts/tree7-r2-failure-preservation-v1'
    assert sha(preserved / 'manifest.json') == '5f3a295f8bb353bcb6342662145046a5ece60420f7b04244b2de6bfbfc02f807'
    assert sha(preserved / 'failed-r2.tar.gz') == FAILED_ARCHIVE_SHA
    recovery = plan['recovery']
    for name, expected in recovery['entry_source_files'].items():
        assert sha(Path(recovery['entry_source']) / name) == expected, name
    return rows, root


def validate_gate(plan, submission, gate):
    actual = original.validate_gate(plan, submission, gate)
    assert plan['job'] == JOB and plan['repeat'] == 2
    assert submission['superseded_uid'] == gate['superseded_uid'] == OLD_UID
    assert submission['entry_source_commit'] == gate['entry_source_commit'] == plan['recovery']['entry_source_commit']
    return actual
