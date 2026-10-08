"""Derive a factual CPU receipt from exact original results; grant no GPU scope."""
import datetime
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

base = Path.cwd()
root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
directory = base / 'outputs/gigpo-e1-instruct200/cpu-controls-controls-v8-final'
pull = json.loads((directory / 'RAW-PULL-RECEIPT.json').read_bytes())
assert pull['status'] == 'EXACT_ORIGINAL_BYTES_READ_ONLY' and pull['tests_reexecuted'] is False
originals = {}
for row in pull['files']:
    local = Path(row['local_record']['path'])
    raw = local.read_bytes()
    record = row['remote_record']
    assert len(raw) == record['bytes'] == row['local_record']['bytes']
    assert hashlib.sha256(raw).hexdigest() == record['sha256'] == row['local_record']['sha256']
    assert record['path'] == root + '/cpu-tests/controls-v8-final/' + local.name
    assert local.name not in originals
    originals[local.name] = record
assert set(originals) == {'STARTED.json', 'COLLECTION-RESULT.json', 'RESULT.json', 'collection.log', 'pytest.log', 'junit.xml'}
started = json.loads((directory / 'STARTED.json').read_bytes())
collection = json.loads((directory / 'COLLECTION-RESULT.json').read_bytes())
result = json.loads((directory / 'RESULT.json').read_bytes())
review_path = base / 'outputs/gigpo-e1-instruct200/CONTROLS8-INDEPENDENT-PREEXEC-REVIEW-v1.json'
review_raw = review_path.read_bytes()
review = json.loads(review_raw)
policy = review['approved_cpu_policy']
assert hashlib.sha256(review_raw).hexdigest() == '795578800f0e57a10ec121225bd15ac035f368616433d57eacc66a59d6b423d7'
assert result['status'] == 'PASS_CPU_ONLY' and result['exit_code'] == collection['exit_code'] == 0
assert started['command'] == result['command'] == policy['executed_command']
assert started['collection_command'] == collection['command'] == policy['collection_command']
for record in (started, collection, result):
    assert record['management_helper'] == policy['management_helper']
    assert record['controls_manifest_sha256'] == policy['snapshot_sha256']
    assert record['review_sha256'] == hashlib.sha256(review_raw).hexdigest()
    assert record['execution_environment'] == policy['execution_environment']
    assert record['input_file_records'] == policy['input_file_records']
    assert record['actual_execution'] is True and record['gpu_acceptance'] is False
    assert record['production_ready'] is False and record['formal_iterations'] == 0
    assert record['actual_slurm_submitted'] is False and record['source_or_runtime_imported'] is False
assert result['input_file_records_after'] == started['input_file_records_before'] == policy['input_file_records']
assert result['reused_dependency_alias_after'] == started['reused_dependency_alias_before']
assert len(result['reused_dependency_alias_after']['records']) == 51
assert result['controls_directory_names_before'] == result['controls_directory_names_after'] == started['controls_directory_names_before']
assert result['exact_collection_execution_coverage'] is True

xml = (directory / 'junit.xml').read_bytes()
assert b'<!DOCTYPE' not in xml and b'<!ENTITY' not in xml
xml_root = ET.fromstring(xml)
assert xml_root.tag in ('testsuites', 'testsuite')
suites = list(xml_root) if xml_root.tag == 'testsuites' else [xml_root]
counts = dict(collected=0, passed=0, failed=0, errors=0, skipped=0)
ids = []
for suite in suites:
    assert suite.tag == 'testsuite' and suite.find('testsuite') is None
    assert suite.get('hostname') == 'h100-8'
    suite_counts = dict.fromkeys(counts, 0)
    for case in suite.findall('testcase'):
        identity = case.attrib['classname'] + '::' + case.attrib['name']
        ids.append(identity)
        markers = [name for name in ('failure', 'error', 'skipped') if case.find(name) is not None]
        assert len(markers) <= 1
        outcome = {'failure': 'failed', 'error': 'errors', 'skipped': 'skipped'}[markers[0]] if markers else 'passed'
        suite_counts['collected'] += 1
        suite_counts[outcome] += 1
    for xml_key, key in (('tests', 'collected'), ('failures', 'failed'), ('errors', 'errors'), ('skipped', 'skipped')):
        assert int(suite.attrib[xml_key]) == suite_counts[key]
    for key in counts:
        counts[key] += suite_counts[key]
assert counts['collected'] == counts['passed'] > 0
assert counts['failed'] == counts['errors'] == counts['skipped'] == 0
assert len(ids) == len(set(ids)) and counts == result['actual_cases']
assert sorted(ids) == sorted(result['executed_canonical_case_ids']) == sorted(collection['canonical_case_ids'])
log = (directory / 'collection.log').read_text()
nodes = [line for line in log.splitlines() if line.startswith('test_deployment_controls_v8.py::')]
summaries = [int(m.group(1)) for line in log.splitlines() if (m := re.fullmatch(r'(\d+) tests? collected in [0-9.]+s', line.strip()))]
assert nodes == collection['node_ids'] and summaries == [len(nodes)] and len(nodes) == counts['collected']
assert not any('deselected' in line.lower() or 'error collecting' in line.lower() for line in log.splitlines())

current_run = dict(record=originals['RESULT.json'], log_record=originals['pytest.log'], junit_record=originals['junit.xml'],
    started_record=originals['STARTED.json'], collection_record=originals['COLLECTION-RESULT.json'],
    collection_result_record=originals['COLLECTION-RESULT.json'], collection_log_record=originals['collection.log'],
    execution_scope=result['scope'], executed_command=result['command'], collection_command=collection['command'],
    test_files=result['test_files'], exit_code=result['exit_code'], failure_ids=result['failure_ids'], **counts)
receipt = dict(schema='FRESH200_CONTROLS_CPU_FACTUAL_RECEIPT_V1', status='PASS', scope=result['scope'],
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), factual_derivation_only=True,
    independent_raw_evidence_acceptance_pending=True, actual_execution=result['actual_execution'], gpu_used=False,
    actual_slurm_submitted=False, source_or_runtime_imported=False, exit_code=result['exit_code'], **counts,
    manifest_sha256=result['controls_manifest_sha256'], controls_files_digest=result['controls_files_digest'],
    management_helper=result['management_helper'], current_run=current_run, raw_execution_records=list(originals.values()),
    executed_case_ids=sorted(ids), actual_cases=counts, approved_cpu_policy=policy,
    original_static_review=dict(path=policy['original_review_path'], bytes=len(review_raw), sha256=hashlib.sha256(review_raw).hexdigest()),
    actual_input_records_before=started['input_file_records_before'], actual_input_records_after=result['input_file_records_after'],
    known_51_dependency_alias_before=started['reused_dependency_alias_before'], known_51_dependency_alias_after=result['reused_dependency_alias_after'],
    prior_runs_borrowed=[], source_runtime_CPU_rerun=False, GPU_acceptance=False, READY=False, submitted=False,
    formal_iterations=0, actual_formal_adam_writes=0)
out = base / 'outputs/gigpo-e1-instruct200/CONTROLS8-HELPER9-FACTUAL-CPU-RECEIPT-v1.json'
raw = (json.dumps(receipt, indent=2) + '\n').encode()
with out.open('xb') as f:
    f.write(raw)
print(json.dumps(dict(path=str(out.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), actual_cases=counts,
    independent_raw_evidence_acceptance_pending=True, GPU_executed=False, formal_iterations=0)))
