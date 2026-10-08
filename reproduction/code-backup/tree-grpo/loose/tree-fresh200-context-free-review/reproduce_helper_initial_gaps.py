"""Independent frozen-helper boundary counterexamples. CPU mocks only; no remote or scheduler calls."""
from pathlib import Path
import concurrent.futures
import contextlib
import hashlib
import importlib.util
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
HELPERS = HERE.parent / 'tree-fresh200-dispatch-preparation'
EXPECTED = {
    'fresh200_dispatch_common.py': 'd45b2be2abdd91b922d243f4bfb5fa207de207cd6b07ab06e75c9b6290ae3af0',
    'prepare_tree_fresh200_package.py': 'd736cd1f7822012340fc40e73ab0c985759fa77d8196a97f0d288b74826b5d0f',
    'submit_tree_fresh200_once.py': 'd3cd39a29a9cf0e46d2dbaf9d029b7562e8705609ab24561c7241b0b0818c037',
    'test_fresh200_dispatch_cpu.py': '3e58316f96c330c8e26edc87d63ec16fdd1eb25c5d4b289191b9d33a73339b54',
}
for name, expected in EXPECTED.items():
    assert hashlib.sha256((HELPERS/name).read_bytes()).hexdigest() == expected
spec = importlib.util.spec_from_file_location('frozen_helper_tests', HELPERS/'test_fresh200_dispatch_cpu.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
C, S = fixtures.C, fixtures.S
results = {'execution': {'remote_calls': 0, 'real_scheduler_calls': 0, 'model_assets': 0, 'gpu': False}, 'helper_files': EXPECTED}

# Real initial audit inventories and review receipt; synthetic interface owner metadata
# and package hashes only. No source/runtime files are copied or edited.
audit = json.loads((HERE/'BYTE_AUDIT.json').read_bytes())
review_raw = (HERE/'RECEIPT.json').read_bytes()
assert hashlib.sha256(review_raw).hexdigest() == 'e8ede518cd306030aa2b7a4e53b357b4f0a6a61b9ea8b791ed179258a674d316'
review = json.loads(review_raw)
copied_wrapper = ('supervisor.py','calibration_contract.py','launch.sbatch','CONFIGURATION_OVERRIDES.json','INPUTS.schema.json',
                  'run_tree.sh','validation_cost_entry.py','validation_cost_hooks_instruct300.py','sitecustomize.py')
manifest = {'files': {'source/repo/'+name: entry['sha256'] for name, entry in audit['source_files'].items()}}
manifest['files'].update({name: audit['wrapper_files'][name]['sha256'] for name in copied_wrapper})
interface = {'status': 'FRESH_CALIBRATION_MODE_AND_DIRECT_RANK_SCHEDULER_SOURCE_CPU_IMPLEMENTED',
             'source_files': {name: audit['source_files'][name]['sha256'] for name in
                              ('search_r1/baselines/sampling_consistency.py','verl/workers/fsdp_workers.py')}}
inputs = {'source_interface_receipt': {'path': 'synthetic-owner'}, 'context_free_review': {'path': 'real-review'},
          'context_byte_audit': {'path': 'real-audit', 'sha256': review['artifacts']['BYTE_AUDIT.json']}}
by_path = {'synthetic-owner': interface, 'real-review': review, 'real-audit': audit}
with patch.object(C, 'json_bound', lambda entry: by_path[entry['path']]):
    try:
        C.validate_reviews(inputs, manifest)
        result = {'rejected': False}
    except Exception as exc:
        result = {'rejected': True, 'exception_type': type(exc).__name__, 'exception': str(exc)}
result.update(audited_wrapper_count=len(audit['wrapper_files']), builder_wrapper_count=len(copied_wrapper),
              missing_audited_wrappers=sorted(set(audit['wrapper_files'])-set(copied_wrapper)),
              scope='actual frozen audit + initial blocked review; derived builder inventory and synthetic owner metadata')
assert result['exception_type'] == 'KeyError' and result['exception'] == "'PLAN.md'"
results['H01_inventory_mismatch'] = result

results['H02_code_tar_asset_filter'] = []
for name in ('assets/wiki/e5_Flat.index', 'data/wiki-18.jsonl', 'data/wiki-18.jsonl.gz'):
    case = fixtures.Tests(); case.setUp()
    try:
        tiny = b'SYNTHETIC ONLY: no real model/index/data asset\n'
        case.contents[name] = tiny
        case.manifest['files'][name] = C.digest(tiny)
        accepted, target = case.extract()
        result = {'name': name, 'accepted_and_extracted': (target/name).read_bytes() == tiny,
                  'synthetic_bytes': len(tiny), 'real_assets_transferred': False}
        assert result['accepted_and_extracted']
        results['H02_code_tar_asset_filter'].append(result)
    finally:
        case.tearDown()

case = fixtures.Tests(); case.setUp()
try:
    C.validate_manifest(case.manifest)
    results['H03_unbound_external_closure'] = {'accepted': True, 'only_external_binding': sorted(case.manifest['external_files']),
        'actual_imported_external_dependencies_bound': False,
        'scope': 'manifest validator accepts tiny owner fixture external.py instead of any actual required helper'}
finally:
    case.tearDown()

case = fixtures.Tests(); case.setUp()
try:
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    calls = []
    def run(command, **kwargs):
        # This is a mock, never subprocess.run.
        with lock:
            calls.append(command)
        if command[0] == 'squeue':
            barrier.wait(timeout=5)
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        assert command[0] == 'sbatch'
        with lock:
            job = 700 + sum(call[0] == 'sbatch' for call in calls)
        return SimpleNamespace(returncode=0, stdout=str(job), stderr='')
    def attempt(sha):
        return S.submit_once(C, case.root/'instruct300/candidates/new', sha, 'c'*64, {}, run=run)
    with patch.object(C, 'protected_processes', lambda: None), contextlib.redirect_stdout(io.StringIO()):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(attempt, ('a'*64, 'b'*64)))
    journals = sorted(path.name for path in (case.root/'instruct300').glob('SUBMISSION_ONCE*.json'))
    results['H04_cross_manifest_submit_race'] = {'empty_queue_snapshots': sum(call[0] == 'squeue' for call in calls),
        'mock_sbatch_calls': sum(call[0] == 'sbatch' for call in calls), 'statuses': [out['status'] for out in outputs],
        'separate_journals': journals, 'real_sbatch_calls': 0}
    assert results['H04_cross_manifest_submit_race']['mock_sbatch_calls'] == 2
finally:
    case.tearDown()

output = HERE/'helper_initial_gap_results.json'
output.write_text(json.dumps(results, indent=2, sort_keys=True)+'\n')
print(json.dumps(results, indent=2, sort_keys=True))
