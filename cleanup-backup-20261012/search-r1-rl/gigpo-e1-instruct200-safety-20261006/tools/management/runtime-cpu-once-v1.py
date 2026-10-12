"""Stdlib orchestration for one independently approved, complete offline CPU suite.

This is management code, not a trainer/algorithm import. Existing output is refused.
Usage: review_path review_sha manifest_path manifest_sha output_dir
"""
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
SOURCE = ROOT / 'repo-v8-53105d9'
SOURCE_MANIFEST = ROOT / 'logs/FRESH200-SOURCE-STATIC-SNAPSHOT-v8.json'
SOURCE_SHA = '7555930ecd2a18a9c1bd091e3b1d0dfce094abbbd2f70dcfbc834c40a4b77b51'
PYTHON = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python'
DEPS = ROOT / 'cpu-tests/pytest-dependency-v1'
DEP_SHA = '83d9cc2e0f717b38f0a5bbd2d69673350386400f97867e0b06e33cf0c95a8758'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record(path):
    p = Path(path)
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def bound(path, digest):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    return json.loads(data)


def verify_rows(rows, directory=None):
    result = []
    for item in rows:
        p = Path(item['path']) if directory is None else directory / item['relative_path']
        actual = record(p)
        assert actual['bytes'] == item['bytes'] and actual['sha256'] == item['sha256'], str(p)
        result.append(actual)
    return result


def write_once(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def canonical(node):
    pieces = node.split('::')
    return pieces[0][:-3].replace('/', '.') + (
        '.' + '.'.join(pieces[1:-1]) if len(pieces) > 2 else '') + '::' + pieces[-1]


def main():
    assert len(sys.argv) == 6
    review_path, review_sha, manifest_path, manifest_sha, output_path = sys.argv[1:]
    assert socket.gethostname().split('.')[0] == 'h100-8'
    review = bound(review_path, review_sha)
    manifest = bound(manifest_path, manifest_sha)
    assert type(review['schema']) is int and review['schema'] == 1
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert review['scope'] == 'INDEPENDENT_FROZEN_FRESH200_CALIBRATION_V%d_STATIC_PREEXEC_REVIEW_ONLY' % manifest['runtime_version']
    assert review['cpu_execution_decision']['authorized_by_this_report'] is True
    frozen_manifest = review['frozen_inputs']['manifest']
    assert frozen_manifest['sha256'] == manifest_sha
    assert frozen_manifest['bytes'] == Path(manifest_path).stat().st_size
    assert frozen_manifest['actual_hash_match'] is True
    assert frozen_manifest['runtime_file_count'] == frozen_manifest['actual_runtime_hash_matches'] == 39
    assert not frozen_manifest['mismatches']
    assert review['actual_gpu_acceptance'] is False and review['production_READY'] is False
    assert review['formal_iterations'] == 0
    assert manifest['file_count'] == 39
    source = bound(SOURCE_MANIFEST, SOURCE_SHA)
    helpers = ROOT / ('tools/safety-calibration-fresh200-v%d' % manifest['runtime_version'])
    output = Path(output_path)
    assert output.parent == ROOT / 'cpu-tests' and output.name.startswith('runtime-v')
    source_inputs = verify_rows(source['full_tracked_source_files'], SOURCE)
    runtime_inputs = verify_rows(manifest['files'], helpers)
    tests = review['cpu_execution_decision']['allowed_test_files']
    test_records = verify_rows(tests, helpers)
    names = [item['relative_path'] for item in tests]
    assert len(names) == len(set(names)) == 15
    assert all('/' not in name and name.startswith('test_') and name.endswith('.py') for name in names)
    dep_receipt = bound(DEPS / 'dependency-receipt.json', DEP_SHA)
    verify_rows(json.loads((DEPS / 'installed-target-files.json').read_text()))
    for package in json.loads((DEPS / 'reused-packages-before.json').read_text()):
        verify_rows(package['files'])
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    clean = not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    assert clean and commit == source['new_code_commit'] == '53105d9203924f50d9311ca953ec31f0e09bee02'
    tracked = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files', '-z']).decode().split('\0')[:-1]
    assert {str(SOURCE / name) for name in tracked} == {row['path'] for row in source_inputs}
    inputs = source_inputs + runtime_inputs
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output.mkdir(exist_ok=False)
    for name in ['home', 'tmp', 'hf-cache']:
        (output / name).mkdir()
    environment = dict(PATH='/usr/bin:/bin', HOME=str(output / 'home'), LANG='C.UTF-8',
        TMPDIR=str(output / 'tmp'), CUDA_VISIBLE_DEVICES='',
        PYTHONPATH=':'.join([str(ROOT / 'cpu-tests/pytest-libs-v1'), str(SOURCE), str(helpers)]),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTEST_ADDOPTS='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME=str(output / 'hf-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    interpreter = record(Path(PYTHON).resolve())
    interpreter.update(argv0=PYTHON, resolved_path=str(Path(PYTHON).resolve()))
    assert interpreter['sha256'] == '94df6ff4b32aafdba90e38c0aed70880fe44ee4cf88ecdf411641158f120c1af'
    common = dict(code_commit=commit, scope='FRESH200_RUNTIME_CPU_REGRESSIONS', node='h100-8',
        cwd=str(helpers), interpreter=interpreter, test_files=names, test_file_records=test_records,
        input_file_records=inputs, input_files_digest=digest, execution_environment=environment,
        environment_sanitization=dict(strategy='EXACT_ALLOWLIST_NO_INHERITANCE', inherited_environment_forwarded=False,
            cleared_inputs=dict(PYTEST_ADDOPTS='empty', PYTEST_PLUGINS='absent'),
            removed_input_names=sorted(set(os.environ) - set(environment))),
        config_policy='EMPTY_DEV_NULL', plugin_policy='BUILTIN_ONLY_AUTOLOAD_DISABLED_NO_CONFTEST',
        rootdir=str(helpers), source_snapshot_sha256=SOURCE_SHA, runtime_snapshot_sha256=manifest_sha,
        review_sha256=review_sha, dependency_receipt=record(DEPS / 'dependency-receipt.json'),
        actual_execution=True, gpu_acceptance=False, production_ready=False, formal_iterations=0)
    options = ['-q', '--disable-warnings', '-p', 'no:cacheprovider', '-c', '/dev/null', '--noconftest', '--rootdir', str(helpers)]
    collect = [PYTHON, '-m', 'pytest', '--collect-only'] + options + ['--basetemp', str(output / 'collect-tmp')] + names
    normal = [PYTHON, '-m', 'pytest'] + options + ['--basetemp', str(output / 'pytest-tmp'), '--junitxml', str(output / 'junit.xml')] + names
    before = dict(commit=commit, clean=clean, source_files=len(source_inputs), source_digest=source['source_tracked_files_digest'],
        runtime_files=len(runtime_inputs), runtime_manifest_sha256=manifest_sha)
    write_once(output / 'STARTED.json', dict(common, started_utc=utc(), collection_command=collect, command=normal, source_before=before))
    with (output / 'collection.log').open('wb') as f:
        collection_exit = subprocess.run(collect, cwd=helpers, env=environment, stdout=f, stderr=subprocess.STDOUT).returncode
    nodes = [line.strip() for line in (output / 'collection.log').read_text().splitlines()
             if line.strip().split('::')[0] in names and '::' in line]
    assert collection_exit == 0 and nodes and len(nodes) == len(set(nodes))
    cases = [canonical(node) for node in nodes]
    write_once(output / 'COLLECTION-RESULT.json', dict(common, schema='ACTUAL_FULL_PYTEST_COLLECTION_V1',
        command=collect, exit_code=collection_exit, collection_log=str(output / 'collection.log'),
        log_record=record(output / 'collection.log'), log_sha256=record(output / 'collection.log')['sha256'],
        node_ids=nodes, canonical_case_ids=cases, collected=len(nodes), completed_utc=utc()))
    with (output / 'pytest.log').open('wb') as f:
        exit_code = subprocess.run(normal, cwd=helpers, env=environment, stdout=f, stderr=subprocess.STDOUT).returncode
    junit = ET.parse(output / 'junit.xml')
    rows = junit.findall('.//testcase')
    executed = [row.attrib['classname'] + '::' + row.attrib['name'] for row in rows]
    failed = [executed[i] for i, row in enumerate(rows) if row.find('failure') is not None]
    errors = [executed[i] for i, row in enumerate(rows) if row.find('error') is not None]
    skipped = [executed[i] for i, row in enumerate(rows) if row.find('skipped') is not None]
    coverage = sorted(cases) == sorted(executed) and len(executed) == len(set(executed))
    assert verify_rows(source['full_tracked_source_files'], SOURCE) == source_inputs
    assert verify_rows(manifest['files'], helpers) == runtime_inputs
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    result = dict(common, command=normal, executed_command=normal, exit_code=exit_code,
        actual_cases=executed, collected=len(nodes), passed=len(rows)-len(failed)-len(errors)-len(skipped),
        failed=len(failed), errors=len(errors), skipped=len(skipped), failure_ids=failed + errors,
        collection_record=record(output / 'COLLECTION-RESULT.json'),
        collection_result_record=record(output / 'COLLECTION-RESULT.json'), collection_log_record=record(output / 'collection.log'),
        collected_canonical_case_ids=cases, executed_canonical_case_ids=executed, junit_case_ids=executed,
        exact_collection_execution_coverage=coverage, log=str(output / 'pytest.log'), log_record=record(output / 'pytest.log'),
        log_sha256=record(output / 'pytest.log')['sha256'], junit=str(output / 'junit.xml'),
        junit_record=record(output / 'junit.xml'), junit_sha256=record(output / 'junit.xml')['sha256'],
        source_after=before, completed_utc=utc(),
        status='PASS_CPU_ONLY' if exit_code == 0 and coverage and not failed + errors + skipped else 'FAILED_CPU_OR_COVERAGE')
    write_once(output / 'RESULT.json', result)
    print(json.dumps({key: result[key] for key in ['status', 'collected', 'passed', 'failed', 'errors', 'skipped', 'exit_code', 'failure_ids']}))


if __name__ == '__main__':
    main()
