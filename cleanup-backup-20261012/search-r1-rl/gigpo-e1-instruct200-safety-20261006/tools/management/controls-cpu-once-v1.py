"""Management-only stdlib runner for one independently approved controls4 CPU suite."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
CWD = ROOT / 'tools/fresh200-deployment-controls-v4'
OUT = ROOT / 'cpu-tests/controls-v4-final'
MANIFEST_SHA = 'a2ede5890d2586a0f1e6911a63d8bf53b7bbb4ad197c4cfa18df8f0fb62e1abe'
PYTHON = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python'
DEPS = ROOT / 'cpu-tests/pytest-dependency-v1'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record(path):
    path = Path(path); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def bound(path, sha):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha
    return json.loads(data)


def verify(rows):
    for row in rows:
        assert record(row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')}


def write_once(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


def canonical(node):
    pieces = node.split('::')
    return pieces[0][:-3].replace('/', '.') + (
        '.' + '.'.join(pieces[1:-1]) if len(pieces) > 2 else '') + '::' + pieces[-1]


def main():
    assert len(sys.argv) == 4 and socket.gethostname().split('.')[0] == 'h100-8'
    review = bound(sys.argv[1], sys.argv[2]); manifest = bound(sys.argv[3], MANIFEST_SHA)
    assert review['schema'] == 'STRICT_CLEAN_DEPLOYMENT_CONTROLS_PREEXEC_REVIEW_V4'
    assert review['scope'] == 'FRESH200_DEPLOYMENT_CONTROLS_PREEXEC_STATIC_ONLY'
    assert review['status'] in ('PASS', 'PASS_STATIC_ONLY_CPU_SCOPE') and review['independent_review'] is True
    decision = review['decision']
    assert decision['blocking_findings'] == 0 and decision['whole_controls_static_pass'] is True
    assert decision['CPU_fixture_execution_authorized_by_this_report'] is True
    assert all(decision[key] is False for key in ('READY_allowed', 'GPU_execution_allowed',
        'calibration_execution_allowed', 'single_submit_allowed', 'formal200_execution_allowed',
        'probability_acceptance', 'actual_two_outer_acceptance'))
    assert review['manifest_sha256'] == MANIFEST_SHA
    policy = review['approved_cpu_invocation']
    assert policy['schema'] == 'EXACT_FULL_PYTEST_INVOCATION_V1' and policy['snapshot_sha256'] == MANIFEST_SHA
    assert policy['cwd'] == policy['rootdir'] == str(CWD)
    names = ['test_deployment_controls_v4.py']; assert policy['test_files'] == names
    inputs = sorted([record(CWD / item['relative_path']) for item in manifest['files']], key=lambda row: row['path'])
    assert len(inputs) == 6 and len({row['path'] for row in inputs}) == 6
    for item in manifest['files']:
        assert record(CWD / item['relative_path'])['sha256'] == item['sha256']
        assert record(CWD / item['relative_path'])['bytes'] == item['bytes']
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert policy['input_file_records'] == inputs and policy['input_files_digest'] == review['controls_files_digest'] == digest
    assert policy['test_file_records'] == [record(CWD / names[0])]
    interpreter = record(Path(PYTHON).resolve())
    interpreter.update(argv0=PYTHON, resolved_path=str(Path(PYTHON).resolve()))
    assert interpreter == policy['interpreter']
    assert interpreter['sha256'] == '94df6ff4b32aafdba90e38c0aed70880fe44ee4cf88ecdf411641158f120c1af'
    dependency = bound(DEPS / 'dependency-receipt.json', '83d9cc2e0f717b38f0a5bbd2d69673350386400f97867e0b06e33cf0c95a8758')
    installed = bound(DEPS / 'installed-target-files.json', 'e0f59cf010ee21f583009b1128ec96bd76c03ad58bc27a5d72adb8c0ae73a093')
    reused = bound(DEPS / 'reused-packages-before.json', 'bef636e97ddbf84333e2e8183c09e1c8616443c3f91cdff604ac6fe683301563')
    assert len(installed) == dependency['installed_target_file_count'] == 123
    assert len(reused) == 3 and sum(len(package['files']) for package in reused) == 51
    verify(installed)
    for package in reused: verify(package['files'])
    assert record(DEPS / 'dependency-receipt.json') == policy['dependency_receipt']
    env = dict(PATH='/usr/bin:/bin', HOME=str(OUT / 'home'), LANG='C.UTF-8', TMPDIR=str(OUT / 'tmp'),
        CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'cpu-tests/pytest-libs-v1') + ':' + str(CWD),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTEST_ADDOPTS='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME=str(OUT / 'hf-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    assert env == policy['execution_environment']
    assert policy['config_policy'] == 'EMPTY_DEV_NULL' and policy['plugin_policy'] == 'BUILTIN_ONLY_AUTOLOAD_DISABLED_NO_CONFTEST'
    options = ['-q', '--disable-warnings', '-p', 'no:cacheprovider', '-c', '/dev/null', '--noconftest', '--rootdir', str(CWD)]
    collect = [PYTHON, '-m', 'pytest', '--collect-only'] + options + ['--basetemp', str(OUT / 'collect-tmp')] + names
    normal = [PYTHON, '-m', 'pytest'] + options + ['--basetemp', str(OUT / 'pytest-tmp'), '--junitxml', str(OUT / 'junit.xml')] + names
    assert collect == policy['collection_command'] and normal == policy['executed_command']
    OUT.mkdir(exist_ok=False)
    for name in ['home', 'tmp', 'hf-cache']: (OUT / name).mkdir()
    common = dict(code_commit=None, scope='FRESH200_DEPLOYMENT_CONTROLS_CPU_ONLY', node='h100-8',
        cwd=str(CWD), rootdir=str(CWD), interpreter=interpreter, test_files=names,
        test_file_records=policy['test_file_records'], input_file_records=inputs, input_files_digest=digest,
        controls_manifest_sha256=MANIFEST_SHA, controls_files_digest=digest,
        execution_environment=env, environment_sanitization=dict(strategy='EXACT_ALLOWLIST_NO_INHERITANCE',
            inherited_environment_forwarded=False, cleared_inputs=dict(PYTEST_ADDOPTS='empty', PYTEST_PLUGINS='absent'),
            removed_input_names=sorted(set(os.environ) - set(env))), config_policy=policy['config_policy'],
        plugin_policy=policy['plugin_policy'], dependency_receipt=policy['dependency_receipt'], review_sha256=sys.argv[2],
        actual_execution=True, gpu_acceptance=False, production_ready=False, formal_iterations=0,
        source_or_runtime_imported=False, actual_slurm_submitted=False)
    write_once(OUT / 'STARTED.json', dict(common, started_utc=utc(), collection_command=collect, command=normal))
    with (OUT / 'collection.log').open('wb') as f:
        collected_exit = subprocess.run(collect, cwd=CWD, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    nodes = [line.strip() for line in (OUT / 'collection.log').read_text().splitlines()
             if line.strip().split('::')[0] in names and '::' in line]
    assert collected_exit == 0 and nodes and len(nodes) == len(set(nodes))
    ids = [canonical(node) for node in nodes]
    write_once(OUT / 'COLLECTION-RESULT.json', dict(common, schema='ACTUAL_FULL_PYTEST_COLLECTION_V1',
        command=collect, exit_code=collected_exit, collection_log=str(OUT / 'collection.log'),
        log_record=record(OUT / 'collection.log'), log_sha256=record(OUT / 'collection.log')['sha256'],
        node_ids=nodes, canonical_case_ids=ids, collected=len(nodes), completed_utc=utc()))
    with (OUT / 'pytest.log').open('wb') as f:
        exit_code = subprocess.run(normal, cwd=CWD, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    rows = ET.parse(OUT / 'junit.xml').findall('.//testcase')
    executed = [row.attrib['classname'] + '::' + row.attrib['name'] for row in rows]
    failed = [executed[i] for i, row in enumerate(rows) if row.find('failure') is not None]
    errors = [executed[i] for i, row in enumerate(rows) if row.find('error') is not None]
    skipped = [executed[i] for i, row in enumerate(rows) if row.find('skipped') is not None]
    counts = dict(collected=len(rows), passed=len(rows)-len(failed)-len(errors)-len(skipped),
        failed=len(failed), errors=len(errors), skipped=len(skipped))
    coverage = len(executed) == len(set(executed)) and sorted(ids) == sorted(executed)
    verify(inputs)
    result = dict(common, command=normal, executed_command=normal, exit_code=exit_code,
        actual_cases=counts, **counts, failure_ids=failed+errors, collection_record=record(OUT / 'COLLECTION-RESULT.json'),
        collection_log_record=record(OUT / 'collection.log'), collected_canonical_case_ids=ids,
        executed_canonical_case_ids=executed, junit_case_ids=executed, exact_collection_execution_coverage=coverage,
        log=str(OUT / 'pytest.log'), log_record=record(OUT / 'pytest.log'), log_sha256=record(OUT / 'pytest.log')['sha256'],
        junit=str(OUT / 'junit.xml'), junit_record=record(OUT / 'junit.xml'), junit_sha256=record(OUT / 'junit.xml')['sha256'],
        completed_utc=utc(), status='PASS_CPU_ONLY' if exit_code == 0 and coverage and not failed+errors+skipped else 'FAILED_CPU_OR_COVERAGE')
    write_once(OUT / 'RESULT.json', result)
    print(json.dumps(dict(status=result['status'], exit_code=exit_code, **counts)))


if __name__ == '__main__':
    main()
