"""One exact offline controls5 pytest invocation, approved by an original review.

Stdlib orchestration only. No source/runtime imports, READY, GPU or Slurm action.
Arguments: original_review_path original_review_sha manifest_path manifest_sha out.
The original report must approve these exact helper bytes and full six inputs.
"""
import datetime
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
CONTROLS = ROOT / 'tools/fresh200-deployment-controls-v5'
SELF = ROOT / 'tools/management/controls-cpu-once-v5.py'
PYTHON = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python'
PYTHON_SHA = '94df6ff4b32aafdba90e38c0aed70880fe44ee4cf88ecdf411641158f120c1af'
LIBS = ROOT / 'cpu-tests/pytest-libs-v1'
DEPS = ROOT / 'cpu-tests/pytest-dependency-v1'
DEP_FILES = {'dependency-receipt.json': '83d9cc2e0f717b38f0a5bbd2d69673350386400f97867e0b06e33cf0c95a8758',
    'installed-target-files.json': 'e0f59cf010ee21f583009b1128ec96bd76c03ad58bc27a5d72adb8c0ae73a093',
    'reused-packages-before.json': 'bef636e97ddbf84333e2e8183c09e1c8616443c3f91cdff604ac6fe683301563'}
NAMES = ('deployment_contract_v5.py', 'prepare_candidate_v5.py', 'verify_physical_v5.py',
    'freeze_ready_v5.py', 'submit_once_v5.py', 'test_deployment_controls_v5.py')
SCOPE = 'FRESH200_DEPLOYMENT_CONTROLS_CPU_ONLY'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def core(row):
    return {key: row[key] for key in ('path', 'bytes', 'sha256')}


def checked_path(value):
    path = Path(value)
    require(type(value) is str and path.is_absolute() and '..' not in path.parts
            and str(path) == value and path.resolve(strict=True) == path,
            'Noncanonical input path')
    return path


def read_bytes(row):
    path = checked_path(row['path'])
    require(type(row['bytes']) is int and row['bytes'] >= 0 and type(row['sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Invalid bound record')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd); require(stat.S_ISREG(before.st_mode) and before.st_size == row['bytes'], 'Input type/size differs')
        blocks, size = [], 0
        while True:
            block = os.read(fd, min(1024 * 1024, row['bytes'] - size + 1))
            if not block:
                break
            size += len(block); require(size <= row['bytes'], 'Input grew'); blocks.append(block)
        raw = b''.join(blocks); after = os.fstat(fd); named = os.lstat(path)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                                  value.st_mtime_ns, value.st_ctime_ns)
        require(identity(before) == identity(after) == identity(named)
                and path.resolve(strict=True) == path and hashlib.sha256(raw).hexdigest() == row['sha256'],
                'Input identity/SHA changed')
        return raw
    finally:
        os.close(fd)


def record(path):
    path = checked_path(str(path)); raw = path.read_bytes()
    row = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    require(read_bytes(row) == raw, 'Record changed while hashing')
    return row


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key'); result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def bound(path, digest):
    row = record(path); require(row['sha256'] == digest, 'Different original JSON SHA')
    return parse_json(read_bytes(row)), row


def records_checked(rows):
    require(type(rows) is list and rows, 'Missing complete input inventory')
    paths = []
    for row in rows:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'Invalid input core')
        require(row['path'] not in paths, 'Duplicate input'); paths.append(row['path']); read_bytes(row)
    return rows


def digest(rows):
    return hashlib.sha256(json.dumps(sorted([core(row) for row in rows], key=lambda row: row['path']),
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def controls_directory():
    directory = checked_path(str(CONTROLS))
    entries = list(directory.iterdir())
    require({entry.name for entry in entries} == set(NAMES)
            and all(not entry.is_symlink() and entry.is_file() for entry in entries),
            'Controls directory contains extra/missing/nonregular entries')
    return sorted(entry.name for entry in entries)


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def canonical(node, filenames):
    names = [name for name in filenames if node.startswith(name + '::')]
    require(len(names) == 1, 'Undeclared collection file')
    filename = names[0]; suffix = node[len(filename) + 2:]
    before, bracket, parameters = suffix.partition('['); parts = before.split('::')
    require(parts and all(re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', part) for part in parts)
            and parts[-1].startswith('test') and (not bracket or parameters.endswith(']')),
            'Nonstandard full collection ID')
    classname = filename[:-3].replace('/', '.') + ('.' + '.'.join(parts[:-1]) if len(parts) > 1 else '')
    return classname + '::' + parts[-1] + bracket + parameters


def collection_ids(raw, filenames):
    nodes, totals = [], []
    for line in raw.decode('utf-8').splitlines():
        if any(line.startswith(name + '::') for name in filenames):
            canonical(line, filenames); nodes.append(line)
        elif re.match(r'\S+\.py::', line):
            raise ValueError('Undeclared collection path')
        matched = re.fullmatch(r'(\d+) tests? collected in [0-9.]+s', line.strip())
        if matched:
            totals.append(int(matched.group(1)))
        require('deselected' not in line.lower() and 'error collecting' not in line.lower(), 'Filtered/failed collection')
    cases = [canonical(node, filenames) for node in nodes]
    require(nodes and len(nodes) == len(set(nodes)) and len(cases) == len(set(cases)) and totals == [len(nodes)],
            'Incomplete or duplicate raw full collection')
    return nodes, cases


def junit(raw):
    require(b'<!DOCTYPE' not in raw and b'<!ENTITY' not in raw, 'Unexpected JUnit entity')
    root = ET.fromstring(raw); require(root.tag in ('testsuites', 'testsuite'), 'Unexpected JUnit root')
    suites = [root] if root.tag == 'testsuite' else list(root)
    require(suites and all(s.tag == 'testsuite' and s.find('testsuite') is None for s in suites), 'Ambiguous JUnit suites')
    ids, failure_ids, hosts = [], [], set(); counts = dict(collected=0, passed=0, failed=0, errors=0, skipped=0)
    for suite in suites:
        hosts.add(suite.get('hostname')); local = dict.fromkeys(counts, 0)
        for case in suite.findall('testcase'):
            require(case.get('classname') and case.get('name'), 'Missing raw JUnit identity')
            identity = case.get('classname') + '::' + case.get('name'); ids.append(identity)
            markers = [(tag, len(case.findall(tag))) for tag in ('failure', 'error', 'skipped')]
            require(sum(number for _, number in markers) <= 1, 'Ambiguous raw JUnit outcome')
            outcome = next(({'failure': 'failed', 'error': 'errors', 'skipped': 'skipped'}[tag]
                           for tag, number in markers if number), 'passed')
            local['collected'] += 1; local[outcome] += 1
            if outcome in ('failed', 'errors'):
                failure_ids.append(identity)
        for attribute, key in [('tests', 'collected'), ('failures', 'failed'), ('errors', 'errors'), ('skipped', 'skipped')]:
            require(suite.get(attribute) is not None and int(suite.get(attribute)) == local[key], 'Raw JUnit totals differ')
        for key in counts:
            counts[key] += local[key]
    require(ids and len(ids) == len(set(ids)) and hosts == {'h100-8'}, 'Empty/duplicate/wrong-host JUnit')
    return ids, sorted(failure_ids), counts


def main():
    require(len(sys.argv) == 6 and not sys.flags.optimize, 'Expected exact five management args without -O')
    require(socket.gethostname().split('.')[0] == 'h100-8' and Path(__file__).resolve() == SELF, 'Wrong management host/helper')
    review_path, review_sha, manifest_path, manifest_sha, output_path = sys.argv[1:]
    review, review_record = bound(review_path, review_sha); manifest, manifest_record = bound(manifest_path, manifest_sha)
    policy = review['approved_cpu_policy']
    require(review['schema'] == 'STRICT_CLEAN_DEPLOYMENT_CONTROLS_PREEXEC_REVIEW_V5'
            and review['status'] in ('PASS', 'PASS_STATIC_ONLY_CPU_SCOPE')
            and review['scope'] == 'FRESH200_DEPLOYMENT_CONTROLS_PREEXEC_STATIC_ONLY'
            and review['independent_review'] is True and review['blocking_findings'] == []
            and review['actual_gpu_acceptance'] is False and review['production_READY'] is False
            and review['submitted'] is False and type(review['formal_iterations']) is int and review['formal_iterations'] == 0
            and review['manifest_sha256'] == manifest_sha and review['controls_files_digest'] == manifest['controls_files_digest'],
            'Wrong original controls5 static approval')
    origin = review['reviewer']; decision = review['decision']
    require(origin['fork_turns'] == 'none' and origin['root_conversation_inherited'] is False
            and origin['implementation_authorship'] is False and origin['independent_after_freeze'] is True
            and type(decision['blocking_findings']) is int and decision['blocking_findings'] == 0
            and decision['whole_controls_static_pass'] is True and decision['CPU_fixture_execution_authorized_by_this_report'] is True
            and all(decision[key] is False for key in ('READY_allowed', 'GPU_execution_allowed', 'calibration_execution_allowed',
                'single_submit_allowed', 'formal200_execution_allowed', 'probability_acceptance', 'actual_two_outer_acceptance')),
            'Original controls approval inflates scope or lacks independence')
    self_record = record(__file__)
    require(policy['schema'] == 'EXACT_FULL_PYTEST_INVOCATION_V1' and policy['role'] == 'controls'
            and policy['execution_scope'] == SCOPE and policy['single_execution_only'] is True
            and policy['GPU_allowed'] is False and policy['READY_allowed'] is False
            and policy['Slurm_submission_allowed'] is False and type(policy['formal_iterations_contributed']) is int
            and policy['formal_iterations_contributed'] == 0 and policy['management_helper'] == self_record == manifest['management_helper']
            and policy['snapshot_sha256'] == manifest_sha and policy['original_review_path'] == review_path,
            'Wrong single-use management helper/policy binding')
    output = Path(output_path)
    require(output.parent == ROOT / 'cpu-tests' and str(output) == policy['output_directory']
            and output.name == 'controls-v5-final' and output.resolve() == output and not output.exists(), 'Wrong/finalized CPU output')
    require(policy['cwd'] == policy['rootdir'] == str(CONTROLS)
            and policy['test_files'] == manifest['cpu_required_test_files'] == ['test_deployment_controls_v5.py']
            and policy['config_policy'] == 'EMPTY_DEV_NULL' and policy['plugin_policy'] == 'BUILTIN_ONLY_AUTOLOAD_DISABLED_NO_CONFTEST',
            'Unapproved file/config/plugin selection')
    inputs = records_checked(policy['input_file_records'])
    require(inputs == manifest['controls_target_files'] and {r['path'] for r in inputs} == {str(CONTROLS / n) for n in NAMES}
            and digest(inputs) == policy['input_files_digest'] == manifest['controls_files_digest'], 'Incomplete frozen six controls inputs')
    directory_before = controls_directory()
    selected = records_checked(policy['test_file_records'])
    require(selected == [next(r for r in inputs if r['path'] == str(CONTROLS / policy['test_files'][0]))], 'Different selected fixture SHA')
    dependencies = {name: bound(DEPS / name, sha)[0] for name, sha in DEP_FILES.items()}
    dep = dependencies['dependency-receipt.json']; installed = dependencies['installed-target-files.json']; reused = dependencies['reused-packages-before.json']
    require(len(installed) == dep['installed_target_file_count'] == 123 and len(reused) == 3
            and sum(len(package['files']) for package in reused) == 51 and dep['host'] == 'h100-8'
            and dep['python_executable'] == PYTHON and dep['target_library'] == str(LIBS), 'Wrong original dependency closure')
    records_checked(installed)
    for package in reused:
        records_checked(package['files'])
    interpreter = record(Path(PYTHON).resolve()); interpreter.update(argv0=PYTHON, resolved_path=str(Path(PYTHON).resolve()))
    require(interpreter['sha256'] == PYTHON_SHA and interpreter == policy['interpreter'], 'Different exact CPU interpreter')
    environment = dict(PATH='/usr/bin:/bin', HOME=str(output / 'home'), LANG='C.UTF-8', TMPDIR=str(output / 'tmp'),
        CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(LIBS) + ':' + str(CONTROLS), PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HOME=str(output / 'hf-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    options = ['-q', '--disable-warnings', '-p', 'no:cacheprovider', '-c', '/dev/null', '--noconftest', '--rootdir', str(CONTROLS)]
    collect = [PYTHON, '-m', 'pytest', '--collect-only'] + options + ['--basetemp', str(output / 'collect-tmp')] + policy['test_files']
    normal = [PYTHON, '-m', 'pytest'] + options + ['--basetemp', str(output / 'pytest-tmp'), '--junitxml', str(output / 'junit.xml')] + policy['test_files']
    require(environment == policy['execution_environment'] and collect == policy['collection_command']
            and normal == policy['executed_command'] and policy['collection_basetemp'] == str(output / 'collect-tmp')
            and policy['run_basetemp'] == str(output / 'pytest-tmp') and policy['junit_path'] == str(output / 'junit.xml')
            and policy['dependency_receipt'] == record(DEPS / 'dependency-receipt.json'), 'Exact original command/environment differs')
    # Freeze originals again before exclusively claiming the one output namespace.
    require(bound(review_path, review_sha)[0] == review and bound(manifest_path, manifest_sha)[0] == manifest,
            'Original approval changed before claim')
    require(records_checked(inputs) == inputs and controls_directory() == directory_before,
            'Controls inputs/directory changed before claim')
    output.mkdir(exist_ok=False)
    for name in ('home', 'tmp', 'hf-cache'):
        (output / name).mkdir()
    common = dict(code_commit=None, scope=SCOPE, node='h100-8', cwd=str(CONTROLS), rootdir=str(CONTROLS),
        interpreter=interpreter, test_files=policy['test_files'], test_file_records=selected, input_file_records=inputs,
        input_files_digest=digest(inputs), controls_manifest_sha256=manifest_sha, controls_files_digest=digest(inputs),
        review_sha256=review_sha, management_helper=self_record, dependency_receipt=policy['dependency_receipt'],
        execution_environment=environment, environment_sanitization=dict(strategy='EXACT_ALLOWLIST_NO_INHERITANCE',
            inherited_environment_forwarded=False, cleared_inputs=dict(PYTEST_ADDOPTS='empty', PYTEST_PLUGINS='absent'),
            removed_input_names=sorted(set(os.environ) - set(environment))), config_policy=policy['config_policy'],
        plugin_policy=policy['plugin_policy'], actual_execution=True, gpu_acceptance=False, production_ready=False,
        formal_iterations=0, source_or_runtime_imported=False, actual_slurm_submitted=False,
        controls_directory_names_before=directory_before)
    write_once(output / 'STARTED.json', dict(common, command=normal, collection_command=collect,
        input_file_records_before=inputs, input_files_digest_before=digest(inputs), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (output / 'collection.log').open('xb') as stream:
        collection_exit = subprocess.run(collect, cwd=CONTROLS, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
    raw_collection = (output / 'collection.log').read_bytes(); parse_error = None
    try:
        nodes, collected_ids = collection_ids(raw_collection, policy['test_files'])
    except (ValueError, UnicodeError) as error:
        nodes, collected_ids = [], []
        parse_error = dict(type=type(error).__name__, message=str(error))
    collection_record = dict(common, schema='ACTUAL_FULL_PYTEST_COLLECTION_V1', command=collect, exit_code=collection_exit,
        collection_log=str(output / 'collection.log'), log_record=record(output / 'collection.log'),
        log_sha256=hashlib.sha256(raw_collection).hexdigest(), node_ids=nodes, canonical_case_ids=collected_ids,
        collected=len(nodes) if parse_error is None else None, collection_parse_error=parse_error,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_once(output / 'COLLECTION-RESULT.json', collection_record)
    require(collection_exit == 0 and parse_error is None, 'Failed/unparsed original collection; normal run not started')
    with (output / 'pytest.log').open('xb') as stream:
        exit_code = subprocess.run(normal, cwd=CONTROLS, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
    raw_junit = (output / 'junit.xml').read_bytes(); executed_ids, failures, counts = junit(raw_junit)
    after = [record(row['path']) for row in inputs]; coverage = sorted(collected_ids) == sorted(executed_ids)
    directory_after = controls_directory()
    successful = exit_code == 0 and coverage and after == inputs and all(counts[key] == 0 for key in ('failed', 'errors', 'skipped'))
    result = dict(common, command=normal, executed_command=normal, exit_code=exit_code, actual_cases=counts, **counts,
        failure_ids=failures, collection_record=record(output / 'COLLECTION-RESULT.json'),
        collection_result_record=record(output / 'COLLECTION-RESULT.json'), collection_log_record=record(output / 'collection.log'),
        collected_canonical_case_ids=collected_ids, executed_canonical_case_ids=executed_ids, junit_case_ids=executed_ids,
        exact_collection_execution_coverage=coverage, input_file_records_after=after, input_files_digest_after=digest(after),
        controls_directory_names_after=directory_after,
        log=str(output / 'pytest.log'), log_record=record(output / 'pytest.log'), log_sha256=record(output / 'pytest.log')['sha256'],
        junit=str(output / 'junit.xml'), junit_record=record(output / 'junit.xml'), junit_sha256=hashlib.sha256(raw_junit).hexdigest(),
        status='PASS_CPU_ONLY' if successful else 'FAILED_CPU_OR_COVERAGE_OR_INPUT_IDENTITY',
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_once(output / 'RESULT.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'collected', 'passed', 'failed', 'errors', 'skipped', 'exit_code')}))
    require(successful, 'Original controls CPU was not a full successful unchanged-input run')


if __name__ == '__main__':
    main()
