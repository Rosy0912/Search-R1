"""One offline source/runtime CPU invocation bound by an original independent review.

This stdlib management helper imports neither trainer nor runtime. It cannot
submit Slurm jobs, establish READY, authorize GPUs, or overwrite an earlier run.
Arguments: review_path review_sha256 snapshot_path snapshot_sha256 output_path.
The reviewer must bind the exact approved_cpu_policy and this helper's bytes.
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
PYTHON = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python'
PYTHON_SHA = '94df6ff4b32aafdba90e38c0aed70880fe44ee4cf88ecdf411641158f120c1af'
DEPS = ROOT / 'cpu-tests/pytest-dependency-v1'
LIBS = ROOT / 'cpu-tests/pytest-libs-v1'
DEP_FILES = {
    'dependency-receipt.json': '83d9cc2e0f717b38f0a5bbd2d69673350386400f97867e0b06e33cf0c95a8758',
    'installed-target-files.json': 'e0f59cf010ee21f583009b1128ec96bd76c03ad58bc27a5d72adb8c0ae73a093',
    'reused-packages-before.json': 'bef636e97ddbf84333e2e8183c09e1c8616443c3f91cdff604ac6fe683301563',
}
SCOPES = {'source': 'ALL_CURRENT_CPU_REGRESSIONS',
          'runtime': 'FRESH200_RUNTIME_CPU_REGRESSIONS'}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def bound(path, digest):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, 'Bound JSON bytes differ: ' + str(path))
    return json.loads(data)


def rows_checked(rows):
    require(isinstance(rows, list) and rows, 'Missing input closure')
    found, seen = [], set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}, 'Invalid input record')
        path = Path(row['path'])
        require(path.is_absolute() and str(path) not in seen and type(row['bytes']) is int,
                'Duplicate/invalid input path')
        seen.add(str(path))
        actual = record(path)
        require(actual == row, 'Actual input bytes differ: ' + str(path))
        found.append(actual)
    return found


def digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relative_digest(rows, directory):
    normalized = [dict(relative_path=str(Path(row['path']).relative_to(directory)),
                       bytes=row['bytes'], sha256=row['sha256']) for row in rows]
    return digest(sorted(normalized, key=lambda row: row['relative_path']))


def directory_inventory(directory, rows, *, omit_git=False):
    """Bind the actual import tree, including ignored/untracked extra files."""
    require(directory.is_dir() and not directory.is_symlink() and
            directory.resolve(strict=True) == directory, 'Noncanonical input directory')
    expected = sorted(str(Path(row['path']).relative_to(directory)) for row in rows)
    actual = []
    for parent, dirs, files in os.walk(directory, followlinks=False):
        parent = Path(parent)
        if omit_git and parent == directory:
            dirs[:] = [name for name in dirs if name != '.git']
            files = [name for name in files if name != '.git']
        for name in dirs:
            require(not (parent / name).is_symlink(), 'Undeclared import-directory symlink')
        for name in files:
            path = parent / name
            require(path.is_file() and not path.is_symlink(), 'Nonregular import-tree input')
            actual.append(str(path.relative_to(directory)))
    require(sorted(actual) == expected, 'Actual input directory differs from full declared closure')
    return dict(directory=str(directory), actual_file_count=len(actual),
                actual_directory_file_names=sorted(actual), exact_declared_inventory=True)


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def canonical(node):
    module, separator, suffix = node.partition('::')
    require(separator and module.endswith('.py') and suffix, 'Invalid collected pytest node ID')
    plain, bracket, parameters = suffix.partition('[')
    parts = plain.split('::')
    require(all(parts), 'Invalid collected pytest class/function ID')
    name = parts[-1] + (bracket + parameters if bracket else '')
    classname = module[:-3].replace('/', '.') + (
        '.' + '.'.join(parts[:-1]) if len(parts) > 1 else '')
    return classname + '::' + name


def git_identity(source, source_rows, expected_commit):
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    clean = not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')[:-1]
    require(commit == expected_commit and clean, 'Source commit/clean status differs')
    require(len(names) == len(source_rows) and {str(source / name) for name in names} ==
            {row['path'] for row in source_rows}, 'Full tracked source coverage differs')
    return dict(commit=commit, clean=clean, source_files=len(source_rows),
                actual_source_input_digest=digest(source_rows))


def main():
    require(len(sys.argv) == 6, 'Expected five management arguments')
    review_path, review_sha, snapshot_path, snapshot_sha, output_path = sys.argv[1:]
    require(socket.gethostname().split('.')[0] == 'h100-8', 'Wrong CPU-test host')
    review = bound(review_path, review_sha)
    snapshot = bound(snapshot_path, snapshot_sha)
    policy = review['approved_cpu_policy']
    role = policy['role']
    require(role in SCOPES and policy['execution_scope'] == SCOPES[role], 'Wrong CPU-only role/scope')
    if role == 'source':
        require(review['schema'] == 'gigpo-e1-fresh200-v11-preexec-review' and
                review['status'] == 'PASS_STATIC_ONLY_CPU_SCOPE' and
                review['scope'] == 'FRESH200_SOURCE_V11_STATIC_CPU_ONLY', 'Wrong source11 original static review')
    else:
        require(type(review['schema']) is int and review['schema'] == 1 and
                review['status'] == 'PASS' and review['runtime_version'] == 12 and
                review['scope'] == 'INDEPENDENT_FROZEN_FRESH200_CONTINUATION_V12_STATIC_PREEXEC_REVIEW_ONLY',
                'Wrong runtime12 original static review')
    require(review['blocking_findings'] == [] and review['independent_review'] is True and
            review['actual_gpu_acceptance'] is False and review['production_READY'] is False and
            review['submitted'] is False and type(review['formal_iterations']) is int and
            review['formal_iterations'] == 0,
            'Original report is not an independent CPU-only approval')
    require(policy['snapshot_sha256'] == snapshot_sha and
            policy['original_review_path'] == review_path and policy['single_execution_only'] is True,
            'Original report/snapshot invocation differs')
    require(policy['GPU_allowed'] is False and policy['READY_allowed'] is False and
            type(policy['formal_iterations_contributed']) is int and policy['formal_iterations_contributed'] == 0 and
            policy['Slurm_submission_allowed'] is False, 'CPU policy overclaims permission')
    require(record(__file__) == policy['management_helper'], 'Management helper bytes/path differ')
    output, source, cwd = Path(output_path), Path(policy['source_repo']), Path(policy['cwd'])
    require(output.parent == ROOT / 'cpu-tests' and str(output) == policy['output_directory'],
            'Output is outside the single approved CPU invocation')
    require(source.parent == ROOT and source.name.startswith('repo-v11-') and
            source == Path(policy['source_directory']), 'Wrong isolated source11 checkout')
    require(str(cwd) == policy['rootdir'] and (cwd == source if role == 'source' else
            cwd == ROOT / 'tools/fresh200-continuation-v12'), 'Wrong source/runtime CPU cwd')
    source_rows = rows_checked(policy['source_input_files'])
    runtime_rows = rows_checked(policy['runtime_input_files']) if role == 'runtime' else []
    require(role != 'source' or policy['runtime_input_files'] == [], 'Source CPU closure imports runtime')
    require(all(Path(row['path']).is_relative_to(source) for row in source_rows), 'Source path escaped')
    require(all(Path(row['path']).is_relative_to(cwd) for row in runtime_rows), 'Runtime path escaped')
    inputs = source_rows + runtime_rows
    require(policy['input_file_records'] == inputs and policy['input_files_digest'] == digest(inputs),
            'Complete approved input closure differs')
    require(policy['source_input_files_digest'] == digest(source_rows), 'Source input digest differs')
    source_snapshot_record = policy['source_snapshot_record']
    require(record(source_snapshot_record['path']) == source_snapshot_record and
            source_snapshot_record['sha256'] == policy['source_snapshot_sha256'], 'Source snapshot receipt differs')
    source_snapshot = bound(source_snapshot_record['path'], source_snapshot_record['sha256'])
    source_mapped = [dict(path=str(source / row['relative_path']), bytes=row['bytes'], sha256=row['sha256'])
                     for row in source_snapshot['full_tracked_source_files']]
    require(source_mapped == source_rows and source_snapshot['new_code_commit'] == policy['code_commit'],
            'Full source snapshot coverage/commit differs')
    require(source_snapshot['source_tracked_files_digest'] == relative_digest(source_rows, source),
            'Source snapshot relative digest differs')
    if role == 'source':
        require(record(snapshot_path) == source_snapshot_record, 'Source invocation bound another snapshot')
    snapshot_rows = snapshot['full_tracked_source_files'] if role == 'source' else snapshot['files']
    snapshot_directory = source if role == 'source' else cwd
    mapped = [dict(path=str(snapshot_directory / row['relative_path']), bytes=row['bytes'], sha256=row['sha256'])
              for row in snapshot_rows]
    require(mapped == (source_rows if role == 'source' else runtime_rows), 'Snapshot full input mapping differs')
    if role == 'runtime':
        require(type(snapshot['runtime_version']) is int and snapshot['runtime_version'] == 12 and
                type(snapshot['file_count']) is int and snapshot['file_count'] == len(runtime_rows),
                'Actual runtime12 full file count differs')
    names = policy['test_files']
    require(isinstance(names, list) and names and len(names) == len(set(names)) and
            all(not Path(name).is_absolute() and '..' not in Path(name).parts and
                Path(name).name.startswith('test_') and name.endswith('.py') for name in names),
            'Invalid/duplicate selected CPU files')
    require(names == snapshot['cpu_required_test_files'], 'Complete declared CPU selection differs')
    test_rows = rows_checked(policy['test_file_records'])
    require([row['path'] for row in test_rows] == [str(cwd / name) for name in names] and
            all(row in inputs for row in test_rows), 'Selected tests lack exact full-input coverage')
    dep = bound(DEPS / 'dependency-receipt.json', DEP_FILES['dependency-receipt.json'])
    installed = bound(DEPS / 'installed-target-files.json', DEP_FILES['installed-target-files.json'])
    reused = bound(DEPS / 'reused-packages-before.json', DEP_FILES['reused-packages-before.json'])
    require(len(installed) == dep['installed_target_file_count'] == 123 and
            len(reused) == 3 and sum(len(package['files']) for package in reused) == 51,
            'Original CPU dependency coverage differs')
    require(dep['host'] == 'h100-8' and dep['python_executable'] == PYTHON and
            dep['target_library'] == str(LIBS), 'CPU dependency identity differs')
    rows_checked(installed)
    for package in reused:
        rows_checked(package['files'])
    before = git_identity(source, source_rows, policy['code_commit'])
    before['source_directory_inventory'] = directory_inventory(source, source_rows, omit_git=True)
    before['runtime_directory_inventory'] = (directory_inventory(cwd, runtime_rows) if runtime_rows else None)
    before.update(runtime_files=len(runtime_rows), runtime_manifest_sha256=(snapshot_sha if role == 'runtime' else None),
                  source_snapshot_sha256=policy['source_snapshot_sha256'],
                  source_digest=relative_digest(source_rows, source))
    interpreter = record(Path(PYTHON).resolve())
    interpreter.update(argv0=PYTHON, resolved_path=str(Path(PYTHON).resolve()))
    require(interpreter['sha256'] == PYTHON_SHA and interpreter == policy['interpreter'], 'Interpreter differs')
    environment = dict(PATH='/usr/bin:/bin', HOME=str(output / 'home'), LANG='C.UTF-8',
        TMPDIR=str(output / 'tmp'), CUDA_VISIBLE_DEVICES='',
        PYTHONPATH=':'.join([str(LIBS), str(source)] + ([str(cwd)] if role == 'runtime' else [])),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTEST_ADDOPTS='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME=str(output / 'hf-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    require(environment == policy['execution_environment'], 'Exact allowlisted CPU environment differs')
    options = ['-q', '--disable-warnings', '-p', 'no:cacheprovider', '-c', '/dev/null', '--noconftest', '--rootdir', str(cwd)]
    collect = [PYTHON, '-m', 'pytest', '--collect-only'] + options + ['--basetemp', str(output / 'collect-tmp')] + names
    normal = [PYTHON, '-m', 'pytest'] + options + ['--basetemp', str(output / 'pytest-tmp'), '--junitxml', str(output / 'junit.xml')] + names
    require(collect == policy['collection_command'] and normal == policy['executed_command'], 'Exact approved pytest argv differs')
    output.mkdir(exist_ok=False)
    for name in ('home', 'tmp', 'hf-cache'):
        (output / name).mkdir()
    common = dict(code_commit=before['commit'], scope=SCOPES[role], node='h100-8',
        cwd=str(cwd), rootdir=str(cwd), interpreter=interpreter, test_files=names, test_file_records=test_rows,
        input_file_records=inputs, input_files_digest=digest(inputs), execution_environment=environment,
        environment_sanitization=dict(strategy='EXACT_ALLOWLIST_NO_INHERITANCE', inherited_environment_forwarded=False,
            cleared_inputs=dict(PYTEST_ADDOPTS='empty', PYTEST_PLUGINS='absent'),
            removed_input_names=sorted(set(os.environ) - set(environment))),
        config_policy='EMPTY_DEV_NULL', plugin_policy='BUILTIN_ONLY_AUTOLOAD_DISABLED_NO_CONFTEST',
        source_snapshot_sha256=policy['source_snapshot_sha256'], review_sha256=review_sha,
        dependency_receipt=record(DEPS / 'dependency-receipt.json'), management_helper=record(__file__),
        actual_execution=True, gpu_acceptance=False, production_ready=False, formal_iterations=0)
    if role == 'runtime':
        common['runtime_snapshot_sha256'] = snapshot_sha
    write_once(output / 'STARTED.json', dict(common, started_utc=utc(), collection_command=collect,
                                           command=normal, source_before=before))
    with (output / 'collection.log').open('wb') as stream:
        collection_exit = subprocess.run(collect, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
    nodes = [line.strip() for line in (output / 'collection.log').read_text().splitlines()
             if '::' in line and line.strip().split('::')[0] in names]
    cases = [canonical(node) for node in nodes]
    collection = dict(common, schema='ACTUAL_FULL_PYTEST_COLLECTION_V1', command=collect,
        exit_code=collection_exit, collection_log=str(output / 'collection.log'),
        log_record=record(output / 'collection.log'), log_sha256=record(output / 'collection.log')['sha256'],
        node_ids=nodes, canonical_case_ids=cases, collected=len(nodes), completed_utc=utc())
    write_once(output / 'COLLECTION-RESULT.json', collection)
    require(collection_exit == 0 and nodes and len(nodes) == len(set(nodes)), 'CPU collection failed; preserved raw evidence')
    with (output / 'pytest.log').open('wb') as stream:
        exit_code = subprocess.run(normal, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
    rows = ET.parse(output / 'junit.xml').findall('.//testcase')
    executed = [row.attrib['classname'] + '::' + row.attrib['name'] for row in rows]
    failed = [executed[i] for i, row in enumerate(rows) if row.find('failure') is not None]
    errors = [executed[i] for i, row in enumerate(rows) if row.find('error') is not None]
    skipped = [executed[i] for i, row in enumerate(rows) if row.find('skipped') is not None]
    coverage = sorted(cases) == sorted(executed) and len(executed) == len(set(executed))
    require(rows_checked(source_rows) == source_rows, 'Source bytes changed during CPU run')
    if runtime_rows:
        require(rows_checked(runtime_rows) == runtime_rows, 'Runtime bytes changed during CPU run')
    after = git_identity(source, source_rows, policy['code_commit'])
    after['source_directory_inventory'] = directory_inventory(source, source_rows, omit_git=True)
    after['runtime_directory_inventory'] = (directory_inventory(cwd, runtime_rows) if runtime_rows else None)
    after.update(runtime_files=len(runtime_rows), runtime_manifest_sha256=(snapshot_sha if role == 'runtime' else None),
                 source_snapshot_sha256=policy['source_snapshot_sha256'],
                 source_digest=relative_digest(source_rows, source))
    require(after == before, 'Full source/runtime identity changed during CPU run')
    counts = dict(collected=len(nodes), passed=len(rows) - len(failed) - len(errors) - len(skipped),
                  failed=len(failed), errors=len(errors), skipped=len(skipped))
    result = dict(common, command=normal, executed_command=normal, exit_code=exit_code,
        actual_cases=counts, **counts, failure_ids=failed + errors,
        collection_record=record(output / 'COLLECTION-RESULT.json'), collection_result_record=record(output / 'COLLECTION-RESULT.json'),
        collection_log_record=record(output / 'collection.log'), collected_canonical_case_ids=cases,
        executed_canonical_case_ids=executed, junit_case_ids=executed, exact_collection_execution_coverage=coverage,
        log=str(output / 'pytest.log'), log_record=record(output / 'pytest.log'), log_sha256=record(output / 'pytest.log')['sha256'],
        junit=str(output / 'junit.xml'), junit_record=record(output / 'junit.xml'), junit_sha256=record(output / 'junit.xml')['sha256'],
        source_before=before, source_after=after, completed_utc=utc(),
        status='PASS_CPU_ONLY' if exit_code == 0 and coverage and not failed + errors + skipped else 'FAILED_CPU_OR_COVERAGE')
    write_once(output / 'RESULT.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'collected', 'passed', 'failed', 'errors', 'skipped', 'exit_code', 'failure_ids')}))


if __name__ == '__main__':
    main()
