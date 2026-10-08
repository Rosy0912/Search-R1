"""Static-only author recorder for one obsolete CPU exception expectation.

Reads project inputs exclusively as bytes/AST and reads original CPU artifacts
as data. No project/helper/test import, collection or execution occurs.
"""
import ast
import copy
import hashlib
import json
import os
import stat
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT / 'work/gigpo-e1-fresh200-compat11-20261006'
SOURCE = ROOT / 'work/gigpo-e1-fresh200-cpufixturefix-20261007'
RUNTIME = ROOT / 'work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v12'
PARENT_SNAPSHOT = ROOT / 'outputs/fresh200-compatibility-20261006/FRESH200-SOURCE-STATIC-SNAPSHOT-v11.json'
PARENT_MANIFEST = ROOT / 'outputs/gigpo-e1-policy-safety/FRESH200-CONTINUATION-STATIC-MANIFEST-v12.json'
SNAPSHOT = PARENT_SNAPSHOT.with_name('FRESH200-SOURCE-STATIC-SNAPSHOT-v11-cpufixturefix1.json')
MANIFEST = PARENT_MANIFEST.with_name('FRESH200-CONTINUATION-STATIC-MANIFEST-v12-cpufixturefix1.json')
SOURCE_RAW = ROOT / 'outputs/gigpo-e1-instruct200/cpu-source-source-v11-4a45458'
RUNTIME_RAW = ROOT / 'outputs/gigpo-e1-instruct200/cpu-runtime-runtime-v12-9e45099'
TARGET = 'tests/baselines/test_policy_safety_actor.py'
MESSAGE = 'Missing independent GPU calibration acceptance artifact'


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def record(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def inventory(directory, omit_git=False):
    rows = []
    for parent, dirs, files in os.walk(directory, followlinks=False):
        if omit_git and Path(parent) == directory:
            dirs[:] = [name for name in dirs if name != '.git']
        for name in dirs:
            require(not (Path(parent) / name).is_symlink(), 'Full input directory alias')
        for name in files:
            path = Path(parent) / name
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Full input file alias/nonregular')
            row = record(path)
            rows.append(dict(relative_path=str(path.relative_to(directory)), bytes=row['bytes'], sha256=row['sha256']))
    return sorted(rows, key=lambda row: row['relative_path'])


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def raw_evidence(directory, expected_status, expected_cases):
    names = ('STARTED.json', 'COLLECTION-RESULT.json', 'RESULT.json', 'collection.log',
             'pytest.log', 'junit.xml', 'RAW-PULL-RECEIPT.json')
    rows = [record(directory / name) for name in names]
    result = json.loads((directory / 'RESULT.json').read_text())
    require(result['status'] == expected_status and result['actual_cases'] == expected_cases,
            'Original actual CPU result differs')
    require(result['exact_collection_execution_coverage'] is True, 'Original CPU coverage differs')
    require(result['code_commit'] == '4a454588ec65db1aff3c6f4738fb0590f1caa928', 'Original CPU source identity differs')
    cases = ET.parse(directory / 'junit.xml').findall('.//testcase')
    require(len(cases) == expected_cases['collected'], 'Original JUnit coverage differs')
    require(sum(case.find('failure') is not None for case in cases) == expected_cases['failed'], 'Original JUnit failures differ')
    require(sum(case.find('error') is not None for case in cases) == expected_cases['errors'], 'Original JUnit errors differ')
    require(sum(case.find('skipped') is not None for case in cases) == expected_cases['skipped'], 'Original JUnit skips differ')
    return dict(raw_directory=str(directory), original_file_records=rows, original_status=result['status'],
        original_actual_cases=result['actual_cases'], original_exit_code=result['exit_code'],
        original_exact_collection_execution_coverage=True, original_code_commit=result['code_commit'],
        accepted_for_successor_inputs=False)


def fixture_expected_ast_only():
    old = ast.parse((PARENT / TARGET).read_bytes())
    actual = ast.parse((SOURCE / TARGET).read_bytes())
    expected = copy.deepcopy(old)
    method = next(node for node in expected.body if isinstance(node, ast.FunctionDef)
        and node.name == 'test_entry_errors_reject_before_any_optimizer_write')
    assignment = next(node for node in method.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'expected' for target in node.targets))
    require(isinstance(assignment.value, ast.IfExp) and isinstance(assignment.value.body, ast.Name)
        and assignment.value.body.id == 'FileNotFoundError' and assignment.value.orelse.id == 'KeyError', 'Original fixture expectation changed')
    assignment.value.body.id = 'ValueError'
    original_with = next(node for node in method.body if isinstance(node, ast.With))
    require(len(original_with.items) == 1 and original_with.items[0].optional_vars is None, 'Original fixture catch shape differs')
    original_with.items[0].optional_vars = ast.Name(id='caught', ctx=ast.Store())
    exact_message = ast.parse("if failure == 'calibration_missing':\n    assert str(caught.value) == " + repr(MESSAGE)).body[0]
    method.body.insert(method.body.index(original_with) + 1, exact_message)
    require(ast.dump(expected, include_attributes=False) == ast.dump(actual, include_attributes=False),
        'Fixture AST change exceeds exact missing-artifact exception/message expectation')
    return dict(expected_ast_equal=True, exact_allowed_changes=[
        'calibration_missing expected FileNotFoundError -> ValueError',
        'capture original raised exception and exact-message assert only for calibration_missing'],
        keyerror_missing_reference_case_unchanged=True, zero_optimizer_write_assertions_unchanged=True,
        numeric_receipt_absence_assertion_unchanged=True, original_parameterized_cases_unchanged=True)


def main():
    parent_source = json.loads(PARENT_SNAPSHOT.read_text())
    parent_runtime = json.loads(PARENT_MANIFEST.read_text())
    old_rows, rows, runtime_rows = inventory(PARENT, omit_git=True), inventory(SOURCE, omit_git=True), inventory(RUNTIME)
    require(old_rows == parent_source['full_tracked_source_files'], 'Parent source11 frozen bytes changed')
    require(runtime_rows == parent_runtime['files'], 'Original runtime12 code bytes changed')
    old_by_name = {row['relative_path']: row for row in old_rows}
    changed = [row for row in rows if old_by_name.get(row['relative_path']) != row]
    unchanged = [row for row in rows if old_by_name.get(row['relative_path']) == row]
    require(len(rows) == len(old_rows) == 494 and [row['relative_path'] for row in changed] == [TARGET], 'Source-only fixture scope changed')
    require(len(unchanged) == 493, 'Other full source file bytes changed')
    tracked = [name for name in git(SOURCE, 'ls-files', '-z').split('\0') if name]
    require(set(tracked) == {row['relative_path'] for row in rows} and not git(SOURCE, 'status', '--porcelain'), 'Source clean/full tracked identity differs')
    commit, tree = git(SOURCE, 'rev-parse', 'HEAD'), git(SOURCE, 'rev-parse', 'HEAD^{tree}')
    require(git(SOURCE, 'rev-parse', 'HEAD^') == parent_source['new_code_commit'], 'Wrong source11 parent')
    require(git(SOURCE, 'rev-parse', '--git-common-dir') == '.git'
        and (SOURCE / '.git').resolve() != (PARENT / '.git').resolve(), 'Source is not independent clone')
    source_evidence = raw_evidence(SOURCE_RAW, 'FAILED_CPU_OR_COVERAGE',
        dict(collected=329, passed=328, failed=1, errors=0, skipped=0))
    source_result = json.loads((SOURCE_RAW / 'RESULT.json').read_text())
    require(source_result['failure_ids'] == [
        'tests.baselines.test_policy_safety_actor::test_entry_errors_reject_before_any_optimizer_write[calibration_missing]'], 'Original sole CPU failure differs')
    failed = [case for case in ET.parse(SOURCE_RAW / 'junit.xml').findall('.//testcase') if case.find('failure') is not None]
    require(failed[0].find('failure').attrib['message'] == 'ValueError: ' + MESSAGE, 'Actual original refusal differs')
    source_evidence['parent_cpu_failed_not_accepted'] = True
    runtime_evidence = raw_evidence(RUNTIME_RAW, 'PASS_CPU_ONLY',
        dict(collected=204, passed=204, failed=0, errors=0, skipped=0))
    runtime_evidence['parent_cpu_only_pass_not_successor_acceptance'] = True
    checks = []
    for row in rows:
        if row['relative_path'].endswith('.py'):
            actual_tree = ast.parse((SOURCE / row['relative_path']).read_bytes())
            if row['relative_path'] != TARGET:
                before_tree = ast.parse((PARENT / row['relative_path']).read_bytes())
                before, after = ast.dump(before_tree, include_attributes=False), ast.dump(actual_tree, include_attributes=False)
                require(before == after, 'Unchanged production/other Python AST differs')
                checks.append(dict(relative_path=row['relative_path'], whole_module_ast_equal=True,
                    parent_ast_dump_sha256=hashlib.sha256(before.encode()).hexdigest(),
                    new_ast_dump_sha256=hashlib.sha256(after.encode()).hexdigest()))
    for row in runtime_rows:
        if row['relative_path'].endswith('.py'):
            ast.parse((RUNTIME / row['relative_path']).read_bytes())
    fixture_scope = fixture_expected_ast_only()
    source_patch = SNAPSHOT.with_name('FRESH200-SOURCE-v11-to-v11-cpufixturefix1.patch')
    with source_patch.open('xb') as stream:
        stream.write(subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--binary', 'HEAD^', 'HEAD']))
    helper = ROOT / 'work/continuous_cpu_once_v4.py'
    require(record(helper) == parent_source['cpu_management_helper_prospective_record']
        == parent_runtime['cpu_management_helper_prospective_record'], 'Helper4 changed')
    source = copy.deepcopy(parent_source)
    source.update(source_repo=str(SOURCE), code_repo=str(SOURCE), branch=git(SOURCE, 'branch', '--show-current'),
        new_code_commit=commit, new_tree=tree, parent_code_commit=parent_source['new_code_commit'],
        canonical_future_source_repo=parent_source['canonical_future_source_repo'].rsplit('repo-v11-', 1)[0] + 'repo-v11-' + commit[:7],
        source_snapshot_parent_record=record(PARENT_SNAPSHOT), source_patch_record=record(source_patch),
        full_tracked_source_files=rows, full_tracked_source_file_count=len(rows), source_tracked_files_digest=digest(rows),
        changed_or_new_source_files=changed, unchanged_tracked_source_file_count=len(unchanged),
        unchanged_tracked_source_files=unchanged, unchanged_source_files_digest=digest(unchanged),
        critical_unchanged_file_records=[row for row in unchanged if row['relative_path'] in
            {item['relative_path'] for item in parent_source['critical_unchanged_file_records']}],
        all_unchanged_python_module_ast_comparison=checks, whole_module_ast_unchanged_python_file_count=len(checks),
        production_and_all_other_source_files_byte_identical=True, fixture_only_ast_scope=fixture_scope,
        narrow_repair_scope=['obsolete CPU missing-artifact exception expectation only; production code unchanged'],
        parent_actual_source_cpu_evidence=source_evidence, parent_actual_runtime_cpu_evidence=runtime_evidence,
        approved_cpu_policy=None, independent_preexec_review=None, actual_cpu_result=None,
        actual_collected_cpu_case_count=None, tests_executed=False, authoring_stopped=True,
        author_static_recorder=record(Path(__file__).resolve()),
        implementation_authors=parent_source['implementation_authors'],
        current_fixture_only_author='/root/continuous12_compat_author',
        successor_cpu_execution_performed=False,
        clone_isolation=dict(method='git clone --no-hardlinks; separate .git, no worktree',
            parent_git_directory=str((PARENT / '.git').resolve()), successor_git_directory=str((SOURCE / '.git').resolve()),
            shared_git_directory=False))
    require(source['source_version'] == 11 and source['cpu_required_test_files'] == source_result['test_files'], 'Source API or full declared selection changed')
    write_new(SNAPSHOT, source)
    runtime = copy.deepcopy(parent_runtime)
    runtime.update(runtime_parent_record=record(PARENT_MANIFEST), runtime_patch_record=None,
        parent_runtime_patch_record=parent_runtime['runtime_patch_record'],
        source_snapshot_record=record(SNAPSHOT), source_commit=commit,
        canonical_future_source_repo=source['canonical_future_source_repo'],
        source_file_count=len(rows), source_tracked_files_digest=digest(rows),
        runtime_changed_or_new_files=[], runtime_unchanged_files=runtime_rows,
        runtime_unchanged_file_count=len(runtime_rows), runtime_code_full_bytes_unchanged=True,
        metadata_only_source_rebinding=True, runtime_parent_full_closure_preserved=True,
        parent_actual_source_cpu_evidence=source_evidence, parent_actual_runtime_cpu_evidence=runtime_evidence,
        narrow_repair_scope=['metadata-only new source11 fixture commit binding; all runtime12 code bytes unchanged'],
        new_fixture_execution='No successor CPU execution. Parent runtime fixture results bind only the original 4a45458 closure; raw originals retained.',
        approved_cpu_policy=None, independent_preexec_review=None, actual_cpu_result=None,
        actual_collected_cpu_case_count=None, tests_executed=False, authoring_stopped=True,
        author_static_recorder=record(Path(__file__).resolve()), successor_cpu_execution_performed=False,
        fixture_actual_production_body_records=[record(SOURCE / relative) for relative in (
            'search_r1/baselines/fresh200_live.py', 'search_r1/baselines/fresh200.py',
            'search_r1/baselines/recovery.py', 'verl/workers/fsdp_workers.py', 'verl/single_controller/ray/base.py')] +
            [record(RUNTIME / 'metadata_colocation_v1.py')])
    require(runtime['runtime_version'] == 12 and runtime['files'] == parent_runtime['files']
        and runtime['runtime_files_digest'] == parent_runtime['runtime_files_digest'], 'Runtime API/full input bytes changed')
    write_new(MANIFEST, runtime)
    # Read original raw files again to establish this author recorder did not mutate them.
    for evidence in (source_evidence, runtime_evidence):
        require([record(Path(row['path'])) for row in evidence['original_file_records']]
            == evidence['original_file_records'], 'Original CPU evidence changed during author recording')
    print(json.dumps(dict(source_snapshot=record(SNAPSHOT), runtime_manifest=record(MANIFEST),
        source_commit=commit, source_file_count=len(rows), other_source_files_byte_unchanged=len(unchanged),
        runtime_file_count=len(runtime_rows), runtime_code_bytes_unchanged=True,
        current_cpu_cases=None, successor_execution=False, authoring_stopped=True), indent=2))


if __name__ == '__main__':
    main()
