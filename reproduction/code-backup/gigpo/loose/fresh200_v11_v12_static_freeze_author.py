"""Author-only static/hash/AST/local Git metadata recorder; no project imports."""
import ast
import copy
import difflib
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_PARENT = ROOT / 'work/gigpo-e1-fresh200-typed-safety-20261006'
SOURCE = ROOT / 'work/gigpo-e1-fresh200-compat11-20261006'
RUNTIME_PARENT = ROOT / 'work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v11'
RUNTIME = ROOT / 'work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v12'
SOURCE_PARENT_SNAPSHOT = ROOT / 'outputs/fresh200-compatibility-20261006/FRESH200-SOURCE-STATIC-SNAPSHOT-v10.json'
RUNTIME_PARENT_MANIFEST = ROOT / 'outputs/gigpo-e1-policy-safety/FRESH200-CONTINUATION-STATIC-MANIFEST-v11.json'
SOURCE_SNAPSHOT = ROOT / 'outputs/fresh200-compatibility-20261006/FRESH200-SOURCE-STATIC-SNAPSHOT-v11.json'
RUNTIME_MANIFEST = ROOT / 'outputs/gigpo-e1-policy-safety/FRESH200-CONTINUATION-STATIC-MANIFEST-v12.json'
HELPER = ROOT / 'work/continuous_cpu_once_v4.py'
LATEST = 1791299045
FORMAL_ROOT = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def record(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def actual_inventory(directory, omit_git=False):
    values = []
    for parent, dirs, files in os.walk(directory, followlinks=False):
        if omit_git and Path(parent) == directory:
            dirs[:] = [name for name in dirs if name != '.git']
        for name in dirs:
            require(not (Path(parent) / name).is_symlink(), 'Directory alias in full inventory')
        for name in files:
            path = Path(parent) / name
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Nonregular/aliased full input')
            row = record(path)
            values.append(dict(relative_path=str(path.relative_to(directory)), bytes=row['bytes'], sha256=row['sha256']))
    return sorted(values, key=lambda row: row['relative_path'])


def parse_all(directory, rows):
    names = []
    for row in rows:
        if row['relative_path'].endswith('.py'):
            path = directory / row['relative_path']
            ast.parse(path.read_bytes(), filename=str(path))
            names.append(row['relative_path'])
    return names


def function(path, name):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, 'Ambiguous/missing original function: ' + name)
    return ast.dump(matches[0], include_attributes=False)


def comparison(relative, name):
    old, new = function(SOURCE_PARENT / relative, name), function(SOURCE / relative, name)
    require(old == new, 'Training/math AST changed: ' + name)
    return dict(relative_path=relative, function=name, ast_equal=True,
        parent_ast_dump_sha256=hashlib.sha256(old.encode()).hexdigest(),
        new_ast_dump_sha256=hashlib.sha256(new.encode()).hexdigest())


def normalizer_records():
    path = SOURCE / 'verl/workers/fsdp_workers.py'
    tree = ast.parse(path.read_bytes(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ActorRolloutRefWorker')
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    nodes = [node for node in constructor.body if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute) and isinstance(node.test.value, ast.Name)
        and node.test.value.id == 'self' and node.test.attr in {'_is_actor', '_is_rollout', '_is_ref'}
        and any(isinstance(child, ast.AugAssign) and isinstance(child.op, ast.FloorDiv) for child in ast.walk(node))]
    require(len(nodes) == 3, 'Original three normalization clauses changed')
    return [dict(role_flag=node.test.attr, start=node.lineno, end=node.end_lineno,
        ast_dump_sha256=hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()) for node in nodes]


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parent_source = json.loads(SOURCE_PARENT_SNAPSHOT.read_text())
    parent_runtime = json.loads(RUNTIME_PARENT_MANIFEST.read_text())
    old_source_rows = actual_inventory(SOURCE_PARENT, omit_git=True)
    old_runtime_rows = actual_inventory(RUNTIME_PARENT)
    require(old_source_rows == parent_source['full_tracked_source_files'], 'Frozen source10 changed')
    require(old_runtime_rows == parent_runtime['files'], 'Frozen runtime11 changed')
    source_rows, runtime_rows = actual_inventory(SOURCE, omit_git=True), actual_inventory(RUNTIME)
    tracked = git(SOURCE, 'ls-files', '-z').split('\0')
    tracked = [name for name in tracked if name]
    require(set(tracked) == {row['relative_path'] for row in source_rows}, 'Source full actual/tracked inventory differs')
    require(git(SOURCE, 'status', '--porcelain') == '', 'Source11 is not clean')
    commit, tree = git(SOURCE, 'rev-parse', 'HEAD'), git(SOURCE, 'rev-parse', 'HEAD^{tree}')
    require(git(SOURCE, 'rev-parse', 'HEAD^') == parent_source['new_code_commit'], 'Wrong parent source10 commit')
    require((SOURCE / '.git').is_dir() and git(SOURCE, 'rev-parse', '--git-common-dir') == '.git', 'Source11 must use independent clone')
    require((SOURCE / '.git').resolve() != (SOURCE_PARENT / '.git').resolve(), 'Source11 shares parent Git directory')
    old_source = {row['relative_path']: row for row in old_source_rows}
    old_runtime = {row['relative_path']: row for row in old_runtime_rows}
    source_changed = [row for row in source_rows if old_source.get(row['relative_path']) != row]
    runtime_changed = [row for row in runtime_rows if old_runtime.get(row['relative_path']) != row]
    source_unchanged = [row for row in source_rows if old_source.get(row['relative_path']) == row]
    runtime_unchanged = [row for row in runtime_rows if old_runtime.get(row['relative_path']) == row]
    require({row['relative_path'] for row in source_changed} == {
        'search_r1/baselines/fresh200_live.py', 'tests/baselines/test_fresh200_live_goal_binding_v11.py'}, 'Source repair scope expanded')
    require({row['relative_path'] for row in runtime_changed} == {
        'metadata_colocation_v1.py', 'calibration_contract_v1.py', 'test_calibration_contract_v1.py',
        'test_live_continuation_v10.py', 'test_typed_refusal_producer_v11.py',
        'test_colocated_live_admission_v12.py'}, 'Runtime repair scope expanded')
    old_live = (SOURCE_PARENT / 'search_r1/baselines/fresh200_live.py').read_bytes()
    require(old_live.replace(b'LIVE_GOAL_UPDATED_AT = 1791295951', b'LIVE_GOAL_UPDATED_AT = 1791299045', 1)
        == (SOURCE / 'search_r1/baselines/fresh200_live.py').read_bytes(), 'Source production change is not one constant')
    require(record(HELPER)['bytes'] == 19553 and record(HELPER)['sha256'] ==
        '520819b9e7954ecb66b45c17a673af40497b0e4a8f6ee045efb78352f84ad46f', 'Helper4 exact core differs')
    require((RUNTIME / 'calibration_v1.sbatch').read_bytes() == (RUNTIME_PARENT / 'calibration_v1.sbatch').read_bytes(), 'Sbatch protection changed')
    source_ast = parse_all(SOURCE, source_rows)
    runtime_ast = parse_all(RUNTIME, runtime_rows)
    math_checks = [comparison(row['relative_path'], row['function'])
        for row in parent_source['seven_math_functions_ast_comparison']]
    prefix_checks = [comparison('search_r1/baselines/action_prefix_forward.py', row['function'])
        for row in parent_source['valid_action_prefix_functions_ast_comparison']]
    live_checks = [comparison('search_r1/baselines/fresh200_live.py', node.name)
        for node in ast.parse(old_live).body if isinstance(node, ast.FunctionDef)]
    source_patch = SOURCE_SNAPSHOT.with_name('FRESH200-SOURCE-v10-to-v11.patch')
    source_raw_patch = subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--binary', 'HEAD^', 'HEAD'])
    with source_patch.open('xb') as stream:
        stream.write(source_raw_patch)
    runtime_patch = RUNTIME_MANIFEST.with_name('FRESH200-RUNTIME-v11-to-v12.patch')
    runtime_raw_patch = ''.join(''.join(difflib.unified_diff(
        (RUNTIME_PARENT / row['relative_path']).read_text().splitlines(keepends=True)
            if (RUNTIME_PARENT / row['relative_path']).exists() else [],
        (RUNTIME / row['relative_path']).read_text().splitlines(keepends=True),
        fromfile='runtime11/' + row['relative_path'], tofile='runtime12/' + row['relative_path']))
        for row in runtime_changed)
    with runtime_patch.open('x') as stream:
        stream.write(runtime_raw_patch)
    fail = ROOT / 'outputs/gigpo-e1-instruct200/CONTINUOUS-PREEXEC-SOURCE10-RUNTIME11-REVIEW-v1.json'
    require(record(fail)['bytes'] == 355585 and record(fail)['sha256'] ==
        '01a18015bbead609d71ded10d50be0cc019ddb44afc3f9fea13bc89dc54ce8e9', 'Original complete FAIL changed')
    goal = dict(parent_source['goal_provenance'], live_goal_updated_at=LATEST,
        superseded_source10_live_goal_updated_at=parent_source['goal_provenance']['live_goal_updated_at'])
    source_tests = parent_source['cpu_required_test_files'] + ['tests/baselines/test_fresh200_live_goal_binding_v11.py']
    runtime_tests = parent_runtime['cpu_required_test_files'] + ['test_colocated_live_admission_v12.py']
    common = dict(status='FROZEN_STATIC_IMPLEMENTATION_ONLY_NOT_ACCEPTED', authoring_stopped=True,
        approved_cpu_policy=None, independent_preexec_review=None, actual_collected_cpu_case_count=None,
        actual_cpu_result=None, actual_gpu_acceptance=False, production_READY=False, submitted=False,
        formal_iterations=0, actual_formal_adam_writes=0, tests_executed=False,
        project_modules_imported=False, remote_operations_performed=False, goal_provenance=goal,
        implementation_authors=['/root', '/root/continuous12_compat_author'],
        cpu_management_helper_prospective_record=record(HELPER),
        canonical_future_cpu_management_helper=FORMAL_ROOT + '/tools/management/continuous-cpu-once-v4.py',
        original_failed_preexec_review_record=record(fail),
        target_brief_record=record(ROOT / 'outputs/gigpo-e1-instruct200/PREEXEC-REVIEW-TARGET-v2.md'),
        live_scope=parent_source['live_scope'], production_review_scope=parent_source['production_review_scope'],
        digest_serialization='SHA256(JSON sorted relative_path/bytes/sha256 list, sort_keys=True,separators=(comma,colon))',
        math_comparison_serialization=parent_source['math_comparison_serialization'],
        author_static_recorder=record(Path(__file__).resolve()))
    source_snapshot = dict(common, schema='gigpo-e1-fresh200-source-v11-static-snapshot', source_version=11,
        source_repo=str(SOURCE), code_repo=str(SOURCE), branch=git(SOURCE, 'branch', '--show-current'),
        new_code_commit=commit, new_tree=tree, parent_code_commit=parent_source['new_code_commit'],
        source_snapshot_parent_record=record(SOURCE_PARENT_SNAPSHOT), source_patch_record=record(source_patch),
        canonical_future_source_repo=FORMAL_ROOT + '/repo-v11-' + commit[:7],
        full_tracked_source_file_count=len(source_rows), full_tracked_source_files=source_rows,
        source_tracked_files_digest=digest(source_rows), changed_or_new_source_files=source_changed,
        unchanged_tracked_source_file_count=len(source_unchanged), unchanged_tracked_source_files=source_unchanged,
        unchanged_source_files_digest=digest(source_unchanged), source_parent_full_closure_preserved=set(old_source).issubset(tracked),
        critical_unchanged_file_records=[row for row in source_unchanged if row['relative_path'] in
            {item['relative_path'] for item in parent_source['critical_unchanged_file_records']}],
        seven_math_functions_ast_comparison=math_checks, valid_action_prefix_functions_ast_comparison=prefix_checks,
        all_live_functions_ast_comparison=live_checks, ast_parsed_python_file_count=len(source_ast),
        original_worker_micro_normalization_ast=normalizer_records(),
        candidate_cpu_test_files=source_tests, cpu_required_test_files=source_tests,
        clone_isolation=dict(method='git clone --no-hardlinks; separate .git, no worktree',
            parent_git_directory=str((SOURCE_PARENT / '.git').resolve()),
            successor_git_directory=str((SOURCE / '.git').resolve()), shared_git_directory=False),
        narrow_repair_scope=['latest live human goal clock only in production source',
            'unexecuted real validator fixture accepts latest and refuses superseded goal binding'],
        optional_checkpoint_seams=parent_source['optional_checkpoint_seams'])
    write_new(SOURCE_SNAPSHOT, source_snapshot)
    runtime_manifest = dict(common, schema=parent_runtime['schema'], runtime_version=12, runtime_root=str(RUNTIME),
        runtime_parent_record=record(RUNTIME_PARENT_MANIFEST), runtime_patch_record=record(runtime_patch),
        canonical_future_runtime_root=FORMAL_ROOT + '/tools/fresh200-continuation-v12',
        canonical_future_source_repo=source_snapshot['canonical_future_source_repo'],
        source_snapshot_record=record(SOURCE_SNAPSHOT), source_commit=commit,
        source_file_count=len(source_rows), source_tracked_files_digest=digest(source_rows),
        file_count=len(runtime_rows), files=runtime_rows, runtime_files_digest=digest(runtime_rows),
        python_file_count=len(runtime_ast), runtime_changed_or_new_files=runtime_changed,
        runtime_unchanged_file_count=len(runtime_unchanged), runtime_unchanged_files=runtime_unchanged,
        runtime_parent_full_closure_preserved=set(old_runtime).issubset({row['relative_path'] for row in runtime_rows}),
        full_source_and_runtime_input_file_count=len(source_rows) + len(runtime_rows),
        protected_sbatch_relative_path='calibration_v1.sbatch', protected_sbatch_record=record(RUNTIME / 'calibration_v1.sbatch'),
        candidate_cpu_test_files=runtime_tests, cpu_required_test_files=runtime_tests,
        driver_entrypoint=parent_runtime['driver_entrypoint'], owned_run_entrypoint=parent_runtime['owned_run_entrypoint'],
        continuation=parent_runtime['continuation'], model_initialization=parent_runtime['model_initialization'],
        narrow_repair_scope=['actual constructor config-only role-private deepcopy',
            'latest live goal contract/fixture clock and source11 fixture fallbacks',
            'unexecuted shared OmegaConf real factory -> original role normalization -> actual source installer/restorer and metadata roundtrip'],
        new_fixture_execution='UNEXECUTED; no collected cases or test results; CPU boundaries are not actual Ray/FSDP/GPU evidence',
        fixture_actual_production_body_records=[record(SOURCE / relative) for relative in (
            'search_r1/baselines/fresh200_live.py', 'search_r1/baselines/fresh200.py',
            'search_r1/baselines/recovery.py', 'verl/workers/fsdp_workers.py', 'verl/single_controller/ray/base.py')] +
            [record(RUNTIME / 'metadata_colocation_v1.py')],
        original_worker_micro_normalization_ast=normalizer_records())
    write_new(RUNTIME_MANIFEST, runtime_manifest)
    print(json.dumps(dict(source_snapshot=record(SOURCE_SNAPSHOT), runtime_manifest=record(RUNTIME_MANIFEST),
        source_commit=commit, source_file_count=len(source_rows), runtime_file_count=len(runtime_rows),
        source_test_file_count=len(source_tests), runtime_test_file_count=len(runtime_tests),
        actual_case_count=None, tests_executed=False, authoring_stopped=True), indent=2))


if __name__ == '__main__':
    main()
