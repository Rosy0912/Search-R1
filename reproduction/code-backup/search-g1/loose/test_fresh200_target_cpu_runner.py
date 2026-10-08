"""Constructed stdlib launcher tests; no actual Git, SSH, target, CUDA or models.

These fixtures are not physical target evidence. All subprocess/controller
operations are mocked; source reads/writes remain in temporary fixture folders.
"""
import ast
import base64
import contextlib
import copy
import importlib.util
import io
import json
import multiprocessing
import os
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('fresh200_target_cpu_launcher',
                                             HERE / 'test_fresh200_target_cpu.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
COMMIT = 'b' * 40
BASE = '/dataset/shaoshitong/g1-fresh200-cpu-' + 'a' * 32


class SourceFixture:
    """In-memory committed Git responses plus ordinary temporary source files."""
    def __init__(self, repo):
        self.repo = repo
        self.repo.mkdir()
        names = set(m.VERSION_FILES) | set(m.SHELL_FILES) | {'module.py'}
        names.update(module.replace('.', '/') + '.py' for module in m.MODULES)
        self.committed = {name: ('# fixture source: ' + name + '\n').encode()
                          for name in sorted(names)}
        self.committed.update({'evidence/ignored.json': b'{}', 'cache/ignored.py': b'',
                               '.git/ignored.py': b'', '.pytest_cache/ignored.json': b'{}'})
        self.modes = {}
        self.head = COMMIT
        self.calls = []
        for name, data in self.committed.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def git(self, repo, *args):
        if Path(repo) != self.repo:
            raise AssertionError('Fixture cannot inspect a real repository')
        self.calls.append(args)
        if args == ('rev-parse', 'HEAD'):
            return (self.head + '\n').encode()
        if args == ('ls-tree', '-r', '-z', COMMIT):
            return b''.join((self.modes.get(name, '100644') + ' blob ' + 'c' * 40 + '\t' + name).encode() + b'\0'
                            for name in sorted(self.committed))
        if len(args) == 2 and args[0] == 'show' and args[1].startswith(COMMIT + ':'):
            return self.committed[args[1][41:]]
        if len(args) == 3 and args[:2] == ('cat-file', '-s') and args[2].startswith(COMMIT + ':'):
            return (str(len(self.committed[args[2][41:]])) + '\n').encode()
        raise AssertionError('Unexpected fixture Git operation: ' + repr(args))

    def build(self, commit=COMMIT, base=BASE):
        with patch.object(m, 'git', side_effect=self.git), \
                patch.object(m.subprocess, 'run', side_effect=AssertionError('No actual subprocess in fixture')):
            return m.build_payload(self.repo, commit, base)


def payload_fixture():
    with tempfile.TemporaryDirectory() as folder:
        return SourceFixture(Path(folder) / 'repo').build()


def execution_fixture():
    return dict(returncode=0, stdout='', stderr='', observation_timed_out=False,
                owned_process_group_cleaned=False, owned_process_group_remaining=False)


def summary_fixture():
    return dict(tests_run=len(m.MODULES), skipped=[], failures=0, errors=0, successful=True,
        expected_failures=0, unexpected_successes=0, expected_failure_details=[], unexpected_success_details=[],
        failure_details=[], error_details=[], module_test_counts={name: 1 for name in m.MODULES},
        cuda_available=False, cuda_initialized_before=False, cuda_initialized_after=False,
        cuda_visible_devices='', distribution_versions=dict(torch='2.4.0+cu121',
            transformers='4.46.3', vllm='0.6.3', tensordict='0.5.0'),
        torch_version='2.4.0+cu121', transformers_module_version='4.46.3', vllm_module_version='dev',
        expected_distribution_versions_matched=True, python_version='3.10.16',
        python_executable=m.TARGET_PYTHON, system='Linux', modules=list(m.MODULES),
        runner_error=None, duration_seconds=1.5, output='constructed fixture only; not target proof')


def proof_fixture(payload, summary=None):
    record = {key: payload[key] for key in ('schema_version', 'role', 'base', 'source_commit',
        'source_map_sha256', 'source_sha256', 'expected_source_count', 'modules', 'module_count',
        'scope', 'target_python', 'cpu_runner_sha256', 'runner_timeout_seconds')}
    record.update(repo_path=payload['base'] + '/repo', runner_path=payload['base'] + '/runner/cpu_runner.py',
        source_initial_sha256=payload['source_sha256'], source_before_tests_sha256=payload['source_sha256'],
        source_after_tests_sha256=payload['source_sha256'], source_unchanged=True, source_inventory_error=None,
        actual_source_count=payload['expected_source_count'], cpu_runner_before_sha256=payload['cpu_runner_sha256'],
        cpu_runner_after_sha256=payload['cpu_runner_sha256'], cpu_runner_unchanged=True, runner_read_error=None,
        cuda_visible_devices='', actual_production_updates=0, production_admission=False,
        healthy_checkpoint_accepted=False, started_unix=100., checked_unix=102., duration_seconds=2.,
        shell_syntax_checks=[dict(execution_fixture(), path=name) for name in m.SHELL_FILES],
        **execution_fixture())
    record['stdout'] = m.SUMMARY_PREFIX + json.dumps(summary or summary_fixture()) + '\n'
    return copy.deepcopy(record)


class CommittedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SourceFixture(Path(self.temporary.name) / 'repo')

    def tearDown(self):
        self.temporary.cleanup()

    def test_dynamic_complete_count_and_canonical_map(self):
        payload = self.fixture.build()
        self.assertEqual(payload['expected_source_count'], len(payload['files']))
        self.assertEqual(payload['source_sha256'], m.runtime_map(self.fixture.repo))
        self.assertEqual(payload['source_map_sha256'], m.sha(m.canonical(payload['source_sha256'])))
        self.assertEqual(set(payload['files']), {name for name in self.fixture.committed if m.runtime_name(name)})
        self.assertNotIn('EXPECTED_FILES', vars(m))
        self.assertEqual(payload['source_commit'], COMMIT)
        self.assertEqual(sum(call == ('rev-parse', 'HEAD') for call in self.fixture.calls), 2)

    def test_added_committed_empty_runtime_file_changes_frozen_count(self):
        before = self.fixture.build()
        self.fixture.committed['empty.py'] = b''
        (self.fixture.repo / 'empty.py').write_bytes(b'')
        after = self.fixture.build()
        self.assertEqual(after['expected_source_count'], before['expected_source_count'] + 1)
        self.assertEqual(after['source_sha256']['empty.py'], m.sha(b''))

    def test_short_nonhex_and_wrong_head_rejected_before_tree_read(self):
        for commit in (COMMIT[:7], 'z' * 40, 'f' * 40, True):
            with self.subTest(commit=commit), self.assertRaises(ValueError):
                self.fixture.build(commit=commit)

    def test_uncommitted_bytes_untracked_and_deleted_runtime_files_rejected(self):
        path = self.fixture.repo / 'module.py'
        original = path.read_bytes()
        path.write_bytes(b'!' * len(original))
        with self.assertRaisesRegex(ValueError, 'changed after commit'):
            self.fixture.build()
        path.write_bytes(original)
        extra = self.fixture.repo / 'untracked.py'
        extra.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'file set differs'):
            self.fixture.build()
        extra.unlink()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'file set differs'):
            self.fixture.build()

    def test_committed_and_current_source_links_rejected(self):
        self.fixture.modes['module.py'] = '120000'
        with self.assertRaisesRegex(ValueError, 'symlink or non-file'):
            self.fixture.build()
        self.fixture.modes.clear()
        path = self.fixture.repo / 'module.py'
        path.unlink()
        path.symlink_to(m.VERSION_FILES[0])
        with self.assertRaisesRegex(ValueError, 'ordinary regular file'):
            self.fixture.build()

    def test_runtime_directory_symlink_rejected(self):
        (self.fixture.repo / 'linked').symlink_to(self.fixture.repo / 'reproduction', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'directory symlink'):
            self.fixture.build()

    def test_local_runtime_walk_propagates_unreadable_root_and_child_errors(self):
        real_scandir = os.scandir
        for unreadable in (self.fixture.repo, self.fixture.repo / 'reproduction'):
            def scan(path):
                if Path(path) == unreadable:
                    raise PermissionError('constructed unreadable runtime directory')
                return real_scandir(path)
            with self.subTest(path=unreadable), patch.object(os, 'scandir', side_effect=scan), \
                    self.assertRaisesRegex(PermissionError, 'unreadable runtime directory'):
                m.runtime_map(self.fixture.repo)

    def test_source_inventory_detects_addition_deletion_and_edit(self):
        original = m.runtime_map(self.fixture.repo)
        added = self.fixture.repo / 'extra.py'
        added.write_bytes(b'')
        self.assertNotEqual(original, m.runtime_map(self.fixture.repo))
        added.unlink()
        path = self.fixture.repo / 'module.py'
        path.write_bytes(b'edited')
        self.assertNotEqual(original, m.runtime_map(self.fixture.repo))
        path.unlink()
        self.assertNotEqual(original, m.runtime_map(self.fixture.repo))

    def test_namespace_and_small_source_limit(self):
        for base in ('/tmp/not-owned', BASE + '/repo', BASE[:-1], BASE.replace('a', 'g')):
            with self.subTest(base=base), self.assertRaises(ValueError):
                self.fixture.build(base=base)
        with patch.object(m, 'MAX_FILE_BYTES', 1), self.assertRaisesRegex(ValueError, 'size differs from bound'):
            self.fixture.build()

    def test_runtime_map_rejects_fifo_symlink_replacement_without_reading(self):
        path = self.fixture.repo / 'module.py'
        real_open = os.open
        observed = []
        def substitute(filename, flags):
            if Path(filename) == path:
                path.unlink()
                os.mkfifo(path)
                observed.append(flags)
            return real_open(filename, flags)
        with patch.object(os, 'open', side_effect=substitute), \
                self.assertRaisesRegex(ValueError, 'opened a nonordinary file'):
            m.runtime_map(self.fixture.repo)
        self.assertTrue(observed[0] & os.O_NONBLOCK)
        self.assertTrue(observed[0] & os.O_NOFOLLOW)

    def test_sparse_huge_local_runtime_file_rejected_before_stream_read(self):
        path = self.fixture.repo / 'module.py'
        with path.open('wb') as stream:
            stream.truncate(m.MAX_FILE_BYTES + 1)
        with patch.object(os, 'read') as read, \
                self.assertRaisesRegex(ValueError, 'size differs from bound'):
            m.runtime_map(self.fixture.repo)
        read.assert_not_called()

    def test_committed_blob_size_is_bounded_before_git_show(self):
        def oversized(repo, *args):
            if args == ('cat-file', '-s', COMMIT + ':module.py'):
                self.fixture.calls.append(args)
                return (str(m.MAX_FILE_BYTES + 1) + '\n').encode()
            return self.fixture.git(repo, *args)
        with patch.object(m, 'git', side_effect=oversized), \
                patch.object(m.subprocess, 'run', side_effect=AssertionError('No actual subprocess')):
            with self.assertRaisesRegex(ValueError, 'Committed runtime source file is too large'):
                m.build_payload(self.fixture.repo, COMMIT, BASE)
        self.assertNotIn(('show', COMMIT + ':module.py'), self.fixture.calls)

    def test_git_reader_is_bounded_and_read_only(self):
        result = subprocess.CompletedProcess([], 0, stdout=b'read only', stderr=b'')
        with patch.object(m.subprocess, 'run', return_value=result) as run:
            self.assertEqual(m.git(self.fixture.repo, 'rev-parse', 'HEAD'), b'read only')
        run.assert_called_once_with(['git', '-C', str(self.fixture.repo), 'rev-parse', 'HEAD'],
                                    capture_output=True, check=True, timeout=30)


class PayloadAndRunnerTests(unittest.TestCase):
    def setUp(self):
        self.payload = payload_fixture()

    def test_exact_union_contains_real_gloo_and_all_original_modules(self):
        self.assertEqual(len(m.FRESH200_MODULES), 11)
        self.assertEqual(len(m.BEHAVIOR_MODULES), 16)
        self.assertEqual(len(m.MODULES), 25)
        self.assertEqual(len(m.MODULES), len(set(m.MODULES)))
        self.assertEqual(set(m.MODULES), set(m.FRESH200_MODULES) | set(m.BEHAVIOR_MODULES))
        self.assertIn('reproduction.common8.test_scheduler_collective', m.MODULES)
        self.assertIn('reproduction.common8.test_sampling_capture', m.MODULES)
        self.assertEqual(self.payload['module_count'], len(m.MODULES))

    def test_payload_schema_source_digest_and_runner_binding(self):
        self.assertIs(m.validate_payload(self.payload), self.payload)
        mutations = [('schema_version', True), ('role', 'healthy'), ('scope', 'training'),
            ('source_commit', 'b' * 7), ('base', '/tmp/x'), ('expected_source_count', 1),
            ('source_map_sha256', 'f' * 64), ('modules', list(m.MODULES[:-1])), ('module_count', True),
            ('target_python', '/other/python'), ('cpu_runner', 'print(1)'),
            ('cpu_runner_sha256', 'f' * 64), ('runner_timeout_seconds', 901)]
        for key, value in mutations:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.payload)
                bad[key] = value
                with self.assertRaises(ValueError):
                    m.validate_payload(bad)
        for action in ('add', 'remove'):
            bad = copy.deepcopy(self.payload)
            if action == 'add':
                bad['unexpected'] = 1
            else:
                del bad['files']
            with self.assertRaisesRegex(ValueError, 'schema'):
                m.validate_payload(bad)

    def test_payload_rejects_wrong_bytes_noncanonical_base64_missing_module_and_unsafe_paths(self):
        bad = copy.deepcopy(self.payload)
        bad['files']['module.py'] = base64.b64encode(b'different').decode()
        with self.assertRaisesRegex(ValueError, 'bytes/hash differ'):
            m.validate_payload(bad)
        bad = copy.deepcopy(self.payload)
        bad['files']['module.py'] += '\n'
        with self.assertRaises(ValueError):
            m.validate_payload(bad)
        for name in ('../escape.py', '/absolute.py', 'a\\b.py', './a.py', 'weights.safetensors'):
            self.assertFalse(m.runtime_name(name))
        bad = copy.deepcopy(self.payload)
        name = m.MODULES[0].replace('.', '/') + '.py'
        del bad['files'][name]
        del bad['source_sha256'][name]
        bad['expected_source_count'] -= 1
        bad['source_map_sha256'] = m.sha(m.canonical(bad['source_sha256']))
        with self.assertRaisesRegex(ValueError, 'module source absent'):
            m.validate_payload(bad)

    def test_file_backed_spawn_entry_and_skip_serialization_are_explicit(self):
        tree = ast.parse(m.CPU_RUNNER)
        guards = [node for node in tree.body if isinstance(node, ast.If) and
                  ast.unparse(node.test) == "__name__ == '__main__'"]
        self.assertEqual(len(guards), 1)
        self.assertIn('sys.exit(main())', ast.unparse(guards[0]))
        self.assertNotIn('import torch', '\n'.join(ast.unparse(node) for node in tree.body
                                                  if isinstance(node, (ast.Import, ast.ImportFrom))))
        self.assertIn('[test_name(case), str(reason)]', m.CPU_RUNNER)
        self.assertIn("[p['target_python'],str(runner),*p['modules']]", m.REMOTE_CODE)
        self.assertNotIn("[p['target_python'],'-c'", m.REMOTE_CODE)
        self.assertNotIn('354', m.REMOTE_CODE)
        self.assertIn("initial==before==after==p['source_sha256']", m.REMOTE_CODE)
        self.assertIn('start_new_session=True', m.REMOTE_CODE)
        self.assertIn('os.killpg(process.pid,sig)', m.REMOTE_CODE)
        self.assertIn("p['modules']==list(expected_modules)", m.REMOTE_CODE)
        self.assertIn("PYTHONNOUSERSITE='1'", m.REMOTE_CODE)
        self.assertIn("CUDA_VISIBLE_DEVICES=''", m.REMOTE_CODE)
        ast.parse(m.REMOTE_CODE)

    def test_strict_json_duplicate_nonfinite_and_invalid_rejected(self):
        for text in ('{"a":1,"a":2}', '{"n":NaN}', '{"n":Infinity}', 'not JSON'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                m.strict_json(text)


class ProofValidationTests(unittest.TestCase):
    def setUp(self):
        self.payload = payload_fixture()

    def test_complete_constructed_receipt_and_module_version_label(self):
        summary = m.validate_proof(proof_fixture(self.payload), self.payload)
        self.assertEqual(summary['vllm_module_version'], 'dev')
        self.assertEqual(summary['distribution_versions']['vllm'], '0.6.3')
        self.assertEqual(summary['tests_run'], sum(summary['module_test_counts'].values()))

    def test_every_identity_source_count_and_execution_slot_checked(self):
        mutations = [('schema_version', True), ('role', 'healthy'), ('base', '/tmp/x'),
            ('repo_path', BASE), ('runner_path', BASE + '/repo/runner.py'), ('source_commit', 'e' * 40),
            ('source_map_sha256', 'e' * 64), ('expected_source_count', self.payload['expected_source_count'] + 1), ('actual_source_count', True),
            ('modules', []), ('module_count', True), ('scope', 'healthy'), ('target_python', 'python'),
            ('cpu_runner_sha256', 'f' * 64), ('cpu_runner_before_sha256', 'f' * 64),
            ('cpu_runner_after_sha256', 'f' * 64), ('cpu_runner_unchanged', False),
            ('runner_read_error', 'ValueError: runner became FIFO'),
            ('runner_timeout_seconds', 901), ('source_unchanged', False), ('source_inventory_error', 'deleted'),
            ('actual_production_updates', True), ('production_admission', True),
            ('healthy_checkpoint_accepted', True), ('cuda_visible_devices', '0'),
            ('returncode', True), ('observation_timed_out', True), ('owned_process_group_remaining', True),
            ('owned_process_group_cleaned', None), ('started_unix', float('nan')),
            ('checked_unix', 99), ('duration_seconds', 1), ('stderr', None)]
        mutations.extend((key, {}) for key in ('source_sha256', 'source_initial_sha256',
                                              'source_before_tests_sha256', 'source_after_tests_sha256'))
        for key, value in mutations:
            with self.subTest(key=key):
                bad = proof_fixture(self.payload)
                bad[key] = value
                with self.assertRaises(ValueError):
                    m.validate_proof(bad, self.payload)

    def test_missing_extra_or_changed_source_keys_rejected(self):
        for action in ('missing', 'extra', 'source_addition'):
            bad = proof_fixture(self.payload)
            if action == 'missing':
                del bad['returncode']
            elif action == 'extra':
                bad['healthy'] = True
            else:
                bad['source_after_tests_sha256']['untracked.py'] = 'f' * 64
            with self.subTest(action=action), self.assertRaises(ValueError):
                m.validate_proof(bad, self.payload)

    def test_each_shell_check_independently_bound_and_zero_exit(self):
        for index in range(len(m.SHELL_FILES)):
            bad = proof_fixture(self.payload)
            bad['shell_syntax_checks'][index]['returncode'] = 2
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.validate_proof(bad, self.payload)
        bad = proof_fixture(self.payload)
        bad['shell_syntax_checks'].reverse()
        with self.assertRaises(ValueError):
            m.validate_proof(bad, self.payload)

    def test_zero_skip_failures_errors_test_coverage_and_cuda_required(self):
        mutations = [('tests_run', 0), ('tests_run', True), ('skipped', [['case', 'missing Linux']]),
            ('failures', 1), ('errors', 1), ('failure_details', [{'test': 'case', 'traceback': 'failure'}]),
            ('expected_failures', 1), ('unexpected_successes', 1),
            ('expected_failure_details', [{'test': 'case', 'traceback': 'expected failure'}]),
            ('unexpected_success_details', ['case']),
            ('successful', False), ('cuda_available', True), ('cuda_initialized_before', True),
            ('cuda_initialized_after', True), ('cuda_visible_devices', '0'), ('system', 'Darwin'),
            ('python_executable', '/local/python'), ('python_version', None), ('module_test_counts', {}),
            ('modules', []), ('runner_error', 'import failed'), ('expected_distribution_versions_matched', False)]
        for key, value in mutations:
            with self.subTest(key=key):
                summary = summary_fixture()
                summary[key] = value
                with self.assertRaises(ValueError):
                    m.validate_proof(proof_fixture(self.payload, summary), self.payload)
        summary = summary_fixture()
        summary['module_test_counts'][m.MODULES[0]] = 0
        summary['tests_run'] -= 1
        with self.assertRaises(ValueError):
            m.validate_proof(proof_fixture(self.payload, summary), self.payload)

    def test_actual_distribution_versions_checked_independently(self):
        for name in ('torch', 'transformers', 'vllm', 'tensordict'):
            summary = summary_fixture()
            summary['distribution_versions'][name] = 'wrong'
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'versions differ'):
                m.validate_proof(proof_fixture(self.payload, summary), self.payload)

    def test_summary_exact_schema_and_unique_actual_marker(self):
        summary = summary_fixture()
        summary['unexpected'] = True
        with self.assertRaisesRegex(ValueError, 'schema'):
            m.validate_proof(proof_fixture(self.payload, summary), self.payload)
        for stdout in ('{}', m.SUMMARY_PREFIX + '{}\n' + m.SUMMARY_PREFIX + '{}\n'):
            bad = proof_fixture(self.payload)
            bad['stdout'] = stdout
            with self.assertRaisesRegex(ValueError, 'Exactly one'):
                m.validate_proof(bad, self.payload)
        bad = proof_fixture(self.payload)
        bad['stdout'] = 'child process log\n' + bad['stdout']
        self.assertEqual(m.validate_proof(bad, self.payload)['tests_run'], len(m.MODULES))


class ClosedTargetInventoryTests(unittest.TestCase):
    """Exercise the actual remote inventory function on an isolated payload tree."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / 'repo'
        self.repo.mkdir()
        self.payload = payload_fixture()
        for name, data in self.payload['files'].items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(data, validate=True))
        tree = ast.parse(m.REMOTE_CODE)
        helpers = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                        node.name == name) for name in ('safe_regular_bytes', 'inventory')]
        module = ast.fix_missing_locations(ast.Module(body=helpers, type_ignores=[]))
        namespace = dict(repo=self.repo, p=self.payload, Path=Path, os=os, stat=stat,
                         need=m.require, require=m.require, sha=m.sha,
                         decoded={name: base64.b64decode(data, validate=True)
                                  for name, data in self.payload['files'].items()})
        # Only this pure filesystem reader is loaded; remote orchestration,
        # subprocesses and controller code are never executed by the fixture.
        exec(compile(module, 'actual remote closed-target inventory fixture', 'exec'), namespace)
        self.inventory = namespace['inventory']

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_payload_tree_matches_complete_map_and_count(self):
        actual = self.inventory()
        self.assertEqual(actual, self.payload['source_sha256'])
        self.assertEqual(len(actual), self.payload['expected_source_count'])

    def test_nonruntime_added_files_are_hashed_and_rejected_in_before_after_receipts(self):
        for name in ('extra.txt', 'extension.so'):
            path = self.repo / name
            path.write_bytes(b'unexpected nonruntime bytes')
            changed = self.inventory()
            self.assertEqual(changed[name], m.sha(path.read_bytes()))
            self.assertEqual(len(changed), self.payload['expected_source_count'] + 1)
            for slot in ('source_before_tests_sha256', 'source_after_tests_sha256'):
                proof = proof_fixture(self.payload)
                proof[slot] = changed
                if slot == 'source_after_tests_sha256':
                    proof['source_sha256'] = changed
                    proof['actual_source_count'] = len(changed)
                with self.subTest(name=name, slot=slot), self.assertRaisesRegex(ValueError, 'source map differs'):
                    m.validate_proof(proof, self.payload)
            path.unlink()

    def test_excluded_cache_injection_and_empty_added_directory_rejected(self):
        injected = self.repo / 'cache' / 'injected.py'
        injected.parent.mkdir()
        injected.write_bytes(b'injected = True\n')
        with self.assertRaisesRegex(ValueError, 'directory graph differs'):
            self.inventory()
        injected.unlink()
        with self.assertRaisesRegex(ValueError, 'directory graph differs'):
            self.inventory()
        injected.parent.rmdir()
        (self.repo / 'unexpected-empty').mkdir()
        with self.assertRaisesRegex(ValueError, 'directory graph differs'):
            self.inventory()

    def test_target_walk_propagates_unreadable_root_and_expected_child_errors(self):
        real_scandir = os.scandir
        for unreadable in (self.repo, self.repo / 'reproduction'):
            def scan(path):
                if Path(path) == unreadable:
                    raise PermissionError('constructed unreadable target directory')
                return real_scandir(path)
            with self.subTest(path=unreadable), patch.object(os, 'scandir', side_effect=scan), \
                    self.assertRaisesRegex(PermissionError, 'unreadable target directory'):
                self.inventory()

    def test_discovered_extra_unreadable_child_rejected_before_descent(self):
        unreadable = self.repo / 'extra-unreadable'
        unreadable.mkdir()
        real_scandir = os.scandir
        scanned = []
        def scan(path):
            scanned.append(Path(path))
            if Path(path) == unreadable:
                raise PermissionError('constructed unreadable extra directory')
            return real_scandir(path)
        with patch.object(os, 'scandir', side_effect=scan), \
                self.assertRaisesRegex(ValueError, 'directory graph differs'):
            self.inventory()
        self.assertNotIn(unreadable, scanned)

    def test_existing_python_bytes_drift_rejected_in_complete_proof(self):
        path = self.repo / 'module.py'
        path.write_bytes(b'!' * len(path.read_bytes()))
        changed = self.inventory()
        self.assertEqual(len(changed), self.payload['expected_source_count'])
        self.assertNotEqual(changed, self.payload['source_sha256'])
        proof = proof_fixture(self.payload)
        proof.update(source_sha256=changed, source_after_tests_sha256=changed,
                     actual_source_count=len(changed))
        with self.assertRaisesRegex(ValueError, 'source map differs'):
            m.validate_proof(proof, self.payload)

    def test_any_added_file_directory_or_broken_symlink_rejected(self):
        for name, target, directory in (
                ('extra.txt', self.repo / 'module.py', False),
                ('source-dir-link', self.repo / 'reproduction', True),
                ('cache', self.repo / 'reproduction', True),
                ('broken-link', self.repo / 'does-not-exist', False)):
            path = self.repo / name
            path.symlink_to(target, target_is_directory=directory)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'link/nonordinary|ordinary regular file'):
                self.inventory()
            path.unlink()

    def test_nonordinary_fifo_rejected_without_opening_it(self):
        path = self.repo / 'not-ordinary.txt'
        os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, 'ordinary regular file'):
            self.inventory()

    def test_counts_cannot_disguise_a_changed_target_map(self):
        proof = proof_fixture(self.payload)
        proof['actual_source_count'] += 1
        with self.assertRaisesRegex(ValueError, 'source or file-backed runner changed'):
            m.validate_proof(proof, self.payload)


class SafeRegularReaderTests(unittest.TestCase):
    """Actual remote reader definitions with regular/FIFO/race temp fixtures."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        tree = ast.parse(m.REMOTE_CODE)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                      node.name == 'safe_regular_bytes')
        module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
        namespace = dict(Path=Path, os=os, stat=stat, require=m.require)
        exec(compile(module, 'actual remote safe regular-file reader fixture', 'exec'), namespace)
        self.read = namespace['safe_regular_bytes']

    def tearDown(self):
        self.temporary.cleanup()

    def test_regular_runner_before_after_hash_and_empty_source_pass(self):
        runner = self.root / 'cpu_runner.py'
        raw = m.CPU_RUNNER.encode()
        runner.write_bytes(raw)
        before = m.sha(self.read(runner, expected_size=len(raw)))
        after = m.sha(self.read(runner, expected_size=len(raw)))
        self.assertEqual(before, m.sha(raw))
        self.assertEqual(before, after)
        empty = self.root / 'empty.py'
        empty.write_bytes(b'')
        self.assertEqual(self.read(empty, expected_size=0), b'')

    def test_local_and_remote_reader_asts_are_identical_and_packaging_uses_it(self):
        local_tree = ast.parse(m.safe_regular_bytes(Path(m.__file__)).decode())
        local = next(node for node in local_tree.body if isinstance(node, ast.FunctionDef) and
                     node.name == 'safe_regular_bytes')
        remote_tree = ast.parse(m.REMOTE_CODE)
        remote = next(node for node in remote_tree.body if isinstance(node, ast.FunctionDef) and
                      node.name == 'safe_regular_bytes')
        self.assertEqual(ast.dump(local, include_attributes=False), ast.dump(remote, include_attributes=False))
        for name in ('runtime_map', 'build_payload'):
            function = next(node for node in local_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                                node.func.id == 'safe_regular_bytes' for node in ast.walk(function)))
            self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                                 node.func.attr == 'read_bytes' for node in ast.walk(function)))

    def test_symlink_directory_wrong_expected_size_and_max_bound_rejected(self):
        original = self.root / 'source.py'
        original.write_bytes(b'12345678')
        link = self.root / 'linked.py'
        link.symlink_to(original)
        for path in (link, self.root):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'ordinary regular file'):
                self.read(path)
        with patch.object(os, 'read') as read:
            with self.assertRaisesRegex(ValueError, 'size differs from bound'):
                self.read(original, expected_size=7)
            with self.assertRaisesRegex(ValueError, 'size differs from bound'):
                self.read(original, max_bytes=4)
            read.assert_not_called()

    def test_inode_substituted_between_lstat_and_open_rejected_and_fd_closed(self):
        path = self.root / 'source.py'
        replacement = self.root / 'replacement.py'
        path.write_bytes(b'same bytes')
        replacement.write_bytes(b'same bytes')
        real_open, real_close = os.open, os.close
        def substitute(filename, flags):
            os.replace(replacement, path)
            return real_open(filename, flags)
        with patch.object(os, 'open', side_effect=substitute), \
                patch.object(os, 'close', wraps=real_close) as close:
            with self.assertRaisesRegex(ValueError, 'inode/file changed before open'):
                self.read(path, expected_size=10)
            self.assertEqual(close.call_count, 1)

    def test_inode_substituted_during_stream_read_rejected(self):
        path = self.root / 'source.py'
        replacement = self.root / 'replacement.py'
        path.write_bytes(b'same bytes')
        replacement.write_bytes(b'same bytes')
        real_read = os.read
        replaced = []
        def substitute(fd, length):
            data = real_read(fd, length)
            if not replaced:
                os.replace(replacement, path)
                replaced.append(True)
            return data
        with patch.object(os, 'read', side_effect=substitute), \
                self.assertRaisesRegex(ValueError, 'inode/file changed during read'):
            self.read(path, expected_size=10)

    def test_file_growth_cannot_overrun_expected_stream_bound(self):
        path = self.root / 'source.py'
        path.write_bytes(b'1234')
        real_read = os.read
        grew = []
        def append_after_first_chunk(fd, length):
            data = real_read(fd, length)
            if data and not grew:
                with path.open('ab') as stream:
                    stream.write(b'5')
                grew.append(True)
            return data
        with patch.object(os, 'read', side_effect=append_after_first_chunk), \
                self.assertRaisesRegex(ValueError, 'grew beyond bounded size'):
            self.read(path, expected_size=4)

    def test_fifo_substitution_open_is_nonblocking_and_rejected_within_bounded_child(self):
        regular = self.root / 'original.py'
        regular.write_bytes(b'x')
        initial_stat = regular.lstat()
        fifo = self.root / 'cpu_runner.py'
        os.mkfifo(fifo)
        context = multiprocessing.get_context('fork')
        output = context.Queue()
        read = self.read
        def child():
            started = time.monotonic()
            flags_seen = []
            real_open, real_close = os.open, os.close
            def opening(path, flags):
                flags_seen.append(flags)
                return real_open(path, flags)
            with patch.object(Path, 'lstat', return_value=initial_stat), \
                    patch.object(os, 'open', side_effect=opening), \
                    patch.object(os, 'close', wraps=real_close) as close:
                try:
                    read(fifo, expected_size=1)
                    result = dict(error=None)
                except BaseException as error:
                    result = dict(error=type(error).__name__, detail=str(error))
                result.update(duration=time.monotonic() - started, flags=flags_seen,
                              fd_close_calls=close.call_count)
            output.put(result)
        process = context.Process(target=child)
        process.start()
        process.join(timeout=3)
        blocked = process.is_alive()
        if blocked:
            process.terminate()
            process.join(timeout=3)
        try:
            self.assertFalse(blocked, 'FIFO substitution blocked the actual remote reader')
            self.assertEqual(process.exitcode, 0)
            result = output.get(timeout=1)
            self.assertEqual(result['error'], 'ValueError')
            self.assertIn('opened a nonordinary file', result['detail'])
            self.assertLess(result['duration'], 2)
            self.assertEqual(result['fd_close_calls'], 1)
            self.assertTrue(result['flags'][0] & os.O_NONBLOCK)
            self.assertTrue(result['flags'][0] & os.O_NOFOLLOW)
        finally:
            output.close()
            output.join_thread()

    def test_post_test_runner_read_failure_is_an_unaccepted_proof(self):
        payload = payload_fixture()
        proof = proof_fixture(payload)
        proof.update(cpu_runner_after_sha256=None, cpu_runner_unchanged=False,
                     runner_read_error='ValueError: Safe reader requires an ordinary regular file')
        with self.assertRaisesRegex(ValueError, 'file-backed runner changed'):
            m.validate_proof(proof, payload)
        self.assertIn('runner_after=None;runner_read_error=', m.REMOTE_CODE)
        self.assertIn('sha(safe_regular_bytes(runner,expected_size=runner_size))', m.REMOTE_CODE)
        self.assertNotIn('runner.read_bytes()', m.REMOTE_CODE)
        self.assertNotIn('sha(path.read_bytes())', m.REMOTE_CODE)


class ActualUnittestOutcomeTests(unittest.TestCase):
    """Run the actual CPU runner with stdlib test suites and dependency stubs."""
    def run_actual(self, outcome):
        namespace = {'__name__': 'constructed_cpu_runner_fixture'}
        exec(compile(m.CPU_RUNNER, 'actual guarded CPU runner fixture', 'exec'), namespace)
        class Passing(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)
        class ExpectedFailure(unittest.TestCase):
            @unittest.expectedFailure
            def test_expected_failure(self):
                self.fail('real constructed expected failure')
        class UnexpectedSuccess(unittest.TestCase):
            @unittest.expectedFailure
            def test_unexpected_success(self):
                self.assertTrue(True)
        class Skipped(unittest.TestCase):
            def test_skipped(self):
                self.skipTest('real constructed skip reason')
        class Failure(unittest.TestCase):
            def test_failure(self):
                self.fail('real constructed failure')
        class Error(unittest.TestCase):
            def test_error(self):
                raise RuntimeError('real constructed error')
        selected = dict(pass_=Passing, expected_failure=ExpectedFailure,
                        unexpected_success=UnexpectedSuccess, skip=Skipped,
                        failure=Failure, error=Error)[outcome]
        def suite_for(module):
            case = selected if module == m.MODULES[0] else Passing
            return unittest.TestSuite([case(next(name for name in dir(case) if name.startswith('test_')))])
        torch = types.ModuleType('torch')
        torch.__version__ = '2.4.0+cu121'
        torch.cuda = types.SimpleNamespace(is_initialized=lambda: False, is_available=lambda: False)
        transformers = types.ModuleType('transformers')
        transformers.__version__ = '4.46.3'
        vllm = types.ModuleType('vllm')
        vllm.__version__ = 'dev'
        versions = dict(torch='2.4.0+cu121', transformers='4.46.3', vllm='0.6.3', tensordict='0.5.0')
        stdout = io.StringIO()
        with patch.dict(sys.modules, torch=torch, transformers=transformers, vllm=vllm), \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), \
                patch('importlib.metadata.version', side_effect=versions.__getitem__), \
                patch('platform.system', return_value='Linux'), \
                patch.object(sys, 'executable', m.TARGET_PYTHON), \
                patch.object(unittest.defaultTestLoader, 'loadTestsFromName', side_effect=suite_for), \
                contextlib.redirect_stdout(stdout):
            code = namespace['main'](list(m.MODULES))
        return code, m.extract_summary(stdout.getvalue())

    def test_real_expected_failure_is_reported_and_rejected_despite_was_successful(self):
        code, summary = self.run_actual('expected_failure')
        self.assertTrue(summary['successful'])
        self.assertEqual(summary['failures'], 0)
        self.assertEqual(summary['errors'], 0)
        self.assertEqual(summary['expected_failures'], 1)
        self.assertEqual(len(summary['expected_failure_details']), 1)
        self.assertIsInstance(summary['expected_failure_details'][0]['test'], str)
        self.assertIn('real constructed expected failure', summary['expected_failure_details'][0]['traceback'])
        self.assertEqual(code, 1)
        payload = payload_fixture()
        with self.assertRaisesRegex(ValueError, 'zero-skip CPU tests not accepted'):
            m.validate_proof(proof_fixture(payload, summary), payload)

    def test_real_unexpected_success_is_explicitly_reported_and_rejected(self):
        code, summary = self.run_actual('unexpected_success')
        self.assertEqual(summary['unexpected_successes'], 1)
        self.assertEqual(len(summary['unexpected_success_details']), 1)
        self.assertIsInstance(summary['unexpected_success_details'][0], str)
        self.assertEqual(code, 1)

    def test_real_skip_has_string_case_and_reason_and_is_rejected(self):
        code, summary = self.run_actual('skip')
        self.assertTrue(summary['successful'])
        self.assertEqual(summary['tests_run'], len(m.MODULES))
        self.assertEqual(summary['failures'], 0)
        self.assertEqual(summary['errors'], 0)
        self.assertEqual(len(summary['skipped']), 1)
        case, reason = summary['skipped'][0]
        self.assertIsInstance(case, str)
        self.assertTrue(case.endswith('.Skipped.test_skipped'))
        self.assertEqual(reason, 'real constructed skip reason')
        self.assertEqual(code, 1)
        payload = payload_fixture()
        with self.assertRaisesRegex(ValueError, 'zero-skip CPU tests not accepted'):
            m.validate_proof(proof_fixture(payload, summary), payload)

    def test_real_failure_records_actual_traceback_and_is_rejected(self):
        code, summary = self.run_actual('failure')
        self.assertFalse(summary['successful'])
        self.assertEqual(summary['tests_run'], len(m.MODULES))
        self.assertEqual(summary['failures'], 1)
        self.assertEqual(summary['errors'], 0)
        self.assertEqual(summary['skipped'], [])
        self.assertEqual(len(summary['failure_details']), 1)
        detail = summary['failure_details'][0]
        self.assertIsInstance(detail['test'], str)
        self.assertTrue(detail['test'].endswith('.Failure.test_failure'))
        self.assertIn('AssertionError: real constructed failure', detail['traceback'])
        self.assertEqual(code, 1)
        payload = payload_fixture()
        with self.assertRaisesRegex(ValueError, 'zero-skip CPU tests not accepted'):
            m.validate_proof(proof_fixture(payload, summary), payload)

    def test_real_error_records_actual_traceback_and_is_rejected(self):
        code, summary = self.run_actual('error')
        self.assertFalse(summary['successful'])
        self.assertEqual(summary['tests_run'], len(m.MODULES))
        self.assertEqual(summary['failures'], 0)
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['skipped'], [])
        self.assertEqual(len(summary['error_details']), 1)
        detail = summary['error_details'][0]
        self.assertIsInstance(detail['test'], str)
        self.assertTrue(detail['test'].endswith('.Error.test_error'))
        self.assertIn('RuntimeError: real constructed error', detail['traceback'])
        self.assertEqual(code, 1)
        payload = payload_fixture()
        with self.assertRaisesRegex(ValueError, 'zero-skip CPU tests not accepted'):
            m.validate_proof(proof_fixture(payload, summary), payload)

    def test_real_all_passing_suite_has_zero_expected_outcomes_and_is_accepted(self):
        code, summary = self.run_actual('pass_')
        self.assertEqual(code, 0)
        self.assertEqual(summary['expected_failures'], 0)
        self.assertEqual(summary['unexpected_successes'], 0)
        self.assertEqual(summary['expected_failure_details'], [])
        self.assertEqual(summary['unexpected_success_details'], [])
        payload = payload_fixture()
        self.assertEqual(m.validate_proof(proof_fixture(payload, summary), payload)['tests_run'], len(m.MODULES))


class SideEffectBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.payload = payload_fixture()

    def test_default_and_check_inputs_make_no_claim_rpc_or_controller_import(self):
        for flag in ([], ['--check-inputs']):
            before_path = list(sys.path)
            with tempfile.TemporaryDirectory() as folder, patch.object(m, 'PREP', Path(folder)), \
                    patch.object(m, 'build_payload', return_value=self.payload), \
                    patch.object(m, 'exclusive_json') as write, \
                    patch.object(m.subprocess, 'run') as rpc, patch('builtins.print'):
                self.assertEqual(m.main(['--source-commit', COMMIT, *flag]), 0)
                write.assert_not_called()
                rpc.assert_not_called()
                self.assertEqual(list(Path(folder).iterdir()), [])
                self.assertEqual(sys.path, before_path)

    def test_mode_flags_are_mutually_exclusive(self):
        with patch.object(m, 'build_payload') as build, patch('sys.stderr'), self.assertRaises(SystemExit):
            m.main(['--source-commit', COMMIT, '--execute', '--check-inputs'])
        build.assert_not_called()

    def test_receipts_exclusive_and_prior_claim_prevents_route(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / (m.STEM + '-claim.json')
            m.exclusive_json(path, {'fixture': 'immutable'})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                m.exclusive_json(path, {'overwrite': True})
            self.assertEqual(path.read_bytes(), before)
            with patch.object(m, 'PREP', Path(folder)), patch.object(m, 'build_payload', return_value=self.payload), \
                    patch.object(m.subprocess, 'run') as rpc:
                with self.assertRaisesRegex(ValueError, 'same handle'):
                    m.main(['--source-commit', COMMIT, '--execute'])
                rpc.assert_not_called()
                self.assertEqual(path.read_bytes(), before)

    def test_explicit_execution_uses_only_existing_h1008_route_and_preserves_claim(self):
        controller = types.SimpleNamespace(route=lambda host, command: ['MOCKED_ROUTE', host, command])
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(proof_fixture(self.payload)), stderr='')
        with tempfile.TemporaryDirectory() as folder, patch.object(m, 'PREP', Path(folder)), \
                patch.object(m, 'build_payload', return_value=self.payload), \
                patch.dict(sys.modules, control=controller), patch.object(sys, 'path', list(sys.path)), \
                patch.object(m.subprocess, 'run', return_value=result) as rpc, patch('builtins.print'):
            self.assertEqual(m.main(['--source-commit', COMMIT, '--execute']), 0)
            args, kwargs = rpc.call_args
            self.assertEqual(args[0][0:2], ['MOCKED_ROUTE', 'h100-8'])
            self.assertEqual(kwargs['timeout'], m.RPC_TIMEOUT_SECONDS)
            self.assertEqual(m.strict_json(kwargs['input']), self.payload)
            claim = m.strict_json((Path(folder) / (m.STEM + '-claim.json')).read_bytes())
            self.assertEqual(claim['complete_payload_sha256'], m.sha(m.canonical(self.payload)))
            self.assertEqual(claim['remote_code_sha256'], m.sha(m.REMOTE_CODE.encode()))
            self.assertNotIn('files', claim)
            self.assertEqual(set(path.name for path in Path(folder).iterdir()),
                             {m.STEM + '-claim.json', m.STEM + '-result.json'})

    def test_rpc_timeout_retains_owned_claim_and_does_not_restart(self):
        controller = types.SimpleNamespace(route=lambda host, command: ['MOCKED_ROUTE', host, command])
        with tempfile.TemporaryDirectory() as folder, patch.object(m, 'PREP', Path(folder)), \
                patch.object(m, 'build_payload', return_value=self.payload), \
                patch.dict(sys.modules, control=controller), patch.object(sys, 'path', list(sys.path)), \
                patch.object(m.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['mock'], 1)) as rpc:
            with self.assertRaisesRegex(RuntimeError, 'retain claim'):
                m.main(['--source-commit', COMMIT, '--execute'])
            self.assertEqual(rpc.call_count, 1)
            self.assertTrue((Path(folder) / (m.STEM + '-claim.json')).is_file())
            self.assertFalse((Path(folder) / (m.STEM + '-result.json')).exists())

    def test_failed_remote_tests_receipt_saved_without_admission(self):
        controller = types.SimpleNamespace(route=lambda host, command: ['MOCKED_ROUTE', host, command])
        bad = proof_fixture(self.payload)
        bad['returncode'] = 1
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(bad), stderr='')
        with tempfile.TemporaryDirectory() as folder, patch.object(m, 'PREP', Path(folder)), \
                patch.object(m, 'build_payload', return_value=self.payload), \
                patch.dict(sys.modules, control=controller), patch.object(sys, 'path', list(sys.path)), \
                patch.object(m.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(ValueError, 'process failed'):
                m.main(['--source-commit', COMMIT, '--execute'])
            saved = m.strict_json((Path(folder) / (m.STEM + '-result.json')).read_bytes())
            self.assertEqual(saved, bad)
            self.assertFalse(saved['production_admission'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
