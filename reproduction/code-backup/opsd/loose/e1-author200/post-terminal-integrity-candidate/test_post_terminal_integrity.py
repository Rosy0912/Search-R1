"""Small local CPU boundary fixtures; never run the production inspection."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
LOCAL_REPO = WORK.parent / 'opsd-e1-300'
SNAPSHOT = WORK / 'actual-snapshots/3261-20261006T005604410916Z/original-files'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def exact_module(name, path):
    raw = path.read_bytes()
    value = types.ModuleType(name)
    value.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), value.__dict__)
    return value


candidate = exact_module('tested_post_terminal_integrity', HERE / 'post_terminal_integrity.py')
shared_path = WORK / 'final-acceptance-90c2695a7f46/author200_observation.py'
assert digest(shared_path.read_bytes()) == candidate.COMMON_SHA
shared = exact_module('exact_existing_shared90_for_cpu_fixture', shared_path)


def proc_stat(pid, start, state='S'):
    # Field3 starts after comm; the actual proc starttime is field22.
    fields = [state] + ['0'] * 18 + [str(start)]
    return str(pid) + ' (a name with spaces) ' + ' '.join(fields) + '\n'


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.old_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True

    def tearDown(self):
        sys.dont_write_bytecode = self.old_bytecode
        self.tmp.cleanup()

    def file(self, name='asset', raw=b'known content'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path, digest(raw)


class CPUFixtures(_Fixture):
    def test_verified_compile_exact_bytes(self):
        path, pin = self.file('module.py', b'VALUE = 17\n')
        mod = candidate.module_bytes('test_verified', path, pin)
        self.assertEqual(mod.VALUE, 17)
        self.assertEqual(mod.__file__, str(path))

    def test_changed_code_refused_before_exec(self):
        marker = self.root / 'never'
        path, pin = self.file('module.py', b'VALUE = 17\n')
        path.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n')
        with self.assertRaises(ValueError):
            candidate.module_bytes('test_unapproved', path, pin)
        self.assertFalse(marker.exists())

    def test_symlink_file_and_parent_refused(self):
        path, pin = self.file()
        link = self.root / 'link'
        link.symlink_to(path)
        with self.assertRaises(OSError):
            candidate.verified_bytes(link, pin)
        directory = self.root / 'dir'
        directory.mkdir()
        (directory / 'item').write_bytes(b'data')
        alias = self.root / 'diralias'
        alias.symlink_to(directory, target_is_directory=True)
        with self.assertRaises(OSError):
            candidate.verified_bytes(alias / 'item', digest(b'data'))

    def test_executable_nonregular_and_limit_rejected(self):
        path, pin = self.file()
        with self.assertRaises(ValueError):
            candidate.verified_bytes(path, pin, 1)
        with self.assertRaises(ValueError):
            candidate.verified_bytes(self.root, digest(b''))

    def test_executable_typed_pin_rejected(self):
        path, _ = self.file()
        for bad in (None, False, 'A' * 64, '0' * 63):
            with self.subTest(pin=bad), self.assertRaises(ValueError):
                candidate.verified_bytes(path, bad)

    def test_executable_replacement_during_read_rejected(self):
        path, pin = self.file()
        replacement, _ = self.file('replacement', b'known content')
        original = candidate.physical_fd
        calls = []
        def replace_on_reopen(value):
            calls.append(value)
            if len(calls) == 2:
                replacement.replace(path)
            return original(value)
        with mock.patch.object(candidate, 'physical_fd', side_effect=replace_on_reopen):
            with self.assertRaises(ValueError):
                candidate.verified_bytes(path, pin)

    def test_asset_same_fd_full_hash_and_reuse(self):
        path, pin = self.file()
        proofs = candidate.AssetProofs(shared)
        one = proofs.add('actor', path, self.root, pin, len(b'known content'))
        two = proofs.add('teacher', path, self.root, pin, len(b'known content'))
        self.assertIs(one, two)
        proofs.recheck()
        result = proofs.result()
        self.assertEqual(result['unique_files'], 1)
        self.assertEqual(result['single_stable_fd_full_hash_bytes'], 13)
        self.assertEqual(result['files'][str(path)]['inode'], path.stat().st_ino)
        self.assertEqual(set(one), {'physical_path', 'sha256', 'bytes', 'device', 'inode', 'uid'})

    def test_asset_conflicting_pin_duplicate_role_and_bool_bytes(self):
        path, pin = self.file()
        proofs = candidate.AssetProofs(shared)
        proofs.add('actor', path, self.root, pin)
        for role, bad_pin, count in (('actor', pin, None), ('teacher', '0' * 64, None), ('boolbytes', pin, True)):
            with self.subTest(role=role), self.assertRaises(ValueError):
                proofs.add(role, path, self.root, bad_pin, count)

    def test_asset_changed_after_hash_and_legacy_validator_rejected(self):
        path, pin = self.file()
        proofs = candidate.AssetProofs(shared)
        proofs.add('actor', path, self.root, pin)
        path.write_bytes(b'changed bytes')
        with self.assertRaises(ValueError):
            proofs.recheck()

    def test_asset_parent_escape_and_symlink_rejected(self):
        path, pin = self.file()
        proofs = candidate.AssetProofs(shared)
        outside = self.root / 'nested'
        outside.mkdir()
        with self.assertRaises(ValueError):
            proofs.add('escape', path, outside, pin)
        link = self.root / 'link'
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            proofs.add('symlink', link, self.root, pin)

    def test_stable_asset_replacement_in_same_fd_read_rejected(self):
        path, pin = self.file()
        replacement, _ = self.file('replacement', b'known content')
        original = shared.os.read
        done = []
        def racing_read(fd, size):
            block = original(fd, size)
            if block and not done:
                replacement.replace(path)
                done.append(True)
            return block
        with mock.patch.object(shared.os, 'read', side_effect=racing_read):
            with self.assertRaises(ValueError):
                candidate.AssetProofs(shared).add('race', path, self.root, pin)

    def test_logical_components_do_not_normalize_escapes(self):
        for bad in ('../asset', 'a/./x', 'a//x', '/asset', 'a/../x', 'a\\x', '', False):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                candidate.logical_path(self.root, bad)
        self.assertEqual(candidate.logical_path(self.root, 'a/x'), self.root / 'a/x')

    def test_original_driver_absence_not_release(self):
        record = candidate.origin_driver(self.root)
        self.assertEqual(record['status'], 'ORIGINAL_ABSENT_AT_OBSERVATION')
        self.assertNotIn('resource_release', record)

    def test_original_driver_live_refused(self):
        folder = self.root / str(candidate.DRIVER_PID)
        folder.mkdir()
        (folder / 'stat').write_text(proc_stat(candidate.DRIVER_PID, candidate.DRIVER_START))
        with self.assertRaises(ValueError):
            candidate.origin_driver(self.root)

    def test_original_driver_reuse_and_zombie_are_separate(self):
        folder = self.root / str(candidate.DRIVER_PID)
        folder.mkdir()
        path = folder / 'stat'
        path.write_text(proc_stat(candidate.DRIVER_PID, '100'))
        self.assertEqual(candidate.origin_driver(self.root)['status'], 'PID_REUSED_ORIGINAL_GONE')
        path.write_text(proc_stat(candidate.DRIVER_PID, candidate.DRIVER_START, 'Z'))
        self.assertEqual(candidate.origin_driver(self.root)['status'], 'ORIGINAL_EXITED_ZOMBIE')

    def test_bad_original_stat_not_absence(self):
        folder = self.root / str(candidate.DRIVER_PID)
        folder.mkdir()
        (folder / 'stat').write_text('1 (x) S 1\n')
        with self.assertRaises(ValueError):
            candidate.origin_driver(self.root)

    def test_new_execution_identity_uses_actual_new_environment(self):
        folder = self.root / 'self'
        folder.mkdir()
        uid = os.getuid()
        (folder / 'status').write_text('Uid:\t' + '\t'.join([str(uid)] * 4) + '\n')
        (folder / 'stat').write_text(proc_stat(os.getpid(), 123456))
        (folder / 'cgroup').write_text('1:freezer:/slurm_h100-10/uid_0/job_12345/step_7\n')
        env = {'SLURM_JOB_ID': '12345', 'SLURM_STEP_ID': '7', 'CUDA_VISIBLE_DEVICES': ''}
        rec = candidate.execution_identity(self.root, env)
        self.assertEqual(rec['pid'], os.getpid())
        self.assertEqual(rec['start_ticks'], '123456')
        self.assertEqual(rec['slurm_job_id'], '12345')
        self.assertNotEqual(rec['slurm_job_id'], '3261')
        self.assertEqual(rec['uids'], [uid] * 4)
        for job, step in (('3261', '7'), ('12345', '8'), ('12345', '7/other'), (False, '7')):
            with self.subTest(job=job, step=step), self.assertRaises(ValueError):
                candidate.execution_identity(self.root, dict(env, SLURM_JOB_ID=job, SLURM_STEP_ID=step))

    def test_new_execution_four_uids_required(self):
        folder = self.root / 'self'
        folder.mkdir()
        uid = os.getuid()
        for bad in ('Uid:\t' + str(uid) + '\n', 'Uid:\t' + '\t'.join([str(uid)] * 3 + [str(uid + 1)]) + '\n', 'No Uid\n'):
            (folder / 'status').write_text(bad)
            with self.subTest(status=bad), self.assertRaises(ValueError):
                candidate.execution_identity(self.root, {})

    def test_native_cpu_version_and_no_cuda_initialization(self):
        fake = types.SimpleNamespace(__version__='2.4.0+cu121', cuda=types.SimpleNamespace(is_initialized=lambda: False))
        candidate.validate_cpu_runtime(fake)
        for version, initialized in (('2.8.0+cpu', False), ('2.4.0', True), ('2.4.0', 0)):
            bad = types.SimpleNamespace(__version__=version, cuda=types.SimpleNamespace(is_initialized=lambda: initialized))
            with self.subTest(version=version, initialized=initialized), self.assertRaises(ValueError):
                candidate.validate_cpu_runtime(bad)

    def test_production_cpu_only_guard_before_any_helpers(self):
        with mock.patch.object(candidate.os, 'getuid', return_value=0), mock.patch.object(candidate.socket, 'gethostname', return_value='h100-10'), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), mock.patch.object(candidate, 'shared_context', side_effect=AssertionError('Must not execute helpers')):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                candidate.inspect(candidate.RUN / 'evidence/test.json')

    def test_production_live_driver_guard_before_helpers(self):
        with mock.patch.object(candidate.os, 'getuid', return_value=0), mock.patch.object(candidate.socket, 'gethostname', return_value='h100-10'), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), mock.patch.object(candidate, 'execution_identity', return_value={}), mock.patch.object(candidate, 'origin_driver', side_effect=ValueError('Original training driver remains alive')), mock.patch.object(candidate, 'shared_context', side_effect=AssertionError('Must not execute helpers')):
            with self.assertRaisesRegex(ValueError, 'remains alive'):
                candidate.inspect(candidate.RUN / 'evidence/test.json')


class ExactImports(_Fixture):
    def tearDown(self):
        for name in tuple(sys.modules):
            if name == 'tiny_frozen' or name.startswith('tiny_frozen.'):
                del sys.modules[name]
        super().tearDown()

    def setup_package(self):
        init, pin = self.file('tiny_frozen/__init__.py', b'VALUE = "source"\n')
        leaf, leafpin = self.file('tiny_frozen/leaf.py', b'NUMBER = 5\n')
        return candidate.ExactSourceFinder({'tiny_frozen': {'path': str(init), 'sha256': pin}, 'tiny_frozen.leaf': {'path': str(leaf), 'sha256': leafpin}}, prefixes=('tiny_frozen',)), init, leaf

    def test_loader_never_executes_actual_stale_pyc(self):
        finder, init, leaf = self.setup_package()
        marker = self.root / 'bad-pyc-was-executed'
        good = leaf.read_bytes()
        leaf.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\nNUMBER=999\n')
        py_compile.compile(str(leaf), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
        leaf.write_bytes(good)
        finder.install()
        try:
            result = importlib.import_module('tiny_frozen.leaf')
            self.assertEqual(result.NUMBER, 5)
            self.assertFalse(marker.exists())
            self.assertIsNone(result.__cached__)
            self.assertTrue(finder.executed['tiny_frozen.leaf']['compiled_same_verified_bytes'])
        finally:
            finder.remove()

    def test_unknown_existing_project_submodule_refused(self):
        finder, _, _ = self.setup_package()
        self.file('tiny_frozen/unpinned.py', b'UNKNOWN=1\n')
        finder.install()
        try:
            with self.assertRaisesRegex(ValueError, 'Unpinned'):
                importlib.import_module('tiny_frozen.unpinned')
        finally:
            finder.remove()

    def test_modified_pinned_project_refused_before_exec(self):
        finder, _, leaf = self.setup_package()
        marker = self.root / 'badsource'
        leaf.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n')
        finder.install()
        try:
            with self.assertRaises(ValueError):
                importlib.import_module('tiny_frozen.leaf')
            self.assertFalse(marker.exists())
        finally:
            finder.remove()

    def test_preloaded_project_refused(self):
        finder, _, _ = self.setup_package()
        sys.modules['tiny_frozen'] = types.ModuleType('tiny_frozen')
        with self.assertRaisesRegex(ValueError, 'Preloaded'):
            finder.install()

    def test_bounded_namespace_no_initializer(self):
        path, pin = self.file('tiny_frozen/baselines/leaf.py', b'NUMBER = 7\n')
        finder = candidate.ExactSourceFinder({'tiny_frozen.baselines.leaf': {'path': str(path), 'sha256': pin}}, prefixes=('tiny_frozen',), namespaces={'tiny_frozen': path.parent.parent, 'tiny_frozen.baselines': path.parent})
        finder.install()
        try:
            self.assertEqual(importlib.import_module('tiny_frozen.baselines.leaf').NUMBER, 7)
            self.assertEqual(set(finder.namespace_observations), {'tiny_frozen', 'tiny_frozen.baselines'})
            self.assertNotIn('tiny_frozen', finder.executed)
        finally:
            finder.remove()

    def test_namespace_unpinned_init_or_symlink_refused(self):
        path, _ = self.file('ns/__init__.py', b'NEVER=1\n')
        with self.assertRaisesRegex(ValueError, 'initializer'):
            candidate.physical_namespace(path.parent)
        alias = self.root / 'nsalias'
        alias.symlink_to(path.parent, target_is_directory=True)
        with self.assertRaises(OSError):
            candidate.physical_namespace(alias)


class ActualFrozenReadOnlyJoin(_Fixture):
    def setUp(self):
        super().setUp()
        closure_raw = (SNAPSHOT / 'bindings/import_closure.json').read_bytes()
        self.assertEqual(digest(closure_raw), shared.CLOSURE_SHA)
        original = json.loads(closure_raw)['modules']
        records = {}
        for name, rec in original.items():
            if name.startswith(('search_r1', 'verl', 'reproduction')):
                path = LOCAL_REPO / Path(rec['path']).relative_to(candidate.REPO)
                self.assertEqual(digest(path.read_bytes()), rec['sha256'])
                records[name] = {'path': str(path), 'sha256': rec['sha256']}
        self.finder = candidate.ExactSourceFinder(records, namespaces={'reproduction': LOCAL_REPO / 'reproduction', 'reproduction.baselines': LOCAL_REPO / 'reproduction/baselines'})
        self.finder.install()
        self.profile = importlib.import_module('search_r1.baselines.e1_profile')
        self.holdout = importlib.import_module('search_r1.baselines.sharedholdout')

    def tearDown(self):
        self.finder.remove()
        for name in tuple(sys.modules):
            if self.finder.targeted(name):
                del sys.modules[name]
        super().tearDown()

    def config(self):
        path = SNAPSHOT / 'full_state/global_step_5/resolved-config.json'
        raw = path.read_bytes()
        self.assertEqual(digest(raw), shared.CONFIG_SHA)
        from omegaconf import OmegaConf
        return OmegaConf.create(json.loads(raw))

    def test_actual_saved73f_profile_with_pinned188_helpers(self):
        cfg = self.config()
        resolved = self.profile.validate_profile(cfg)
        self.assertEqual(resolved['trainer']['total_training_steps'], 200)
        self.assertEqual(resolved['actor_rollout_ref']['actor']['optim']['lr'], 5e-6)
        self.assertFalse(resolved['actor_rollout_ref']['actor']['use_kl_loss'])
        self.assertEqual(len(self.finder.records), 206)

    def test_actual_saved_config_algorithm_and_endpoint_drift_rejected(self):
        for key, value in (('actor_rollout_ref.actor.use_kl_loss', True), ('trainer.total_training_steps', 201), ('actor_rollout_ref.actor.optim.lr', 1e-6), ('retriever.url', 'http://other-node/retrieve')):
            cfg = self.config()
            from omegaconf import OmegaConf
            OmegaConf.update(cfg, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.profile.validate_profile(cfg)

    def test_actual_saved_four_data_input_join(self):
        cfg = self.config()
        expected = {'train': cfg.data.train_files, 'validation': cfg.data.val_files,
                    'final': cfg.data.final_val_files, 'stage': cfg.baselines.opsd_e1.stage_files}
        self.assertTrue(all(isinstance(v, str) for v in expected.values()))
        joined = candidate.validate_effective_paths(cfg, {'effective_paths': expected}, self.holdout.paths)
        self.assertEqual(joined['train'], [cfg.data.train_files])
        wrong = dict(expected)
        wrong['stage'] = wrong['validation']
        with self.assertRaises(ValueError):
            candidate.validate_effective_paths(cfg, {'effective_paths': wrong}, self.holdout.paths)

    def tiny_frame(self):
        import pandas as pd
        prefix = self.holdout.E1_INSTRUCTION + '\nQuestion: '
        rows = [dict(prompt=[{'role': 'user', 'content': prefix + 'First question?'}], extra_info={'index': 'nq:train:0'}, data_source='nq', reward_model={'style': 'rule', 'ground_truth': {'target': ['one']}}), dict(prompt=[{'role': 'user', 'content': prefix + 'Second question?'}], extra_info={'index': 'hotpotqa:train:1'}, data_source='hotpotqa', reward_model={'style': 'rule', 'ground_truth': {'target': ['two']}})]
        frame = pd.DataFrame(rows)
        path = self.root / 'tiny.parquet'
        frame.to_parquet(path)
        rec = {'sha256': digest(path.read_bytes()), 'rows': 2, 'source_counts': {'nq': 1, 'hotpotqa': 1}, 'question_ids_sha256': self.holdout.ordered_question_ids_hash(frame), 'canonical_questions_sha256': self.holdout.canonical_questions_hash(frame), 'content_identity': self.holdout.frame_identity(frame, 'prompt')}
        return path, rec, frame

    def test_real_frozen_small_parquet_content_identity(self):
        path, rec, _ = self.tiny_frame()
        frame, ids, questions = self.holdout._prebuilt_frame(path, rec, 'prompt', 'validation')
        self.assertEqual(ids, {'nq:train:0', 'hotpotqa:train:1'})
        self.assertEqual(len(questions), 2)

    def test_real_frozen_id_content_rows_corruption_rejected(self):
        path, rec, _ = self.tiny_frame()
        for key, value in (('question_ids_sha256', '0' * 64), ('rows', True), ('content_identity', {'sha256': '0' * 64}), ('source_counts', {'nq': 2})):
            changed = dict(rec)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.holdout._prebuilt_frame(path, changed, 'prompt', 'validation')


if __name__ == '__main__':
    unittest.main(verbosity=2)
