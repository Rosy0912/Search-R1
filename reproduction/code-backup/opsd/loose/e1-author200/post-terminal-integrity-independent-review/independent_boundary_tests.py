"""Independent small CPU counterexamples; never calls production inspect/main."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
W = HERE.parent
CANDIDATE = W / 'post-terminal-integrity-candidate/post_terminal_integrity.py'
EXPECTED = '90f7b3e9a8d7942fa2e44b5f8c76d8cd0104952f6718d5eea23acbbed4da69f4'


def load(name, path, pin):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError('Independent candidate pin drift')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


c = load('independent_post_terminal_candidate', CANDIDATE, EXPECTED)
shared = load('independent_exact_common9360', W / 'final-acceptance-90c2695a7f46/author200_observation.py', c.COMMON_SHA)


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True

    def tearDown(self):
        sys.dont_write_bytecode = self.bytecode
        for key in tuple(sys.modules):
            if key == 'boundary_exact' or key.startswith('boundary_exact.'):
                del sys.modules[key]
        self.directory.cleanup()

    def fake_self_proc(self, cgroup):
        p = self.root / 'self'
        p.mkdir()
        uid = os.getuid()
        (p / 'status').write_text('Uid:\t' + '\t'.join([str(uid)] * 4) + '\n')
        (p / 'stat').write_text(str(os.getpid()) + ' (fixture CPU) S ' + ' '.join(['0'] * 18 + ['654321']) + '\n')
        (p / 'cgroup').write_text(cgroup)

    def test_same_job_prefix_or_step_prefix_is_not_identity(self):
        self.fake_self_proc('1:freezer:/slurm_h100-10/uid_0/job_32610/step_00\n')
        env = {'SLURM_JOB_ID': '3261', 'SLURM_STEP_ID': '0', 'CUDA_VISIBLE_DEVICES': ''}
        with self.assertRaisesRegex(ValueError, 'job differs'):
            c.execution_identity(self.root, env)
        (self.root / 'self/cgroup').write_text('1:freezer:/slurm_h100-10/uid_0/job_3261/step_00\n')
        with self.assertRaisesRegex(ValueError, 'step differs'):
            c.execution_identity(self.root, env)

    def test_missing_slurm_environment_does_not_fill_training_job(self):
        self.fake_self_proc('0::/system.slice/ssh.service\n')
        actual = c.execution_identity(self.root, {'CUDA_VISIBLE_DEVICES': ''})
        self.assertIsNone(actual['slurm_job_id'])
        self.assertIsNone(actual['slurm_step_id'])
        self.assertEqual(actual['pid'], os.getpid())
        self.assertEqual(actual['start_ticks'], '654321')
        self.assertIn('NOT_SAVED_TRAINING3261', actual['scope'])

    def test_known_source_changed_then_restored_is_not_stable(self):
        path = self.root / 'code.py'
        original = b'VALUE=42\n'
        path.write_bytes(original)
        pin = hashlib.sha256(original).hexdigest()
        opened = c.physical_fd
        count = []
        def change_on_reopen(p):
            count.append(p)
            if len(count) == 2:
                path.write_bytes(b'VALUE=99\n')
                path.write_bytes(original)
            return opened(p)
        with mock.patch.object(c, 'physical_fd', side_effect=change_on_reopen):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                c.verified_bytes(path, pin)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), pin)

    def test_legacy_read_mutation_restored_before_recheck_is_rejected(self):
        path = self.root / 'asset'
        original = b'known-full-asset-content'
        path.write_bytes(original)
        proofs = c.AssetProofs(shared)
        proofs.add('original', path, self.root, hashlib.sha256(original).hexdigest(), len(original))
        path.write_bytes(b'bad-full-asset-content!!')
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'changed after streamed SHA'):
            proofs.recheck()

    def test_actual_pinned_manifests_make_exact_37_unique_physical_paths(self):
        snap = W / 'actual-snapshots/3261-20261006T005604410916Z/original-files'
        pins = {'model_manifest.json':'7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44',
                'data_manifest.json':'0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
                'tool_manifest.json':'87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87'}
        objects = {}
        for name, pin in pins.items():
            raw = (snap / 'bindings' / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin)
            objects[name] = json.loads(raw)
        raw = (snap / 'full_state/global_step_5/resolved-config.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), shared.CONFIG_SHA)
        config = json.loads(raw)
        records = []
        recorder = types.SimpleNamespace(add=lambda *args: records.append(args))
        c.asset_inventory(recorder, config, objects['model_manifest.json'], objects['data_manifest.json'], objects['tool_manifest.json'])
        self.assertEqual(len(records), 37)
        self.assertEqual(len({str(x[1]) for x in records}), 37)
        self.assertEqual(len({x[0] for x in records}), 37)
        self.assertEqual(len({x[3] for x in records}), 37)
        self.assertEqual(sum(x[0].startswith('official/') for x in records), 11)
        self.assertEqual(sum(x[0].startswith('tool/') for x in records), 11)

    def test_import_record_pyc_cannot_execute(self):
        path = self.root / 'boundary_exact.pyc'
        path.write_bytes(b'not an allowed source')
        finder = c.ExactSourceFinder({'boundary_exact': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}, prefixes=('boundary_exact',))
        finder.install()
        try:
            with self.assertRaisesRegex(ValueError, 'Python source'):
                importlib.import_module('boundary_exact')
        finally:
            finder.remove()
        self.assertNotIn('boundary_exact', finder.executed)

    def test_cpu_new_identity_four_fields_reject_duplicate_missing_uid(self):
        self.fake_self_proc('0::/system.slice/ssh.service\n')
        uid = str(os.getuid())
        (self.root / 'self/status').write_text('Uid:\t' + '\t'.join([uid] * 5) + '\n')
        with self.assertRaisesRegex(ValueError, 'UID fields differ'):
            c.execution_identity(self.root, {})

    def test_namespace_alias_to_foreign_initializer_is_rejected(self):
        foreign = self.root / 'foreign'
        foreign.mkdir()
        (foreign / '__init__.py').write_text('UNAPPROVED=True\n')
        alias = self.root / 'namespace'
        alias.symlink_to(foreign, target_is_directory=True)
        with self.assertRaises(OSError):
            c.physical_namespace(alias)


if __name__ == '__main__':
    unittest.main(verbosity=2)
