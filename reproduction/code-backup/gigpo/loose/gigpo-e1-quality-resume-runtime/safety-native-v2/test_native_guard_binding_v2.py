"""No GPU: require exact local versioned paths and complete helper SHA entries."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from native_guard_binding_v2 import validate_guard_helper_binding


def require(condition, message):
    if not condition: raise RuntimeError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeGuardBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.helpers = Path(self.temp.name) / 'policy-safety-native-52befe9-v2'
        self.helpers.mkdir()
        names = ('native_run_safety_v1.py', 'native_guard_binding_v2.py',
                 'resource_guard_v2.py', 'resource_ownership_v2.py',
                 'test_resource_ownership_v2.py', 'test_native_guard_binding_v2.py')
        for name in names: (self.helpers / name).write_text(name + '\n')
        self.value = dict(helpers=str(self.helpers), files=[dict(path=str(p), sha256=sha(p)) for p in self.helpers.iterdir()])
        for key, name in (('resource_guard', 'resource_guard_v2.py'), ('resource_ownership', 'resource_ownership_v2.py')):
            self.value[key] = dict(path=str(self.helpers / name), sha256=sha(self.helpers / name))
    def tearDown(self): self.temp.cleanup()
    def call(self, value): return validate_guard_helper_binding(self.helpers, value, require, sha)
    def test_exact_bound_isolated_guard_returned(self):
        expected={key:dict(path=str(Path(item['path']).resolve()),sha256=item['sha256']) for key,item in self.value.items() if key in ('resource_guard','resource_ownership')}
        self.assertEqual(self.call(self.value), expected)
    def test_old_helper_directory_refused(self):
        value = copy.deepcopy(self.value);value['helpers'] = str(self.helpers.parent / 'v1')
        with self.assertRaisesRegex(RuntimeError, 'directory differs'): self.call(value)
    def test_missing_each_executable_module_or_test_refused(self):
        for item in self.value['files']:
            value = copy.deepcopy(self.value);value['files'].remove(item)
            with self.assertRaisesRegex(RuntimeError, 'missing or duplicate'): self.call(value)
    def test_wrong_hash_refused(self):
        value = copy.deepcopy(self.value);value['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'): self.call(value)
    def test_duplicate_entries_refused(self):
        value = copy.deepcopy(self.value);value['files'].append(value['files'][0].copy())
        with self.assertRaisesRegex(RuntimeError, 'missing or duplicate'): self.call(value)
    def test_foreign_path_same_basename_refused(self):
        value = copy.deepcopy(self.value);value['files'][0]['path'] = str(self.helpers.parent / Path(value['files'][0]['path']).name)
        with self.assertRaisesRegex(RuntimeError, 'missing or duplicate'): self.call(value)
    def test_missing_explicit_guard_or_module_refused(self):
        for key in ('resource_guard', 'resource_ownership'):
            value = copy.deepcopy(self.value);del value[key]
            with self.assertRaisesRegex(RuntimeError, 'identity missing'): self.call(value)
    def test_explicit_guard_or_module_wrong_path_refused(self):
        for key in ('resource_guard', 'resource_ownership'):
            value = copy.deepcopy(self.value);value[key]['path']=str(self.helpers.parent / Path(value[key]['path']).name)
            with self.assertRaisesRegex(RuntimeError, 'path mismatch'): self.call(value)
    def test_explicit_guard_or_module_wrong_sha_refused(self):
        for key in ('resource_guard', 'resource_ownership'):
            value = copy.deepcopy(self.value);value[key]['sha256']='0'*64
            with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'): self.call(value)


if __name__ == '__main__': unittest.main(verbosity=2)
