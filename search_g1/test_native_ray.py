import os
import shutil
import unittest
import uuid
from unittest.mock import patch
from reproduction.common8.native_ray import prepare, root_for, verify


class NativeRayRootTest(unittest.TestCase):
    def setUp(self):
        # macOS aliases /tmp to /private/tmp; use its canonical parent locally.
        original = root_for
        self.root_patch = patch('reproduction.common8.native_ray.root_for',
                                side_effect=lambda run_id: original(run_id).parent.resolve() / original(run_id).name)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_bound_short_root_and_exclusive_creation(self):
        run = {'run_id': uuid.uuid4().hex}
        path = root_for(run['run_id']).parent.resolve() / root_for(run['run_id']).name
        try:
            prepare(run)
            self.assertEqual(verify(run, str(path)), str(path))
            with self.assertRaises(FileExistsError):
                prepare(dict(run_id=run['run_id']))
            with self.assertRaises(AssertionError):
                verify(run, str(path / 'other'))
            os.chmod(path, 0o755)
            with self.assertRaises(AssertionError):
                verify(run, str(path))
        finally:
            shutil.rmtree(path)

    def test_symlink_replacement_rejected(self):
        run = {'run_id': uuid.uuid4().hex}
        path = root_for(run['run_id']).parent.resolve() / root_for(run['run_id']).name
        other = path.with_name(path.name + '-replacement')
        try:
            prepare(run)
            path.rename(other)
            path.symlink_to(other, target_is_directory=True)
            with self.assertRaises(AssertionError):
                verify(run, str(path))
        finally:
            if path.is_symlink(): path.unlink()
            if other.exists(): shutil.rmtree(other)
