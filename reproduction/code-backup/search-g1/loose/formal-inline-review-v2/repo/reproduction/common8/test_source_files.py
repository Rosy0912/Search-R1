import tempfile
import unittest
from pathlib import Path
from reproduction.common8.source_files import VERSION_FILES, runtime_files


class RuntimePackageTests(unittest.TestCase):
    def test_extensionless_import_dependencies_are_bound(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            for name in VERSION_FILES + ('verl/__init__.py', 'reproduction/evidence/fixture.json', 'notes.md'):
                p = repo / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture')
            self.assertEqual({str(p.relative_to(repo)) for p in runtime_files(repo)},
                             set(VERSION_FILES) | {'verl/__init__.py'})

    def test_incomplete_package_is_rejected_before_submission(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(AssertionError, 'Incomplete runtime package'):
                runtime_files(root)


if __name__ == '__main__': unittest.main()
