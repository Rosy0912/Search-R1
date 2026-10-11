"""Small CPU fixtures only; no model, real checkpoint or GPU admission claim."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.common8 import independent_backup as backup


class IndependentBackupTests(unittest.TestCase):
    def test_actual_source_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.txt'
            source.write_bytes(b'original')
            original_open = Path.open

            class MutatingReader:
                def __init__(self, handle):
                    self.handle = handle
                    self.changed = False

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.handle.close()

                def read(self, count):
                    data = self.handle.read(count)
                    if data and not self.changed:
                        self.changed = True
                        with original_open(source, 'ab') as writer:
                            writer.write(b'changed')
                    return data

            def opening(path, mode='r', *args, **kwargs):
                handle = original_open(path, mode, *args, **kwargs)
                return MutatingReader(handle) if path == source and mode == 'rb' else handle

            with patch.object(Path, 'open', opening):
                with self.assertRaisesRegex(ValueError, 'File changed'):
                    backup.paced_read(source, rate=1024**2)

    def fixture(self, folder, wrong_hash=False):
        source = Path(folder) / 'source'
        root = Path(folder) / 'destination'
        source.mkdir()
        root.mkdir()
        raw = b'{"fixture_only": true}\n'
        (source / 'complete.json').write_bytes(raw)
        (source / 'nested').mkdir()
        payload = b'cpu fixture data' * 1024
        (source / 'nested/payload.txt').write_bytes(payload)
        files = [dict(relative_path='complete.json', bytes=len(raw), expected_sha256=hashlib.sha256(raw).hexdigest()),
                 dict(relative_path='nested/payload.txt', bytes=len(payload),
                      expected_sha256=('0' * 64 if wrong_hash else hashlib.sha256(payload).hexdigest()))]
        return source, root, files, sum(row['bytes'] for row in files), 1024**4, hashlib.sha256(raw).hexdigest()

    def test_full_target_readback_and_atomic_final(self):
        with tempfile.TemporaryDirectory() as folder:
            source, root, files, need, free, sha = self.fixture(folder)
            with patch.object(backup, 'admit', return_value=(source, files, need, free, sha)), patch.object(backup.os, 'nice'), patch.object(backup, 'RESERVE', 0):
                backup.backup(5, root)
            result = json.loads((root / 'RESULT.json').read_text())
            self.assertEqual(result['status'], 'INDEPENDENT_CHECKPOINT_FULL_SHA_PASS')
            self.assertFalse(result['semantic_reload_performed'])
            self.assertFalse((root / 'checkpoint.partial').exists())
            for row in result['files']:
                target = root / 'checkpoint' / row['relative_path']
                self.assertEqual(target.read_bytes(), (source / row['relative_path']).read_bytes())
                self.assertEqual(row['source_full_read_sha256'], row['destination_full_read_sha256'])

    def test_hash_failure_preserves_source_and_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            source, root, files, need, free, sha = self.fixture(folder, wrong_hash=True)
            original = (source / 'nested/payload.txt').read_bytes()
            with patch.object(backup, 'admit', return_value=(source, files, need, free, sha)), patch.object(backup.os, 'nice'), patch.object(backup, 'RESERVE', 0):
                with self.assertRaisesRegex(ValueError, 'Source full read differs'):
                    backup.backup(5, root)
            self.assertEqual((source / 'nested/payload.txt').read_bytes(), original)
            self.assertTrue((root / 'checkpoint.partial').is_dir())
            self.assertFalse((root / 'checkpoint').exists())
            result = json.loads((root / 'RESULT.json').read_text())
            self.assertEqual(result['status'], 'FAILED_PRESERVE_SOURCE_AND_PARTIAL')


if __name__ == '__main__':
    unittest.main()
