import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('recompress', Path(__file__).with_name('recompress.py'))
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class ExactBytes(unittest.TestCase):
    def test_roundtrip_and_truncation_preserve_original(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve(); source = root/'tiny-gnu.tar'
            original = b'GNU header and padding\0' * 8192 + bytes(range(256))
            source.write_bytes(original); sha = hashlib.sha256(original).hexdigest()
            compressed = root/'tiny.xz'; r.compress_exact(source, compressed, sha, len(original))
            proof = r.verify_disk_xz(compressed, sha, len(original))
            self.assertEqual(proof['decoded_sha256'], sha)
            self.assertEqual(proof['decoded_bytes'], len(original))
            self.assertTrue(proof['single_stream'])
            bad = root/'truncated.partial'; bad.write_bytes(compressed.read_bytes()[:-8])
            with self.assertRaises((ValueError, r.lzma.LZMAError)):
                r.verify_disk_xz(bad, sha, len(original))
            self.assertTrue(bad.exists())
            self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
