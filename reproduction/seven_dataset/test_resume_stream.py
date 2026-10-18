import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from .resume_stream import inspect_prefix, receive_suffix, send_suffix


class ResumeStreamTests(unittest.TestCase):
    def test_exact_resume_and_source_offset(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'source', Path(d)/'target'
            data = bytes(range(256))*8192
            a.write_bytes(data); b.write_bytes(data[:12345])
            source = inspect_prefix(a, 12345); target = inspect_prefix(b, 12345)
            self.assertEqual(source['prefix_sha256'], target['prefix_sha256'])
            suffix = io.BytesIO(); send_suffix(a, source, suffix); suffix.seek(0)
            result = receive_suffix(b, target, len(data), hashlib.sha256(data).hexdigest(), suffix)
            self.assertEqual(b.read_bytes(), data)
            self.assertEqual(result['status'], 'ACTUAL_FULL_SHA_VERIFIED')

    def test_prefix_mismatch_does_not_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'target'; p.write_bytes(b'abc'); e = inspect_prefix(p, 3)
            e['prefix_sha256'] = '0'*64
            with self.assertRaises(ValueError):receive_suffix(p,e,6,'',io.BytesIO(b'def'))
            self.assertEqual(p.read_bytes(), b'abc')

    def test_short_suffix_preserved_and_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'target'; p.write_bytes(b'abc'); h=hashlib.sha256(b'abcdef').hexdigest()
            with self.assertRaises(ValueError):receive_suffix(p,inspect_prefix(p,3),6,h,io.BytesIO(b'd'))
            self.assertEqual(p.read_bytes(), b'abcd')
            receive_suffix(p,inspect_prefix(p,4),6,h,io.BytesIO(b'ef'))
            self.assertEqual(p.read_bytes(), b'abcdef')

    def test_replaced_inode_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'target'; p.write_bytes(b'abc'); e=inspect_prefix(p,3)
            p.rename(Path(d)/'original'); p.write_bytes(b'abc')
            with self.assertRaises(ValueError):receive_suffix(p,e,6,'',io.BytesIO(b'def'))
            p.unlink(); p.symlink_to(Path(d)/'original')
            with self.assertRaises(OSError):receive_suffix(p,e,6,'',io.BytesIO(b'def'))

    def test_excess_and_bad_full_sha_preserved(self):
        for suffix in (b'def', b'defextra'):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'target'; p.write_bytes(b'abc')
                with self.assertRaises(ValueError):receive_suffix(p,inspect_prefix(p,3),6,'0'*64,io.BytesIO(suffix))
                self.assertEqual(p.read_bytes(), b'abcdef')

    def test_changed_source_prefix_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source'; p.write_bytes(b'abcdef'); e=inspect_prefix(p,3)
            p.write_bytes(b'XYZdef'); out=io.BytesIO()
            with self.assertRaises(ValueError):send_suffix(p,e,out)
            self.assertEqual(out.getvalue(), b'')


if __name__ == '__main__':unittest.main()
