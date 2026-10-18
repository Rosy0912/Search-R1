import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from .parallel_archive import manifest, send, receive, combine
from .resume_stream import inspect_prefix


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.data = bytes(range(251))*701
        self.source = self.root/'source'; self.source.write_bytes(self.data)
        self.sha = hashlib.sha256(self.data).hexdigest()

    def test_exact_append_and_ranges(self):
        target = self.root/'target'; target.write_bytes(self.data[:12345])
        prefix = inspect_prefix(target, 12345)
        proof = manifest(self.source, 12345, len(self.data), self.sha)
        paths = []
        for item in proof['ranges']:
            b = io.BytesIO(); send(self.source, proof['source_fingerprint'], item, b)
            b.seek(0); part = self.root/str(item['index']); receive(part, item, b)
            paths.append(str(part))
        self.assertEqual(combine(target, prefix, proof, paths)['sha256'], self.sha)
        self.assertEqual(target.read_bytes(), self.data)

    def test_complete_four_ranges_from_exclusive_empty_target(self):
        target=self.root/'empty-target';target.touch(exist_ok=False)
        prefix=inspect_prefix(target,0);proof=manifest(self.source,0,len(self.data),self.sha)
        paths=[]
        for item in proof['ranges']:
            stream=io.BytesIO();send(self.source,proof['source_fingerprint'],item,stream);stream.seek(0)
            part=self.root/('zero-'+str(item['index']));receive(part,item,stream);paths.append(str(part))
        result=combine(target,prefix,proof,paths)
        self.assertEqual(result['sha256'],self.sha)
        self.assertEqual(target.read_bytes(),self.data)

    def test_incomplete_preserves(self):
        proof = manifest(self.source, 0, len(self.data), self.sha)
        item = proof['ranges'][0]; path = self.root/'part'
        with self.assertRaises(ValueError): receive(path, item, io.BytesIO(b'abc'))
        self.assertEqual(path.read_bytes(), b'abc')
        self.assertFalse(Path(str(path)+'.verified.json').exists())

    def test_no_replace(self):
        p = self.root/'part'; p.write_bytes(b'keep')
        proof = manifest(self.source, 0, len(self.data), self.sha)
        with self.assertRaises(FileExistsError): receive(p, proof['ranges'][0], io.BytesIO())
        self.assertEqual(p.read_bytes(), b'keep')

    def test_corrupt_range(self):
        proof = manifest(self.source, 0, len(self.data), self.sha); item = proof['ranges'][0]
        p = self.root/'part'
        with self.assertRaises(AssertionError): receive(p, item, io.BytesIO(b'x'*item['length']))
        self.assertFalse(Path(str(p)+'.verified.json').exists())

    def test_source_change_rejected(self):
        proof = manifest(self.source, 0, len(self.data), self.sha)
        self.source.write_bytes(b'changed')
        with self.assertRaises(AssertionError): send(self.source, proof['source_fingerprint'], proof['ranges'][0], io.BytesIO())

    def test_wrong_full_sha(self):
        with self.assertRaises(AssertionError): manifest(self.source, 0, len(self.data), '0'*64)


if __name__ == '__main__': unittest.main()
