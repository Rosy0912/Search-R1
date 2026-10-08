import gzip, hashlib, importlib.util, io, json, pathlib, tempfile, unittest
spec=importlib.util.spec_from_file_location('v',pathlib.Path(__file__).with_name('tool_assets_full_verify.py')); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
class Test(unittest.TestCase):
    def test_segments_boundary(self):
        old=v.CHUNK; v.CHUNK=7
        try:
            r=v.hash_stream(io.BytesIO(b'abcdefghijklmno'),[8,7]); self.assertEqual(r['segment_sha256'],[hashlib.sha256(b'abcdefgh').hexdigest(),hashlib.sha256(b'ijklmno').hexdigest()]); self.assertEqual(r['sha256'],hashlib.sha256(b'abcdefghijklmno').hexdigest())
        finally: v.CHUNK=old
    def test_bad_lengths(self):
        for b in (b'1234',b'123456'):
            with self.assertRaises(ValueError): v.hash_stream(io.BytesIO(b),[2,3])
    def test_stable_mutation_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d).resolve()/'f'; p.write_bytes(b'one'); f,s=v.stable_open(p); p.write_bytes(b'two')
            try:
                with self.assertRaises(ValueError): v.finish(f,p,s)
            finally: f.close()
            link=p.with_name('alias'); link.symlink_to(p)
            with self.assertRaises(ValueError): v.hash_file(link)
    def test_gzip_crc(self):
        b=bytearray(gzip.compress(b'hello'*99)); b[-8]^=1
        with self.assertRaises(gzip.BadGzipFile): v.hash_stream(gzip.GzipFile(fileobj=io.BytesIO(b)))
    def test_atomic_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'manifest.json'; v.atomic_json(p,{'x':1})
            with self.assertRaises(FileExistsError): v.atomic_json(p,{'x':2})
            self.assertEqual(json.loads(p.read_text()),{'x':1})
if __name__=='__main__': unittest.main()
