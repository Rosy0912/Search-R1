"""Focused CPU fixtures; small dictionary, no real CP/Node/transfer/migration."""
import hashlib
import io
import lzma
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest

HERE = Path(__file__).absolute().parent
c = types.ModuleType('compact_fixture'); c.__file__ = str(HERE / 'compact.py')
exec(compile((HERE / 'compact.py').read_bytes(), c.__file__, 'exec'), c.__dict__)
V2 = HERE.parent / 'opsd_ablation_outer_backup_v2'
b, q = c.legacy_modules(V2, '4f9ee158ec1c326498898d26ee5b6a1d3212e94d5ecbbd24c8ef7864fa079b69')


class Compact(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()))
        self.root = Path(self.temp.name)
        self.original_dict = c.DICT_BYTES; c.DICT_BYTES = 1 << 20
        files = []
        for name, data in (('group/rank.pt', b'finite-moment-fixture'*1024), ('closure/config.json', b'{"same":true}')):
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            files.append(dict(relative=name, **b.fileproof(path)))
        self.inventory = dict(files=files, total_bytes=sum(x['bytes'] for x in files))
        output = io.BytesIO(); b.stream_archive(self.inventory, output); self.tar = output.getvalue()
        self.raw = dict(bytes=len(self.tar), sha256=hashlib.sha256(self.tar).hexdigest())

    def tearDown(self):
        c.DICT_BYTES = self.original_dict; self.temp.cleanup()

    def packed(self, raw=None):
        path = self.root / 'fixture.xz'
        with path.open('xb') as destination:
            result = c.compress_stream(b, io.BytesIO(self.tar if raw is None else raw), destination, self.inventory)
        return path, result

    def test_original_v2_ustar_lossless_complete(self):
        path, raw = self.packed()
        proof = c.verify_xz(b, path, self.inventory, raw)
        self.assertTrue(proof['accepted']); self.assertEqual(proof['member_count'], 2)
        self.assertEqual(raw['sha256'], self.raw['sha256'])
        self.assertEqual(raw['bytes'], self.raw['bytes'])
        self.assertEqual(lzma.decompress(path.read_bytes()), self.tar)
        self.assertFalse((self.root / 'verified-files').exists())

    def test_truncated_partial_not_accepted_or_removed(self):
        path, raw = self.packed(); path.write_bytes(path.read_bytes()[:-12])
        with self.assertRaises((RuntimeError, b.Rejected, lzma.LZMAError)):
            c.verify_xz(b, path, self.inventory, raw)
        self.assertTrue(path.exists())

    def test_member_identity_and_exact_whole_tar_required(self):
        path, raw = self.packed()
        wrong = dict(self.inventory, files=[dict(r) for r in self.inventory['files']])
        wrong['files'][0]['sha256'] = '0'*64
        with self.assertRaises(b.Rejected): c.verify_xz(b, path, wrong, raw)
        with self.assertRaises(b.Rejected): c.verify_xz(b, path, self.inventory, dict(raw, sha256='0'*64))

    def test_special_or_escape_members_refused(self):
        for name, kind in (('group/rank.pt', tarfile.SYMTYPE), ('../escaped', tarfile.REGTYPE)):
            with self.subTest(name=name):
                target = io.BytesIO()
                with tarfile.open(fileobj=target, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                    entry = tarfile.TarInfo(name); entry.type = kind; entry.size = 0
                    if kind == tarfile.SYMTYPE: entry.linkname = '/tmp/escaped'
                    archive.addfile(entry)
                path = self.root / 'fixture.xz'
                if path.exists(): path.unlink()
                path, raw = self.packed(target.getvalue())
                with self.assertRaises(b.Rejected): c.verify_xz(b, path, self.inventory, raw)

    def test_concatenated_xz_refused(self):
        path, raw = self.packed(); path.write_bytes(path.read_bytes()+lzma.compress(b''))
        with self.assertRaises(RuntimeError): c.verify_xz(b, path, self.inventory, raw)

    def test_does_not_acquire_or_modify_live_old_queue_lock(self):
        lock = self.root / '.backup-queue.lock'; lock.write_bytes(b'original')
        before = lock.read_bytes()
        command = 'import fcntl,sys;f=open(sys.argv[1],"r");fcntl.flock(f,fcntl.LOCK_EX);print("LOCKED",flush=True);sys.stdin.readline()'
        child = subprocess.Popen([sys.executable, '-B', '-c', command, str(lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            self.assertEqual(child.stdout.readline(), b'LOCKED\n')
            with self.assertRaises(BlockingIOError):
                with c.QueueLease(b, self.root): self.fail('old live lease must reject')
            self.assertEqual(lock.read_bytes(), before)
        finally:
            child.stdin.write(b'exit\n'); child.stdin.close(); child.stdout.close()
            self.assertEqual(child.wait(), 0)
        with c.QueueLease(b, self.root): self.assertEqual(lock.read_bytes(), before)


if __name__ == '__main__': unittest.main()
