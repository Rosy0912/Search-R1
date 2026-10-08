"""CPU-only fixtures. Small synthetic bytes; no node/SSH/model/tensor access."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pinned_backup', HERE / 'backup.py')
b = importlib.util.module_from_spec(spec)
# Compile current verified source bytes directly; no cached experimental code.
exec(compile((HERE / 'backup.py').read_bytes(), str(HERE / 'backup.py'), 'exec'), b.__dict__)


class Fixture:
    def __init__(self, root):
        self.base = root / 'node'; self.base.mkdir()
        self.output = self.base / 'runs'; self.output.mkdir()
        self.cpdir = self.output / 'checkpoints/update-00000901-outer-00199'
        self.cpdir.mkdir(parents=True)
        self.outer = self.output / 'outers/outer-00199'; self.outer.mkdir(parents=True)
        self.configpath = self.base / 'receipts/config.json'; self.configpath.parent.mkdir()
        refs = {}
        for name in ('runtime_manifest', 'source_model_manifest', 'helper_source_manifest'):
            refs[name] = self.save(self.base / 'receipts' / (name + '.json'), {'files': {'source.py': 'bound'}})
        cfg = dict(run_id=b.RUN, paths=dict(base=str(self.base), output_root=str(self.output)), **refs)
        self.cref = self.save(self.configpath, cfg)
        self.rip = self.save(self.output / 'run-identity.json', dict(schema='opsd-online-run-identity-v1',
                             run_id=b.RUN, config_sha256=self.cref['sha256'], config=self.cref))
        pairs = self.save(self.outer / 'sampling-attempt-00001/pairs.jsonl', b'private-fixture\n')
        encoding = self.save(self.outer / 'training/encoding.json', {'rows': []})
        order = self.save(self.outer / 'training/order.json', {'order': []})
        outer = self.save(self.outer / 'outer-manifest.json', dict(run_id=b.RUN,
                         config_sha256=self.cref['sha256'], outer_index=199, pairs=pairs,
                         runtime_identity=refs['runtime_manifest']['sha256'],
                         source_model_identity=refs['source_model_manifest']['sha256']))
        ranks = [dict(rank=i, file=self.save(self.cpdir / ('rank-%d-state.pt' % i), ('state-%d' % i).encode())) for i in range(8)]
        adapter = [self.save(self.cpdir / 'adapter' / name, ('adapter-' + name).encode())
                   for name in ('README.md', 'adapter_config.json', 'adapter_model.safetensors')]
        self.cp = dict(schema='opsd-online-checkpoint-v1', accepted=True, run_id=b.RUN,
                       config_sha256=self.cref['sha256'], optimizer_updates=901,
                       runtime_identity=refs['runtime_manifest']['sha256'],
                       source_model_identity=refs['source_model_manifest']['sha256'],
                       cursor=dict(outer_index=199, usable_pair_count=0, next_pair_cursor=0),
                       protection_reason=['outer_end'], complete_rank_count=8, equal_optimizer_steps=True,
                       ranks=ranks, adapter=dict(path=str(self.cpdir / 'adapter'), files=adapter),
                       training_input=dict(outer_manifest=outer, pairs=pairs, encoding=encoding, order=order))
        self.cpref = self.save(self.cpdir / 'manifest.json', self.cp)

    @staticmethod
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else b.canonical(value) + b'\n')
        return b.fileproof(path)

    def args(self):
        return (str(self.configpath), self.cref['sha256'], str(self.cpdir / 'manifest.json'), self.cpref['sha256'], b.RUN, 901, 199)

    def inv(self):
        return b.inventory(*self.args())

    def rewrite_cp(self):
        self.cpref = self.save(self.cpdir / 'manifest.json', self.cp)


class TestInventory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.f = Fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_real_producer_shape_eighteen_files(self):
        v = self.f.inv()
        self.assertEqual(len(v['files']), 18)
        self.assertEqual(len(b.validate_inventory(v, *self.f.args())), 18)
        self.assertEqual(len(v['scientific_identity_manifests']), 3)
        self.assertIn('no model/runtime', v['scientific_scope'])

    def test_wrong_run_updates_outer_and_config_hash_reject(self):
        for index, value in ((1, '0' * 64), (4, 'wrong'), (5, 902), (6, 198)):
            args = list(self.f.args()); args[index] = value
            with self.subTest(index=index), self.assertRaises((b.Rejected, FileNotFoundError)):
                b.inventory(*args)

    def test_cp89_exact_pin_no_old_checkpoints_or_latest(self):
        b.target_scope(b.RUN, 89, 8, b.CP89_SHA)
        for update, outer, sha in ((89, 8, '0' * 64), (0, 0, '0' * 64), (1, 0, '0' * 64),
                                   (5, 0, '0' * 64), (11, 0, '0' * 64), (902, 9, '0' * 64)):
            with self.subTest(update=update), self.assertRaises(b.Rejected):
                b.target_scope(b.RUN, update, outer, sha)

    def test_missing_rank_duplicate_rank_and_wrong_adapter_inventory(self):
        for mutate in (lambda: self.f.cp['ranks'].pop(),
                       lambda: self.f.cp['ranks'][7].update(rank=0),
                       lambda: self.f.cp['adapter']['files'].pop()):
            original = json.loads(json.dumps(self.f.cp)); mutate(); self.f.rewrite_cp()
            with self.assertRaises(b.Rejected): self.f.inv()
            self.f.cp = original; self.f.rewrite_cp()

    def test_extra_checkpoint_file_rejected(self):
        (self.f.cpdir / 'extra').write_bytes(b'extra')
        with self.assertRaises(b.Rejected): self.f.inv()

    def test_source_bytes_sha_and_stat_declared_must_match(self):
        row = self.f.cp['ranks'][0]['file']; Path(row['path']).write_bytes(b'changed')
        with self.assertRaises(b.Rejected): self.f.inv()

    def test_missing_declared_metadata_rejected(self):
        self.f.cp['ranks'][0]['file'].pop('bytes'); self.f.rewrite_cp()
        with self.assertRaises(b.Rejected): self.f.inv()

    def test_scientific_identity_manifest_mutation_rejected(self):
        cfg = json.loads(self.f.configpath.read_text())
        Path(cfg['runtime_manifest']['path']).write_bytes(b'{}')
        with self.assertRaises(b.Rejected): self.f.inv()

    def test_symlink_file_and_ancestor_rejected(self):
        p = self.root / 'real'; p.mkdir(); (p / 'x').write_bytes(b'x')
        (self.root / 'link').symlink_to(p, target_is_directory=True)
        (p / 'linked').symlink_to(p / 'x')
        for name in (self.root / 'link/x', p / 'linked'):
            with self.subTest(name=name), self.assertRaises(b.Rejected): b.fileproof(name)

    def test_training_input_escape_rejected(self):
        ref = self.f.save(self.root / 'outside.json', {})
        self.f.cp['training_input']['encoding'] = ref; self.f.rewrite_cp()
        with self.assertRaises(b.Rejected): self.f.inv()

    def test_path_escape_and_alias_inventory_rejected(self):
        v = self.f.inv(); v['files'][0]['relative'] = '../escape'
        with self.assertRaises(b.Rejected): b.validate_inventory(v, *self.f.args())
        v = self.f.inv(); v['files'][0]['inode'] = v['files'][1]['inode']; v['files'][0]['device'] = v['files'][1]['device']
        with self.assertRaises(b.Rejected): b.validate_inventory(v, *self.f.args())

    def test_stream_fixed_inventory_sha_and_roundtrip_readback(self):
        v = self.f.inv(); stream = io.BytesIO(); b.stream_archive(v, stream)
        archive = self.root / 'copy.tar'; archive.write_bytes(stream.getvalue())
        expected = b.validate_inventory(v, *self.f.args())
        files, _ = b.extract_verified(archive, self.root / 'verified', expected)
        self.assertEqual(len(files), 18)
        self.assertTrue(all(x['sha256'] == expected[name]['sha256'] for name, x in files.items()))

    def test_post_inventory_mutation_refused_before_stream(self):
        v = self.f.inv(); Path(v['files'][0]['path']).write_bytes(b'changed')
        with self.assertRaises(b.Rejected): b.stream_archive(v, io.BytesIO())

    def test_stream_hash_detects_mutation_even_if_stat_check_mocked(self):
        v = self.f.inv(); Path(v['files'][0]['path']).write_bytes(b'x' * v['files'][0]['bytes'])
        with patch.object(b, 'stable'), patch.object(b, 'meta_from_stat', return_value={k: v['files'][0][k] for k in b.META}):
            with self.assertRaises(b.Rejected): b.stream_archive(dict(v, files=[v['files'][0]]), io.BytesIO())


class TestArchive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
        self.expected = {'runs/file': dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())}

    def tearDown(self): self.tmp.cleanup()

    def tar(self, entries, suffix=b'', format=tarfile.USTAR_FORMAT):
        path = self.root / ('archive-' + str(len(list(self.root.iterdir()))) + '.tar')
        with tarfile.open(path, 'w', format=format) as tf:
            for name, content, kind in entries:
                m = tarfile.TarInfo(name); m.type = kind
                if kind in (tarfile.REGTYPE, tarfile.AREGTYPE): m.size = len(content)
                else: m.linkname = 'runs/file'
                tf.addfile(m, io.BytesIO(content) if m.isfile() else None)
        if suffix:
            with path.open('ab') as handle: handle.write(suffix)
        return path

    def reject(self, entries, suffix=b'', format=tarfile.USTAR_FORMAT):
        path = self.tar(entries, suffix, format)
        with self.assertRaises((b.Rejected, tarfile.TarError)):
            b.extract_verified(path, self.root / ('out-' + path.stem), self.expected)

    def test_duplicate_missing_extra_and_wrong_bytes(self):
        good = ('runs/file', b'abc', tarfile.REGTYPE)
        for rows in ([good, good], [], [good, ('extra', b'a', tarfile.REGTYPE)],
                     [('runs/file', b'xyz', tarfile.REGTYPE)], [('runs/file', b'four', tarfile.REGTYPE)]):
            with self.subTest(rows=rows): self.reject(rows)

    def test_absolute_dot_escape_backslash_and_special_types(self):
        for name in ('/runs/file', '../runs/file', './runs/file', 'runs//file', 'runs\\file'):
            with self.subTest(name=name): self.reject([(name, b'abc', tarfile.REGTYPE)])
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE, tarfile.CHRTYPE):
            with self.subTest(kind=kind): self.reject([('runs/file', b'', kind)])

    def test_pax_headers_refused(self):
        path = self.root / 'pax.tar'
        with tarfile.open(path, 'w', format=tarfile.PAX_FORMAT) as tf:
            m = tarfile.TarInfo('runs/file'); m.size = 3; m.pax_headers = {'comment': 'extra'}
            tf.addfile(m, io.BytesIO(b'abc'))
        with self.assertRaises(b.Rejected): b.extract_verified(path, self.root / 'out', self.expected)

    def test_hidden_gnu_special_header_refused(self):
        path = self.root / 'gnu.tar'
        with tarfile.open(path, 'w', format=tarfile.GNU_FORMAT) as tf:
            data = b'runs/file\0'
            extension = tarfile.TarInfo('././@LongLink'); extension.type = tarfile.GNUTYPE_LONGNAME; extension.size = len(data)
            tf.addfile(extension, io.BytesIO(data))
            member = tarfile.TarInfo('ignored'); member.size = 3
            tf.addfile(member, io.BytesIO(b'abc'))
        with self.assertRaises(b.Rejected): b.extract_verified(path, self.root / 'out', self.expected)

    def test_trailing_second_archive_and_truncated_end(self):
        good = [('runs/file', b'abc', tarfile.REGTYPE)]
        second = self.tar(good).read_bytes()
        self.reject(good, suffix=second)
        path = self.tar(good); path.write_bytes(path.read_bytes()[:1024])
        with self.assertRaises(b.Rejected): b.extract_verified(path, self.root / 'truncated', self.expected)

    def test_independent_local_readback_corruption_refused(self):
        path = self.tar([('runs/file', b'abc', tarfile.REGTYPE)])
        out = self.root / 'out'; b.extract_verified(path, out, self.expected)
        (out / 'runs/file').write_bytes(b'xyz')
        with self.assertRaises(b.Rejected): b.readback(out, self.expected)

    def test_local_symlink_or_extra_after_extraction_refused(self):
        for symlink in (False, True):
            path = self.tar([('runs/file', b'abc', tarfile.REGTYPE)])
            out = self.root / ('out-' + str(symlink)); b.extract_verified(path, out, self.expected)
            if symlink:
                (out / 'runs/file').unlink(); (out / 'runs/file').symlink_to(path)
            else: (out / 'extra').write_bytes(b'no')
            with self.assertRaises(b.Rejected): b.readback(out, self.expected)

    def test_space_archive_plus_extraction_and_margin(self):
        v = dict(files=[dict(bytes=3)], total_bytes=3)
        self.assertGreater(b.required_space(v, 5 << 30), (5 << 30) + 6)
        with self.assertRaises(b.Rejected): b.required_space(v, (5 << 30) - 1)


class TestPublicationAndExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()

    def tearDown(self): self.tmp.cleanup()

    def publication(self):
        root = self.root / 'helper'; root.mkdir()
        files = {}
        for name in ('backup.py', 'test_backup.py', 'README.md'):
            path = root / name; path.write_bytes(('source-' + name).encode())
            fp = b.fileproof(path); files[name] = {k: fp[k] for k in ('bytes', 'sha256')}
        mp = Fixture.save(root / 'source-manifest.json', dict(files=files))
        commit = 'a' * 40
        pub = dict(accepted=True, branch='search-e1', run_id=b.RUN, source_commit=commit,
                   remote_readback_sha=commit, source_manifest_sha256=mp['sha256'], files=[])
        for name in [*files, 'source-manifest.json']:
            fp = b.fileproof(root / name)
            pub['files'].append(dict(path=b.REPO_PREFIX + name, equal=True,
                                     bytes=fp['bytes'], sha256=fp['sha256']))
        pp = Fixture.save(self.root / 'publication.json', pub)
        return root, mp, pub, pp, commit

    def test_publication_current_source_hash_and_remote_commit_bound(self):
        root, mp, pub, pp, commit = self.publication()
        with patch.object(b, '__file__', str(root / 'backup.py')):
            proof = b.source_publication(mp['path'], mp['sha256'], pp['path'], pp['sha256'], commit)
            self.assertEqual(len(proof['files']), 3)
            with self.assertRaises(b.Rejected): b.source_publication(mp['path'], mp['sha256'], pp['path'], pp['sha256'], 'b' * 40)
            (root / 'backup.py').write_bytes(b'changed')
            with self.assertRaises(b.Rejected): b.source_publication(mp['path'], mp['sha256'], pp['path'], pp['sha256'], commit)

    def test_publication_wrong_readback_and_unlisted_source_refused(self):
        root, mp, pub, pp, commit = self.publication()
        pub['files'][0]['equal'] = False; pp = Fixture.save(self.root / 'publication.json', pub)
        with patch.object(b, '__file__', str(root / 'backup.py')):
            with self.assertRaises(b.Rejected): b.source_publication(mp['path'], mp['sha256'], pp['path'], pp['sha256'], commit)
        pub['files'][0]['equal'] = True; pp = Fixture.save(self.root / 'publication.json', pub)
        (root / '__pycache__').mkdir()
        with patch.object(b, '__file__', str(root / 'backup.py')):
            with self.assertRaises(b.Rejected): b.source_publication(mp['path'], mp['sha256'], pp['path'], pp['sha256'], commit)

    def args(self):
        return argparse.Namespace(host='hkust1', remote_helper_root="/data/own/source with space", config='/data/own/config.json',
           config_sha256='a' * 64, checkpoint='/data/own/checkpoint/manifest.json', checkpoint_sha256=b.CP89_SHA,
           expected_run=b.RUN, expected_updates=89, expected_outer=8, source_manifest_sha256='b' * 64,
           remote_publication="/data/own/pub'$(do-not-run).json", publication_sha256='c' * 64,
           source_commit='d' * 40, destination=str(self.root / 'backup'), margin_gib=5)

    def test_ssh_explicit_quoted_argv_and_hkust1_only(self):
        args = self.args(); command = b.remote_command(args, 'inventory')
        self.assertEqual(command[:3], ['ssh', '-oConnectTimeout=15', 'hkust1'])
        remote = shlex.split(command[-1]); self.assertIn(args.remote_publication, remote)
        self.assertIn(args.remote_helper_root + '/backup.py', remote)
        args.host = 'foreign'
        with self.assertRaises(b.Rejected): b.remote_command(args, 'inventory')

    def test_inventory_failure_preserved_no_complete(self):
        args = self.args()
        result = argparse.Namespace(returncode=1, stdout=b'', stderr=b'readonly failure')
        with patch.object(b.subprocess, 'run', return_value=result), self.assertRaises(b.Rejected): b.backup(args, {})
        destination = Path(args.destination)
        self.assertTrue((destination / 'failed.json').exists())
        self.assertFalse((destination / 'complete-sha-verified.json').exists())
        with self.assertRaises(b.Rejected): b.backup(args, {})

    def test_failed_transfer_retains_partial_no_complete(self):
        f = Fixture(self.root); v = f.inv(); args = self.args()
        args.config, args.config_sha256, args.checkpoint, args.checkpoint_sha256, args.expected_run, args.expected_updates, args.expected_outer = f.args()
        pub = dict(manifest=dict(sha256='a' * 64), publication=dict(sha256='b' * 64), source_commit='c' * 40)
        v['source_publication'] = pub
        result = argparse.Namespace(returncode=0, stdout=b.canonical(v), stderr=b'')

        class Child:
            pid = 1; returncode = 7; stdin = None
            def communicate(self, value): pass
            def wait(self): return self.returncode

        def popen(*unused, **kwargs):
            kwargs['stdout'].write(b'partial'); return Child()

        with patch.object(b.subprocess, 'run', return_value=result), patch.object(b.subprocess, 'Popen', side_effect=popen), \
             patch.object(b.shutil, 'disk_usage', return_value=argparse.Namespace(free=20 << 30)):
            with self.assertRaises(b.Rejected): b.backup(args, pub)
        destination = Path(args.destination)
        self.assertEqual((destination / 'checkpoint.tar.partial').read_bytes(), b'partial')
        self.assertTrue(json.loads((destination / 'failed.json').read_text())['partial_retained'])
        self.assertEqual(json.loads((destination / 'transfer-exit.json').read_text())['actual_returncode'], 7)
        self.assertFalse((destination / 'complete-sha-verified.json').exists())

    def full_transfer_fixture(self, corrupt_readback=False):
        f = Fixture(self.root); v = f.inv(); args = self.args()
        args.config, args.config_sha256, args.checkpoint, args.checkpoint_sha256, args.expected_run, args.expected_updates, args.expected_outer = f.args()
        pub = dict(manifest=dict(sha256='a' * 64), publication=dict(sha256='b' * 64), source_commit='c' * 40)
        v['source_publication'] = pub
        result = argparse.Namespace(returncode=0, stdout=b.canonical(v), stderr=b'')

        class Child:
            pid = 2; returncode = 0; stdin = None
            def communicate(self, value): pass
            def wait(self): return 0

        def popen(*unused, **kwargs):
            b.stream_archive(v, kwargs['stdout']); return Child()

        with patch.object(b.subprocess, 'run', return_value=result), patch.object(b.subprocess, 'Popen', side_effect=popen), \
             patch.object(b.shutil, 'disk_usage', return_value=argparse.Namespace(free=20 << 30)):
            if corrupt_readback:
                with patch.object(b, 'readback', side_effect=b.Rejected('Independent readback failed')):
                    with self.assertRaises(b.Rejected): b.backup(args, pub)
            else:
                self.assertTrue(b.backup(args, pub)['accepted'])
        return Path(args.destination)

    def test_synthetic_success_complete_only_after_disk_readback(self):
        destination = self.full_transfer_fixture()
        proof = json.loads((destination / 'complete-sha-verified.json').read_text())
        self.assertEqual(proof['file_count'], 18)
        self.assertTrue(proof['independent_local_disk_readback'])
        self.assertEqual(proof['actual_transfer_returncode'], 0)
        self.assertFalse((destination / 'checkpoint.tar.partial').exists())
        self.assertTrue((destination / 'checkpoint.tar').exists())
        self.assertFalse((destination / 'failed.json').exists())
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((destination / 'verified-files').stat().st_mode), 0o700)

    def test_failed_independent_readback_preserves_partial_and_never_completes(self):
        destination = self.full_transfer_fixture(corrupt_readback=True)
        self.assertTrue((destination / 'checkpoint.tar.partial').exists())
        self.assertTrue((destination / 'verified-files').is_dir())
        self.assertFalse((destination / 'complete-sha-verified.json').exists())


if __name__ == '__main__': unittest.main()
