"""Linux tiny-cache ownership and failed/interrupted cleanup, never large data."""
import json
import hashlib
import io
import os
import shutil
import sys
import tempfile
import tarfile
import unittest
import uuid
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from reproduction.common8 import temporary_cache as cache


class CorpusExtractionTests(unittest.TestCase):
    """Real tiny TAR/file bytes; synthetic native/capacity boundary, no GPU proof."""
    CONTENT = b'{"id":"0","contents":"first"}\n{"id":"1","contents":"second"}\n'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='g1-corpus-cpu-')
        self.base = Path(self.temp.name); (self.base / 'logs').mkdir()
        self.run = dict(profile='common8', run_id=uuid.uuid4().hex)
        namespace_root = Path('/dev/shm') if sys.platform == 'linux' else self.base
        self.parent = namespace_root / ('search-g1-common8-' + self.run['run_id'] + '-99999999')
        self.parent.mkdir(mode=0o700); self.root = self.parent / 'datasets-cache'; self.root.mkdir(mode=0o700)
        self.archive = self.parent / 'wiki-18.jsonl'
        self.header = dict(cache.CORPUS_MEMBER, size=len(self.CONTENT))
        self.creator = dict(pid=os.getpid(), start_ticks=1, sid=os.getsid(0), uid=os.getuid(), state='R')
        self.claim = dict(run_id=self.run['run_id'], slurm_job_id='99999999', uid=os.getuid(), root=str(self.root),
                          parent_device=self.parent.stat().st_dev, parent_inode=self.parent.stat().st_ino,
                          root_device=self.root.stat().st_dev, root_inode=self.root.stat().st_ino,
                          creator=self.creator, phase='CREATED')
        self.stage = dict(run_id=self.run['run_id'], slurm_job_id='99999999', uid=os.getuid(), root=str(self.parent),
                          root_device=self.parent.stat().st_dev, root_inode=self.parent.stat().st_ino,
                          pid=99999999, start_ticks=1, phase='STAGED')
        for name, value in (('run-manifest.json', self.run), ('corpus-cache-claim.json', self.claim),
                            ('retrieval-staging-claim.json', self.stage)):
            (self.base / 'logs' / name).write_text(json.dumps(value))
        self.write_tar()

    def tearDown(self):
        if self.parent.exists(): shutil.rmtree(self.parent)
        self.temp.cleanup()

    def write_tar(self, headers=None):
        with tarfile.open(self.archive, 'w') as archive:
            for header in headers or [self.header]:
                member = tarfile.TarInfo(header['name'])
                for key in ('size', 'uid', 'gid', 'mode', 'linkname'): setattr(member, key, header[key])
                member.type = header['type'].encode('ascii')
                archive.addfile(member, io.BytesIO(self.CONTENT) if member.isfile() else None)
        self.asset = dict(size=self.archive.stat().st_size,
                          sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest())

    def extract(self, capacity_error=None):
        def capacity(base, root):
            self.assertEqual((base, root), (self.base, self.root))
            self.assertFalse((root / 'wiki_dump.jsonl.partial').exists())
            self.assertFalse((root / 'wiki_dump.jsonl').exists())
            if capacity_error is not None: raise capacity_error
            return {'scope': 'tiny fixture capacity boundary only'}
        # Mac's real tiny files adapt only this Linux namespace path. Linux uses
        # the actual /dev/shm namespace even for these synthetic claim fixtures.
        namespace = patch.object(cache, '_cache_parent', return_value=self.parent) if sys.platform != 'linux' else nullcontext()
        with namespace, patch.object(cache, 'CORPUS_MEMBER', self.header), \
             patch.dict(cache.ASSETS, {'wiki-18.jsonl': self.asset}), \
             patch.object(cache, 'identity', return_value=self.creator), \
             patch.object(cache, 'admit_capacity', side_effect=capacity) as admitted:
            result = cache.extract_corpus(self.base, self.root, self.archive)
            admitted.assert_called_once()
            return result

    def test_original_tar_uid_not_restored_and_bytes_order_and_hashes_recorded(self):
        target = self.extract()
        self.assertEqual(target.read_bytes(), self.CONTENT)
        self.assertEqual(target.stat().st_uid, os.getuid())
        self.assertFalse((self.root / 'wiki_dump.jsonl.partial').exists())
        record = json.loads((self.base / 'logs/corpus-extraction.json').read_text())
        self.assertEqual(record['phase'], 'READY'); self.assertEqual(record['member'], self.header)
        self.assertEqual(record['archive_sha256'], self.asset['sha256'])
        self.assertEqual(record['content_sha256'], hashlib.sha256(self.CONTENT).hexdigest())
        self.assertEqual(record['content_size'], len(self.CONTENT))
        self.assertEqual((record['output_device'], record['output_inode']), (target.stat().st_dev, target.stat().st_ino))
        self.assertEqual(self.archive.stat().st_uid, os.getuid())
        self.assertNotIn('accepted', record)
        claim = json.loads((self.base / 'logs/corpus-cache-claim.json').read_text())
        self.assertEqual(claim['capacity_admission']['scope'], 'tiny fixture capacity boundary only')

    def test_capacity_rejection_precedes_archive_read_and_partial_creation(self):
        self.archive.write_bytes(b'not a TAR')
        with self.assertRaisesRegex(AssertionError, 'fixture capacity'):
            self.extract(AssertionError('fixture capacity'))
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertNotIn('corpus_extraction', json.loads((self.base / 'logs/corpus-cache-claim.json').read_text()))

    def test_archive_hash_and_exact_single_regular_header_required(self):
        self.asset['sha256'] = '0' * 64
        with self.assertRaisesRegex(AssertionError, 'TAR differs'): self.extract()
        self.assertEqual(list(self.root.iterdir()), [])
        for headers in ([self.header, self.header],
                        [dict(self.header, name='../escape.jsonl')],
                        [dict(self.header, uid=0)], [dict(self.header, gid=0)],
                        [dict(self.header, mode=0o777)],
                        [dict(self.header, size=0, type='2', linkname='/outside')],
                        [dict(self.header, size=0, type='1', linkname='/outside')]):
            with self.subTest(headers=headers):
                self.write_tar(headers)
                with self.assertRaises(AssertionError): self.extract()
                self.assertEqual(list(self.root.iterdir()), [])

    def test_stream_failure_retains_owned_partial_and_durable_failed_claim(self):
        original = tarfile.TarFile.extractfile
        class Interrupted:
            def __init__(self, stream): self.stream, self.first = stream, True
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def read(self, size):
                if self.first: self.first = False; return self.stream.read(7)
                raise OSError('fixture stream interruption')
        with patch.object(tarfile.TarFile, 'extractfile', lambda tar, member: Interrupted(original(tar, member))):
            with self.assertRaisesRegex(OSError, 'stream interruption'): self.extract()
        partial = self.root / 'wiki_dump.jsonl.partial'
        self.assertEqual(partial.read_bytes(), self.CONTENT[:7]); self.assertEqual(partial.stat().st_uid, os.getuid())
        self.assertFalse((self.root / 'wiki_dump.jsonl').exists())
        record = json.loads((self.base / 'logs/corpus-cache-claim.json').read_text())['corpus_extraction']
        self.assertEqual(record['phase'], 'FAILED')
        self.assertEqual(record['output_inode'], partial.stat().st_ino)
        self.assertFalse((self.base / 'logs/corpus-extraction.json').exists())

    def test_exclusive_output_symlink_and_wrong_claim_inode_refused(self):
        target = self.root / 'wiki_dump.jsonl'; target.write_bytes(b'keep')
        with self.assertRaises(AssertionError): self.extract()
        self.assertEqual(target.read_bytes(), b'keep'); target.unlink()
        target.symlink_to(self.archive)
        with self.assertRaises(AssertionError): self.extract()
        target.unlink()
        path = self.base / 'logs/corpus-cache-claim.json'
        claim = json.loads(path.read_text()); claim['root_inode'] += 1; path.write_text(json.dumps(claim))
        with self.assertRaises(AssertionError): self.extract()
        self.assertEqual(list(self.root.iterdir()), [])
        # The standalone production entry rejects another directory with the
        # correct basename before any capacity mutation or output creation.
        other_parent = self.base / self.parent.name
        other_root = other_parent / 'datasets-cache'; other_root.mkdir(parents=True, exist_ok=True)
        other_archive = other_parent / 'wiki-18.jsonl'
        if other_archive != self.archive: shutil.copyfile(self.archive, other_archive)
        stage_path = self.base / 'logs/retrieval-staging-claim.json'
        stage = json.loads(stage_path.read_text()); stage.update(root=str(other_parent),
            root_device=other_parent.stat().st_dev, root_inode=other_parent.stat().st_ino)
        stage_path.write_text(json.dumps(stage))
        claim.update(root=str(other_root), parent_device=other_parent.stat().st_dev, parent_inode=other_parent.stat().st_ino,
                     root_device=other_root.stat().st_dev, root_inode=other_root.stat().st_ino)
        path.write_text(json.dumps(claim))
        with patch.object(cache, 'identity', return_value=self.creator), patch.object(cache, 'admit_capacity') as admitted:
            with self.assertRaises(AssertionError): cache.extract_corpus(self.base, other_root, other_archive)
            admitted.assert_not_called()

    def test_modified_original_during_stream_cannot_publish_ready_proof(self):
        original = tarfile.TarFile.extractfile
        owner = self
        class Modified:
            def __init__(self, stream): self.stream, self.first = stream, True
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def read(self, size):
                data = self.stream.read(size)
                if self.first:
                    self.first = False
                    old = owner.archive.stat()
                    with owner.archive.open('r+b') as source:
                        source.seek(-1, os.SEEK_END); source.write(b'x'); source.flush(); os.fsync(source.fileno())
                    # Exercise the metadata drift guard even when this tiny
                    # same-size write occurs within one filesystem clock tick.
                    os.utime(owner.archive, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
                    owner.assertNotEqual(owner.archive.stat().st_mtime_ns, old.st_mtime_ns)
                return data
        with patch.object(tarfile.TarFile, 'extractfile', lambda tar, member: Modified(original(tar, member))):
            with self.assertRaisesRegex(AssertionError, 'changed while reading'): self.extract()
        self.assertFalse((self.root / 'wiki_dump.jsonl').exists())
        self.assertEqual((self.root / 'wiki_dump.jsonl.partial').read_bytes(), self.CONTENT)
        self.assertFalse((self.base / 'logs/corpus-extraction.json').exists())

    def test_actual_datasets_reads_jsonl_in_same_order_without_tar_extractor(self):
        try:
            import datasets
            from datasets.utils.extract import TarExtractor
        except ImportError: self.skipTest('Actual datasets dependency unavailable locally')
        target = self.extract()
        with patch.object(TarExtractor, 'extract', side_effect=AssertionError('No TAR re-extraction')):
            corpus = datasets.load_dataset('json', data_files=str(target), split='train', num_proc=1,
                                           cache_dir=str(self.root))
        self.assertEqual(corpus['id'], ['0', '1']); self.assertEqual(corpus['contents'], ['first', 'second'])


@unittest.skipUnless(sys.platform == 'linux', 'Actual private /dev/shm fixtures')
class CacheOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='g1-cache-cpu-')
        self.base = Path(self.temp.name); (self.base / 'logs').mkdir()
        self.run = dict(profile='common8', run_id=uuid.uuid4().hex)
        self.job = dict(JobId='99999999')
        self.parent = Path('/dev/shm') / ('search-g1-common8-' + self.run['run_id'] + '-99999999')
        self.parent.mkdir(mode=0o700)
        stage = dict(run_id=self.run['run_id'], slurm_job_id='99999999', phase='STAGED', uid=os.getuid(),
                     root=str(self.parent), root_device=self.parent.stat().st_dev, root_inode=self.parent.stat().st_ino,
                     pid=99999999, start_ticks=1)
        (self.base / 'logs/run-manifest.json').write_text(json.dumps(self.run))
        (self.base / 'logs/retrieval-staging-claim.json').write_text(json.dumps(stage))
    def tearDown(self):
        if self.parent.exists(): shutil.rmtree(self.parent)
        self.temp.cleanup()
    def stopped(self):
        path = self.base / 'logs/corpus-cache-claim.json'
        claim = json.loads(path.read_text()); claim['creator'].update(pid=99999999, start_ticks=1)
        path.write_text(json.dumps(claim)); return path, claim
    def test_live_creator_rejected_then_nested_cache_cleanup_repeatable(self):
        root = cache.create(self.base, self.run, self.job)
        (root / 'json').mkdir(); kept = root / 'json/fixture.arrow'; kept.write_bytes(b'tiny fixture')
        with self.assertRaises(AssertionError): cache.cleanup(self.base)
        self.assertTrue(kept.exists()); self.stopped()
        self.assertTrue(cache.cleanup(self.base)['cleaned'])
        self.assertTrue(cache.cleanup(self.base)['cleaned']); self.assertTrue(self.parent.exists())
    def test_symlink_refuses_before_any_cache_deletion(self):
        root = cache.create(self.base, self.run, self.job)
        kept = root / 'fixture.arrow'; kept.write_bytes(b'tiny')
        (root / 'bad').symlink_to(kept); self.stopped()
        with self.assertRaises(AssertionError): cache.cleanup(self.base)
        self.assertTrue(kept.exists())
    def test_interrupted_cleanup_and_initial_reservation_are_recoverable(self):
        root = cache.create(self.base, self.run, self.job)
        (root / 'fixture.arrow').write_bytes(b'tiny')
        path, claim = self.stopped(); claim['phase'] = 'CLEANING'; path.write_text(json.dumps(claim))
        self.assertTrue(cache.cleanup(self.base)['cleaned'])
        # Recovery of mkdir-before-inode-publication in this same reserved namespace.
        root.mkdir(); (root / 'partial').write_bytes(b'tiny')
        claim['phase'] = 'RESERVED'; claim.pop('root_device'); claim.pop('root_inode')
        path.write_text(json.dumps(claim)); self.assertTrue(cache.cleanup(self.base)['cleaned'])
    def test_substituted_directory_inode_rejected(self):
        root = cache.create(self.base, self.run, self.job); root.rename(self.parent / 'saved')
        root.mkdir(); self.stopped()
        with self.assertRaises(AssertionError): cache.cleanup(self.base)
        self.assertTrue(root.exists())
    def test_insufficient_tmpfs_refused_before_corpus_loading_and_claim_retained(self):
        root = cache.create(self.base, self.run, self.job)
        usage = type('Usage', (), {'free': 1})()
        with patch.object(cache.shutil, 'disk_usage', return_value=usage):
            with self.assertRaises(AssertionError): cache.admit_capacity(self.base, root)
        claim = json.loads((self.base / 'logs/corpus-cache-claim.json').read_text())
        self.assertEqual(claim['capacity_admission']['actual_tmpfs_free_bytes'], 1)
        self.assertEqual(cache.capacity_floor(), 44254480384)
        self.stopped(); self.assertTrue(cache.cleanup(self.base)['cleaned'])

    def test_owned_stream_and_dataset_loading_partials_clean_without_uid_relaxation(self):
        root = cache.create(self.base, self.run, self.job)
        # Tiny remnants represent interrupted stream/load; both are caller-owned.
        (root / 'wiki_dump.jsonl.partial').write_bytes(b'{"id":"0"}\n')
        (root / 'json').mkdir(); (root / 'json/fixture.arrow.partial').write_bytes(b'partial Arrow')
        self.stopped(); self.assertTrue(cache.cleanup(self.base)['cleaned'])
        self.assertFalse(root.exists())

    def test_actual_tiny_extract_then_load_failure_and_strict_cleanup(self):
        root = cache.create(self.base, self.run, self.job)
        archive = self.parent / 'wiki-18.jsonl'; content = CorpusExtractionTests.CONTENT
        header = dict(cache.CORPUS_MEMBER, size=len(content))
        with tarfile.open(archive, 'w') as tar:
            member = tarfile.TarInfo(header['name'])
            for key in ('size', 'uid', 'gid', 'mode', 'linkname'): setattr(member, key, header[key])
            member.type = header['type'].encode('ascii'); tar.addfile(member, io.BytesIO(content))
        asset = dict(size=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        # Actual Linux creator birth/inodes/tmpfs/RAM checks; only data-size floor
        # is the tiny fixture's boundary. This is no production asset admission.
        with patch.object(cache, 'CORPUS_MEMBER', header), \
             patch.dict(cache.ASSETS, {'wiki-18.jsonl': asset}), patch.object(cache, 'capacity_floor', return_value=1):
            target = cache.extract_corpus(self.base, root, archive)
        self.assertEqual(target.read_bytes(), content); self.assertEqual(target.stat().st_uid, os.getuid())
        claim = json.loads((self.base / 'logs/corpus-cache-claim.json').read_text())
        self.assertEqual(claim['capacity_admission']['required_capacity_bytes'], 1)
        self.assertGreater(claim['capacity_admission']['actual_tmpfs_free_bytes'], 1)
        def interrupted_loader():
            (root / 'json').mkdir(); (root / 'json/fixture.arrow.partial').write_bytes(b'partial Arrow')
            raise OSError('fixture downstream load interrupted')
        with self.assertRaisesRegex(OSError, 'load interrupted'): interrupted_loader()
        self.stopped(); self.assertTrue(cache.cleanup(self.base)['cleaned'])
        self.assertFalse(root.exists()); self.assertEqual(archive.read_bytes().__len__(), asset['size'])

    def test_special_file_and_foreign_uid_still_refused_before_deletion(self):
        root = cache.create(self.base, self.run, self.job)
        kept = root / 'fixture.arrow'; kept.write_bytes(b'keep')
        special = root / 'fifo'; os.mkfifo(special); self.stopped()
        with self.assertRaises(AssertionError): cache.cleanup(self.base)
        self.assertEqual(kept.read_bytes(), b'keep'); special.unlink()
        if os.getuid() == 0:
            os.chown(kept, 1012, 1012)
            with self.assertRaises(AssertionError): cache.cleanup(self.base)
            self.assertEqual(kept.read_bytes(), b'keep')
            os.chown(kept, os.getuid(), os.getgid())


if __name__ == '__main__': unittest.main()
