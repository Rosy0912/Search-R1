"""Linux tiny-cache ownership and failed/interrupted cleanup, never large data."""
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from reproduction.common8 import temporary_cache as cache


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
        self.assertEqual(cache.capacity_floor(), 29860900864)
        self.stopped(); self.assertTrue(cache.cleanup(self.base)['cleaned'])


if __name__ == '__main__': unittest.main()
