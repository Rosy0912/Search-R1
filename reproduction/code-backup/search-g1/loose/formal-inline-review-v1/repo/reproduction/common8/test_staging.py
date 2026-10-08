"""CPU-only tiny-file failure recovery and ownership guards, not GPU admission."""
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from reproduction.common8.stage_retrieval import cleanup,memory_bytes,resource_check


class ResourceFloor(unittest.TestCase):
    def test_low_memory_or_space_rejected(self):
        big=512*2**30
        self.assertEqual(memory_bytes('cpu=192,mem=524288M'),big)
        self.assertEqual(memory_bytes('cpu=192,mem=512G'),big)
        self.assertEqual(memory_bytes('cpu=192,mem=524288'),big)
        self.assertEqual(resource_check(big,big,big)['input_bytes'],78952654893)
        for values in ((32*2**30,big,big),(big,64*2**30,big),(big,big,32*2**30)):
            with self.assertRaises(AssertionError):resource_check(*values)


@unittest.skipUnless(sys.platform=='linux','actual Linux /dev/shm ownership fixtures')
class CleanupRecovery(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='g1-staging-cpu-',dir=os.environ.get('SEARCH_G1_TEST_TMP'))
        self.base=Path(self.temp.name);(self.base/'logs').mkdir()
        self.run_id=uuid.uuid4().hex
        self.root=Path('/dev/shm')/('search-g1-common8-'+self.run_id+'-99999999')
        self.root.mkdir(mode=0o700,exist_ok=False)
        self.claim=dict(run_id=self.run_id,slurm_job_id='99999999',uid=os.getuid(),root=str(self.root),
            pid=99999999,start_ticks=1,phase='RESERVED',root_device=self.root.stat().st_dev,root_inode=self.root.stat().st_ino)
        (self.base/'logs/run-manifest.json').write_text(json.dumps(dict(profile='common8',run_id=self.run_id)))
        self.write_claim()
    def write_claim(self):
        (self.base/'logs/retrieval-staging-claim.json').write_text(json.dumps(self.claim))
    def tearDown(self):
        # Only this fixture's newly created UUID namespace, never real run paths.
        if self.root.exists():shutil.rmtree(self.root)
        self.temp.cleanup()
    def test_failed_owner_write_no_status(self):
        (self.root/'owner.tmp').write_text('{truncated')
        (self.root/'e5_Flat.index.partial').write_bytes(b'fixture')
        self.assertTrue(cleanup(self.base)['cleaned']);self.assertFalse(self.root.exists())
        self.assertTrue(cleanup(self.base)['cleaned'])
    def test_interrupted_cleanup_without_owner(self):
        self.claim['phase']='CLEANING';self.write_claim()
        (self.root/'wiki-18.jsonl.partial').write_bytes(b'fixture')
        self.assertTrue(cleanup(self.base)['cleaned']);self.assertFalse(self.root.exists())
    def test_foreign_name_or_symlink_refuses_without_deletion(self):
        kept=self.root/'e5_Flat.index.partial';kept.write_bytes(b'fixture')
        other=self.root/'foreign';other.write_bytes(b'foreign fixture')
        with self.assertRaises(AssertionError):cleanup(self.base)
        self.assertEqual(kept.read_bytes(),b'fixture');self.assertTrue(other.exists())
        other.unlink();(self.root/'owner.tmp').symlink_to(kept)
        with self.assertRaises(AssertionError):cleanup(self.base)
        self.assertEqual(kept.read_bytes(),b'fixture')
    def test_actual_live_stager_rejected(self):
        from reproduction.a800.owned_processes import identity
        self.claim.update(pid=os.getpid(),start_ticks=identity(os.getpid())['start_ticks']);self.write_claim()
        kept=self.root/'e5_Flat.index.partial';kept.write_bytes(b'fixture')
        with self.assertRaises(AssertionError):cleanup(self.base)
        self.assertEqual(kept.read_bytes(),b'fixture')


if __name__=='__main__':unittest.main()
