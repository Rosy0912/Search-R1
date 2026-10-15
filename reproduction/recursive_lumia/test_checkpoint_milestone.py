"""A saved later milestone must not imply a reload or accept foreign/tampered state."""
import json
from pathlib import Path
import tempfile
import unittest

from .inspect_run import checkpoint, sha


class CheckpointMilestone(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.identity = dict(job_id='1', run_id='run', code_commit='source',
                             config_sha256='config', data_identity='data')

    def write(self, outer=50):
        root = self.run/'checkpoints/training_state'/f'global_step_{outer}'
        identity = {k:v for k,v in self.identity.items() if k != 'run_id'}
        identity['completed_step'] = outer
        paths = []
        for label, directory in [('pre', root), ('post', root/'post-evaluation')]:
            directory.mkdir(parents=True)
            workers = []
            for rank in range(8):
                path = directory/f'rank-{rank:02d}.pt'
                path.write_bytes(f'opaque-state-{rank}'.encode())
                workers.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                    identity=dict(identity, rank=rank, world_size=8), scheduler_last_epoch=outer,
                    full_worker_state_finite=True, exact_same_world_reload_verified=False))
            path = directory/'driver.pt'; path.write_bytes(b'opaque-driver')
            manifest = dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', identity=identity,
                worker_states=workers, driver=dict(path=str(path), sha256=sha(path),
                next_step=outer+1, exact_reload_verified=False))
            if label == 'post': manifest['boundary_phase'] = 'post-evaluation'
            path = directory/'manifest.json'; path.write_text(json.dumps(manifest)); paths.append(path)
        return paths

    def test_later_saved_files_do_not_claim_reload(self):
        self.write()
        result = checkpoint(self.run, self.identity, True, outer=50)
        self.assertTrue(result['all_native_files_hash_verified'])
        self.assertFalse(result['same_world_reload_verified'])
        self.assertEqual(result['post_file_count'], 9)

    def test_pending_post_and_metadata_only_cannot_claim_all_hashes(self):
        _, post = self.write()
        self.assertFalse(checkpoint(self.run, self.identity, False, 50)['all_native_files_hash_verified'])
        post.unlink()
        self.assertFalse(checkpoint(self.run, self.identity, True, 50)['all_native_files_hash_verified'])

    def test_same_size_byte_tampering_is_rejected(self):
        pre, _ = self.write()
        path = pre.parent/'rank-00.pt'
        raw = path.read_bytes();path.write_bytes(b'X'+raw[1:])
        with self.assertRaisesRegex(ValueError, 'SHA'):
            checkpoint(self.run, self.identity, True, 50)

    def test_worker_identity_and_driver_counter_are_checked(self):
        pre, _ = self.write(); original = json.loads(pre.read_text())
        for mutation in ('worker_step', 'driver_next', 'scheduler', 'ranks'):
            with self.subTest(mutation=mutation):
                value = json.loads(json.dumps(original))
                if mutation == 'worker_step': value['worker_states'][0]['identity']['completed_step'] = 5
                if mutation == 'driver_next': value['driver']['next_step'] = 6
                if mutation == 'scheduler': value['worker_states'][0]['scheduler_last_epoch'] = 5
                if mutation == 'ranks': value['worker_states'][1]['identity']['rank'] = 0
                pre.write_text(json.dumps(value))
                with self.assertRaises(ValueError): checkpoint(self.run, self.identity, True, 50)

    def test_default_still_targets_fifth_boundary(self):
        self.write(5)
        self.assertTrue(checkpoint(self.run, self.identity, True)['all_native_files_hash_verified'])
        with self.assertRaises(ValueError): checkpoint(self.run, self.identity, True, 6)


if __name__ == '__main__':
    unittest.main()
