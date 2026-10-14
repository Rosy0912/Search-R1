"""Meaningful refusal checks on tiny metadata fixtures; no GPU/tensor libraries."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import inspect as storage


SOURCE = '1' * 40


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class StorageChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # macOS /var is itself a symlink; fixture must be the real path.
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / storage.GROUPS[0]
        config = self.base / 'receipts/resolved.yaml'
        config.parent.mkdir(parents=True)
        config.write_text('small immutable fixture\n')
        self.receipt = dict(protocol=storage.PROTOCOL, group=storage.GROUPS[0],
            run_id='test-run', source_commit=SOURCE, config=str(config),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            base=str(self.base), output=str(self.base / 'output'),
            data_identity='d' * 64, actual_update_limit=150, evaluation_steps=list(storage.KEYS))
        save(self.base / 'receipts/configuration.json', self.receipt)
        for step in storage.KEYS:
            self.state(step, 'pre')
            self.state(step, 'post')
        self.state(148, 'rolling')
        self.state(149, 'rolling')
        output = self.base / 'output/attempt-fixture'
        entry = dict(pid=234567, uid=0, tick=10, output=str(output), **{
            key: self.receipt[key] for key in ('group', 'run_id', 'config_sha256', 'source_commit')})
        save(self.base / 'receipts/current-entry.json', entry)
        manifest = self.base / 'checkpoints/training_state/global_step_150/post-evaluation/manifest.json'
        complete_sha = save(output / 'training-complete.json', dict(
            status='COMPLETE_NATIVE_150_TRAINING_AND_FOUR_SCHEDULED_DEV', completed_outer=150,
            evaluation_steps=list(storage.KEYS), checkpoint=dict(path=str(manifest),
            sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), native_only=False), **{
            key: self.receipt[key] for key in ('group', 'run_id', 'config_sha256', 'source_commit')}))
        save(output / 'entry-exit.json', dict(rc=0, error=None, **{
            key: entry[key] for key in ('pid', 'uid', 'tick', 'group', 'run_id')}))
        save(self.base / 'receipts/actual-parent-wait.json', dict(pid=entry['pid'], rc=0,
            group=entry['group'], source_commit=SOURCE))
        self.study = dict(source_commit=SOURCE, current='C2-random-final', complete={entry['group']: dict(
            training_complete=str(output / 'training-complete.json'), complete_sha256=complete_sha,
            actual_parent_wait=str(self.base / 'receipts/actual-parent-wait.json'), actual_parent_wait_rc=0,
            owned_gpu_release=dict(complete=True, applications=[], query_rc=0))})
        save(self.root / 'study-progress.json', self.study)

    def tearDown(self):
        self.temp.cleanup()

    def state(self, step, phase):
        if phase == 'rolling':
            directory = self.base / 'checkpoints/native-rolling' / f'global_step_{step}'
        else:
            directory = self.base / 'checkpoints/training_state' / f'global_step_{step}'
            if phase == 'post':
                directory /= 'post-evaluation'
        directory.mkdir(parents=True, exist_ok=True)
        ident = dict(code_commit=SOURCE, config_sha256=self.receipt['config_sha256'],
            data_identity=self.receipt['data_identity'], actual_update_limit=150, completed_step=step,
            model_identity='model@test', job_id='3393')
        workers = []
        for rank in range(8):
            path = directory / f'rank-{rank:02d}.pt'
            path.write_bytes(b'fixture payload')
            workers.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                logical_state_sha256='a' * 64, bytes=path.stat().st_size,
                identity=dict(ident, rank=rank, world_size=8), full_worker_state_finite=True,
                scheduler_last_epoch=step))
        driver = directory / 'driver.pt'
        driver.write_bytes(b'fixture driver')
        value = dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', identity=ident, worker_states=workers,
            driver=dict(path=str(driver), next_step=step+1,
                        sha256=hashlib.sha256(driver.read_bytes()).hexdigest(), logical_state_sha256='b' * 64))
        if phase == 'rolling':
            value.update(snapshot_kind='native_only_rolling_two', hf_checkpoint=None, hf_identity=None)
        else:
            hf = self.base / 'checkpoints/actor' / f'global_step_{step}'
            hf.mkdir(parents=True, exist_ok=True)
            for name in ('config.json', 'tokenizer_config.json', 'model.safetensors'):
                (hf / name).write_bytes(b'fixture HF')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in hf.iterdir()}
            value.update(hf_checkpoint=str(hf), hf_identity=dict(files=hashes,
                sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()))
            if phase == 'post':
                value['boundary_phase'] = 'post-evaluation'
        save(directory / 'manifest.json', value)

    def collect(self, active=False, writers=None):
        with patch.object(storage, 'owner_present', return_value=active), \
             patch.object(storage, 'open_writers', return_value=writers or []):
            return storage.collect(self.root, SOURCE)

    def test_candidates_only_exact_payloads_and_actual_space_guard(self):
        snapshot = self.collect()
        group = snapshot['groups'][0]
        self.assertEqual(group['status'], 'COMPLETE_EXIT_WAIT_RELEASE_ACCEPTED')
        self.assertEqual(len(group['candidates']), 6)
        files = [f for c in group['candidates'] for f in c['files']]
        self.assertEqual(len(files), 54)
        self.assertTrue(all(f['path'].endswith('.pt') and '/post-evaluation/' not in f['path']
                            and '/actor/' not in f['path'] for f in files))
        snapshot['statvfs_free_bytes'] = 10**12
        enough = storage.plan(snapshot)
        self.assertFalse(enough['space_guard']['cleanup_needed'])
        self.assertEqual(enough['candidates_for_review'], [])
        snapshot['statvfs_free_bytes'] = 1
        low = storage.plan(snapshot)
        self.assertEqual(len(low['candidates_for_review']), 6)
        self.assertFalse(low['actual_receipt']['deletion_executed'])
        self.assertTrue(all(Path(f['path']).is_file() for f in files))

    def test_active_owner_and_writer_refuse_all_candidates(self):
        for active, writers in ((True, []), (False, [dict(pid=1, path=str(self.base / 'logs/open.log'))])):
            self.assertEqual(self.collect(active=active, writers=writers)['groups'][0]['candidates'], [])

    def test_missing_protected_payload_and_symlink_refuse(self):
        path = self.base / 'checkpoints/training_state/global_step_50/post-evaluation/rank-02.pt'
        raw = path.read_bytes()
        path.unlink()
        missing = self.collect()['groups'][0]
        self.assertEqual(missing['candidates'], [])
        external = self.root / 'outside-owned-group.pt'
        external.write_bytes(raw)
        path.symlink_to(external)
        linked = self.collect()['groups'][0]
        self.assertEqual(linked['candidates'], [])
        self.assertIn('symlink', json.dumps(linked['refusals']))

    def test_wait_failure_and_missing_keypoint_refuse(self):
        wait = self.base / 'receipts/actual-parent-wait.json'
        value = json.loads(wait.read_text())
        value['rc'] = 1
        save(wait, value)
        self.assertEqual(self.collect()['groups'][0]['candidates'], [])
        value['rc'] = 0
        save(wait, value)
        (self.base / 'checkpoints/training_state/global_step_5/post-evaluation/manifest.json').unlink()
        self.assertEqual(self.collect()['groups'][0]['candidates'], [])

    def test_source_and_config_mismatch_refuse(self):
        state = self.base / 'checkpoints/native-rolling/global_step_149/manifest.json'
        value = json.loads(state.read_text())
        value['identity']['code_commit'] = '2' * 40
        save(state, value)
        self.assertEqual(self.collect()['groups'][0]['candidates'], [])
        Path(self.receipt['config']).write_text('changed config')
        result = self.collect()['groups'][0]
        self.assertEqual(result['status'], 'REFUSED_BINDING_OR_INVENTORY')
        self.assertEqual(result['candidates'], [])


if __name__ == '__main__':
    unittest.main()
