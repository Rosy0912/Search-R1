import copy
import unittest

from .restore_probe import CONFIG, RUNTIME, PARENT_JOB, validate_parent


class ParentBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(group='C1-root-only', source_commit=RUNTIME,
                           config_sha256=CONFIG, data_identity='data')
        identity = dict(job_id=PARENT_JOB, code_commit=RUNTIME, config_sha256=CONFIG,
                        data_identity='data', actual_update_limit=150, completed_step=1)
        self.manifest = dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', identity=identity,
            worker_states=[dict(identity=dict(identity, rank=r, world_size=8),
                actual_total_optimizer_calls=20, scheduler_last_epoch=1) for r in range(8)])

    def test_exact_boundary(self):
        validate_parent(self.record, self.manifest)

    def test_other_run_or_configuration_is_refused(self):
        for key in ('group', 'source_commit', 'config_sha256', 'data_identity'):
            record = dict(self.record, **{key: 'different'})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_parent(record, self.manifest)

    def test_duplicate_or_missing_rank_is_refused(self):
        for ranks in (self.manifest['worker_states'][:-1],
                      self.manifest['worker_states'][:-1] + [self.manifest['worker_states'][0]]):
            with self.assertRaises(ValueError):
                validate_parent(self.record, dict(self.manifest, worker_states=ranks))

    def test_changed_worker_state_is_refused(self):
        for key, value in [('actual_total_optimizer_calls', 0), ('scheduler_last_epoch', 0)]:
            manifest = copy.deepcopy(self.manifest)
            manifest['worker_states'][3][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_parent(self.record, manifest)

    def test_other_boundary_is_refused(self):
        for key, value in [('job_id', 'another'), ('completed_step', 2), ('actual_update_limit', 200)]:
            manifest = copy.deepcopy(self.manifest)
            manifest['identity'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_parent(self.record, manifest)

    def test_worker_identity_does_not_inherit_unchecked_manifest(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['worker_states'][7]['identity']['config_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            validate_parent(self.record, manifest)


if __name__ == '__main__':
    unittest.main()
