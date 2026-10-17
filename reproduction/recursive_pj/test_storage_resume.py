"""Storage-stop continuations must use the preserved boundary, not retired parents."""
import copy
import unittest
from unittest.mock import patch
from .resume_profiles import profile
from . import resume_transport as transport


class StorageResume(unittest.TestCase):
    def fixture(self, name):
        p = profile(name)
        step = p['step']
        identity = dict(job_id=p['parent_job'], code_commit=p['parent_source'],
                        config_sha256=p['config_sha'], data_identity='data',
                        completed_step=step, actual_update_limit=150)
        record = dict(group=p['group'], run_id=p['run_id'], source_commit=p['parent_source'],
                      config_sha256=p['config_sha'], data_identity='data')
        workers = [dict(identity=dict(identity, rank=i, world_size=8),
                        actual_total_optimizer_calls=20*step, scheduler_last_epoch=step,
                        full_worker_state_finite=True, path='/saved/rank%d.pt' % i,
                        sha256=str(i)) for i in range(8)]
        driver = dict(path='/saved/driver.pt', sha256='driver', next_step=step+1)
        manifest = dict(identity=identity, status='COMPLETE_SAME_WORLD_STATE_SAVED',
                        worker_states=workers, driver=driver)
        binding = dict(parent_job=p['parent_job'], completed_outer=step,
                       manifest_sha256='m', parent_entry_sha256='e', protective_stop_sha256='stop',
                       metadata_sha256={'protective-stop-intent.json': 'stop'},
                       members=[dict(name=w['path'].split('/')[-1], original=w['path'],
                                     sha256=w['sha256']) for w in workers+[driver]])
        probe = dict(job=p['target_job'], status='INDEPENDENT_EIGHT_GPU_COLD_STATE_EQUALITY_VERIFIED',
                     training_updates_executed=0, completed_outer=step, parent_manifest_sha256='m',
                     ranks=[dict(rank=i, actual_state_equal=True, observed_adam_steps=[20*step]) for i in range(8)])
        gate = dict(parent_job=p['parent_job'], parent_state='TERMINATED', completed_outer=step,
                    latest_durable_outer=step, manifest_sha256='m', parent_entry_sha256='e',
                    target_job=p['target_job'], probe_sha256='probe', protective_stop_sha256='stop')
        return p, binding, record, manifest, probe, gate

    def test_exact_protected_boundaries_and_reject_stale_or_incomplete_state(self):
        for name in ['budget5-storage53-v6', 'depth2-storage105-v2']:
            p, b, r, m, probe, gate = self.fixture(name)
            values = dict(PROFILE=p, GROUP=p['group'], STEP=p['step'], PARENT_JOB=p['parent_job'],
                          PARENT_SOURCE=p['parent_source'], RUN_ID=p['run_id'], CONFIG_SHA=p['config_sha'])
            with self.subTest(profile=name), patch.multiple(transport, **values), patch.object(transport, 'sha', return_value='probe'):
                self.assertTrue(transport.validate_inputs(b, r, m))
                transport.validate_gate(gate, b, probe)
                for stale_step in [2, 48, 56]:
                    changed = copy.deepcopy(m); changed['identity']['completed_step'] = stale_step
                    with self.assertRaises(ValueError): transport.validate_inputs(b, r, changed)
                changed = copy.deepcopy(m); changed['worker_states'][0]['actual_total_optimizer_calls'] = 0
                with self.assertRaises(ValueError): transport.validate_inputs(b, r, changed)
                changed = copy.deepcopy(probe); changed['ranks'].pop()
                with self.assertRaises(ValueError): transport.validate_gate(gate, b, changed)
                for state in ['RUNNING', 'TERMINATING', 'FAILED', 'COMPLETED']:
                    with self.assertRaises(ValueError): transport.validate_gate(dict(gate, parent_state=state), b, probe)
                for changed in [dict(b, protective_stop_sha256=None), dict(b, metadata_sha256={})]:
                    with self.assertRaises(ValueError): transport.validate_gate(gate, changed, probe)
                with self.assertRaises(ValueError): transport.validate_gate(dict(gate, protective_stop_sha256='other'), b, probe)

    def test_old_profiles_do_not_admit_terminated_parent(self):
        # The new terminal-state admission is limited to explicit storage profiles.
        p, b, r, m, probe, gate = self.fixture('budget5-storage53-v6')
        p.pop('storage_protected_parent')
        with patch.object(transport, 'PROFILE', p), patch.object(transport, 'sha', return_value='probe'):
            with self.assertRaises(ValueError): transport.validate_gate(gate, b, probe)


if __name__ == '__main__':
    unittest.main()
