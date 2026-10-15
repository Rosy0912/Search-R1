import copy
from types import SimpleNamespace
import unittest

from .runtime import admit_config, validate


class TransportAdmission(unittest.TestCase):
    def values(self):
        record = dict(host='gpu-4090-3', source_commit='frozen', run_id='run',
                      group='C1-root-only', config_sha256='config', data_identity='data')
        binding = dict(version=1, parent_host=record['host'], target_node='node10',
            target_host='gpu-l40s-1', frozen_source_commit='frozen', parent_job='10',
            **{k: record[k] for k in ('run_id', 'group', 'config_sha256', 'data_identity')})
        env = dict(SLURM_JOB_ID='11', SLURM_JOB_NODELIST='node10', RECURSIVE_SOURCE_COMMIT='frozen')
        return binding, record, env, 1066, 'gpu-l40s-1'

    def test_entry_and_ray_worker_identity(self):
        args = self.values()
        self.assertTrue(validate(*args))
        args[2]['SEARCH_R1_BASELINE_COMMIT'] = args[2].pop('RECURSIVE_SOURCE_COMMIT')
        self.assertTrue(validate(*args))

    def test_wrong_host_owner_source_run_and_duplicate_allocation_rejected(self):
        for mode in ('node', 'host', 'source', 'run', 'config', 'data', 'parent', 'uid', 'same_host'):
            args = list(copy.deepcopy(self.values()))
            if mode == 'node': args[2]['SLURM_JOB_NODELIST'] = 'node11'
            if mode == 'host': args[4] = 'gpu-l40s-2'
            if mode == 'source': args[2]['RECURSIVE_SOURCE_COMMIT'] = 'different'
            if mode == 'run': args[0]['run_id'] = 'different'
            if mode == 'config': args[0]['config_sha256'] = 'different'
            if mode == 'data': args[0]['data_identity'] = 'different'
            if mode == 'parent': args[2]['SLURM_JOB_ID'] = '10'
            if mode == 'uid': args[3] = 1000
            if mode == 'same_host': args[0]['parent_host'] = 'gpu-l40s-1'
            with self.assertRaises(ValueError, msg=mode): validate(*args)

    def test_admission_never_mutates_training_configuration(self):
        config = SimpleNamespace(baselines=SimpleNamespace(local_runtime_host='gpu-4090-3'),
                                 trainer=SimpleNamespace(n_gpus_per_node=8), learning_rate=1e-6)
        projected = admit_config(config, 'gpu-4090-3', 'gpu-l40s-1')
        self.assertEqual(config.baselines.local_runtime_host, 'gpu-4090-3')
        self.assertEqual(projected.baselines.local_runtime_host, 'gpu-l40s-1')
        self.assertEqual(projected.learning_rate, config.learning_rate)
        self.assertEqual(projected.trainer.n_gpus_per_node, 8)
        with self.assertRaises(ValueError): admit_config(config, 'unbound', 'gpu-l40s-1')

    def test_second_allocation_binds_actual_parent_and_original_config_host(self):
        args = list(self.values())
        args[0].update(version=2, configuration_host='gpu-4090-3', parent_host='gpu-l40s-1')
        self.assertTrue(validate(*args))
        for field, value in [('configuration_host', 'gpu-l40s-1'),
                             ('parent_host', 'unknown'), ('version', 3)]:
            invalid = copy.deepcopy(args)
            invalid[0][field] = value
            with self.assertRaises(ValueError): validate(*invalid)
        args[2]['SLURM_JOB_ID'] = args[0]['parent_job']
        with self.assertRaises(ValueError): validate(*args)


if __name__ == '__main__':
    unittest.main()
