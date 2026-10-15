import copy
from pathlib import Path
import unittest

from .ablation_entry import validate_allocation, check_previous_scheduler, runtime_environment
from .prepare_ablation import BASELINE_CONFIG_SHA, BASELINE_RUN_ID, condition_for


class LumiaEntry(unittest.TestCase):
    def fixture(self):
        r=dict(execution_backend='slurm_lumia',host='gpu-l40s-1',source_commit='a'*40,
            template_sha256=BASELINE_CONFIG_SHA,baseline_run_id=BASELINE_RUN_ID,
            group='C1-root-only',condition=condition_for('C1-root-only'),run_id='run',
            config_sha256='c'*64,data_identity='d'*64,data='/data',model='/model')
        e=dict(SLURM_JOB_ID='123',SLURM_JOB_NODELIST='node10',
               CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',RECURSIVE_SOURCE_COMMIT='a'*40)
        return r,e

    def test_accepts_actual_lumia_boundary(self):
        r,e=self.fixture()
        validate_allocation(r,e,1066,'gpu-l40s-1')
        validate_allocation(dict(r,host='gpu-4090-3'),dict(e,SLURM_JOB_NODELIST='node6'),
                            1066,'gpu-4090-3')

    def test_new_idle_nodes_match_scheduler_host_identity(self):
        r,e=self.fixture()
        for node,host in [('node2','gpu-3090-1'),('node8','gpu-ada6000-1')]:
            validate_allocation(dict(r,host=host),dict(e,SLURM_JOB_NODELIST=node),1066,host)
            with self.assertRaises(ValueError):
                validate_allocation(dict(r,host=host),dict(e,SLURM_JOB_NODELIST='node9'),1066,host)

    def test_rejects_wrong_owner_backend_or_devices(self):
        r,e=self.fixture()
        for field,value in [('SLURM_JOB_NODELIST','h100-11'),('CUDA_VISIBLE_DEVICES','0,0,1,2,3,4,5,6'),
                            ('RECURSIVE_SOURCE_COMMIT','b'*40),('SLURM_JOB_ID','')]:
            changed=dict(e,**{field:value})
            with self.assertRaises(ValueError):validate_allocation(r,changed,1066,'gpu-l40s-1')
        with self.assertRaises(ValueError):validate_allocation(r,e,0,'gpu-l40s-1')
        with self.assertRaises(ValueError):
            validate_allocation(dict(r,execution_backend='slurm_h100'),e,1066,'gpu-l40s-1')

    def test_previous_live_unknown_or_same_allocation_is_not_restarted(self):
        for state in ['RUNNING','PENDING','COMPLETING','SUSPENDED','']:
            with self.assertRaises(ValueError):
                check_previous_scheduler({'allocation':'122'},'123',lambda _:f'122|{state}\n' if state else '')
        with self.assertRaises(ValueError):
            check_previous_scheduler({'allocation':'123'},'123',lambda _:'123|FAILED\n')
        self.assertEqual(check_previous_scheduler({'allocation':'122'},'123',lambda _:'122|FAILED\n'),'FAILED')

    def test_runtime_preserves_native_hooks_without_h100_claim(self):
        r,_=self.fixture();env=runtime_environment(r,Path('/run/output/attempt-x'),Path('/run/receipts/configuration.json'))
        self.assertEqual(env['SEARCH_R1_EXECUTION_BACKEND'],'slurm_lumia')
        self.assertEqual(env['RECURSIVE_A_GUARD_ROUND_PROGRESS'],'1')
        self.assertEqual(env['RECURSIVE_LUMIA_PAIRED_EVALUATION'],'1')
        self.assertFalse(any('H100' in key for key in env))


if __name__ == '__main__':
    unittest.main()
