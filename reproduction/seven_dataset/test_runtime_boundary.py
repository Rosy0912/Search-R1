"""Physical scheduler alias and local Ray boundary regression."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from search_r1.baselines.local_runtime import check_local_runtime

class Options(dict):
    __getattr__ = dict.__getitem__

class RuntimeBoundaryTests(unittest.TestCase):
    def check(self, node='node4', host='gpu-4090-1', uid=1066, ray='10.0.0.4'):
        cfg=SimpleNamespace(baselines=Options(local_runtime_only=True,local_runtime_host=host),
                            retriever=SimpleNamespace(url='http://127.0.0.1:29147/retrieve'))
        env={'SEARCH_R1_EXECUTION_BACKEND':'slurm_lumia','SLURM_JOB_ID':'139020',
             'SLURM_JOB_NODELIST':node,'RAY_ADDRESS':'127.0.0.1:16381'}
        with patch.dict('os.environ',env,clear=True),patch('os.getuid',return_value=uid),patch('socket.gethostname',return_value=host),patch('socket.getaddrinfo',return_value=[(2,1,6,'',('10.0.0.4',0))]):
            check_local_runtime(cfg,[{'Alive':True,'NodeManagerAddress':ray}])

    def test_registered_node4_physical_alias(self):self.check()
    def test_other_physical_host_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical host'):self.check(host='gpu-4090-3')
    def test_unregistered_node_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical host'):self.check(node='node99')
    def test_wrong_owner_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical host'):self.check(uid=1000)
    def test_foreign_ray_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'remote'):self.check(ray='10.0.0.99')
    def test_existing_node10_alias_unchanged(self):self.check(node='node10',host='gpu-l40s-1')

if __name__=='__main__':unittest.main()
