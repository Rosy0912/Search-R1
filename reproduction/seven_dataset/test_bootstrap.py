"""Regress the real pre-binding CUDA import without requiring CUDA locally."""
import os
import subprocess
import sys
import unittest


class LazySetupTests(unittest.TestCase):
    def test_worker_setup_does_not_import_cuda_dependencies(self):
        code='''
import sys
from reproduction.seven_dataset.runner import worker_setup
worker_setup()
worker_setup()
assert not any(n in sys.modules for n in ['torch','vllm','verl.workers.fsdp_workers','reproduction.seven_dataset.worker'])
from reproduction.seven_dataset.bootstrap import _Finder
assert sum(isinstance(v,_Finder) for v in sys.meta_path)==1
'''
        env=dict(os.environ,TREE_SEVEN_EVAL_MODE='frozen_milestone_v1',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
