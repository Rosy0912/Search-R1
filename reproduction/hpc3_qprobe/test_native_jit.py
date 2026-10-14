import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid

from reproduction.common8 import native_ray
from reproduction.hpc3_qprobe import native_jit


class NodeJitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);(self.base/'logs').mkdir()
        self.run=dict(run_id=uuid.uuid4().hex,execution_site='hpc3',training_world_size=4)
        native_ray.prepare(self.run)

    def tearDown(self):
        shutil.rmtree(self.run['ray_tmp_dir']);self.temp.cleanup()

    def test_jit_paths_share_private_bound_node_root_and_preserve_output(self):
        env=native_jit.prepare(self.base,self.run);root=native_jit.verify(self.base,self.run)
        self.assertEqual(root,Path(self.run['ray_tmp_dir'])/'jit')
        self.assertEqual(env['TRITON_CACHE_DIR'],str(root/'triton'))
        self.assertEqual(env['TMPDIR'],str(root/'tmp'))
        self.assertFalse((self.base/'checkpoints').exists())
        with self.assertRaises(FileExistsError):native_jit.prepare(self.base,self.run)

    def test_mode_inode_and_symlink_tampering_rejected(self):
        native_jit.prepare(self.base,self.run);root=Path(self.run['ray_tmp_dir'])/'jit'
        os.chmod(root/'triton',0o770)
        with self.assertRaises(AssertionError):native_jit.verify(self.base,self.run)
        os.chmod(root/'triton',0o700);(root/'triton').rmdir();(root/'triton').symlink_to(root/'tmp')
        with self.assertRaises(AssertionError):native_jit.verify(self.base,self.run)


if __name__=='__main__':unittest.main()
