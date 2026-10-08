"""Real synthetic CPU bytes and FD replacement counterexamples; no real assets."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from search_r1.baselines.probability_probe_capture import file_sha
from search_r1.baselines import probability_probe_io as I


class IOTests(unittest.TestCase):
    def files(self,directory):
        path=Path(directory)/'expected.pt';other=Path(directory)/'other.pt'
        torch.save({'value':torch.tensor([-1.])},path);torch.save({'value':torch.tensor([-99.])},other)
        return path,other,file_sha(path)

    def test_real_tensor_same_descriptor_load_matches_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,sha=self.files(tmp)
            self.assertEqual(I.load_tensor_verified(path,sha)['value'].tolist(),[-1.])

    def test_wrong_initial_bytes_refused_before_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,_=self.files(tmp)
            with patch.object(I.torch,'load') as reader,self.assertRaisesRegex(ValueError,'descriptor SHA'):
                I.load_tensor_verified(path,'0'*64)
            reader.assert_not_called()

    def test_atomic_path_swap_restore_cannot_load_unbound_other_inode(self):
        original=torch.load;actual_loaded=[]
        with tempfile.TemporaryDirectory() as tmp:
            path,other,sha=self.files(tmp);backup=Path(tmp)/'held.pt'
            def malicious_load(stream,**kwargs):
                self.assertFalse(isinstance(stream,(str,Path)))
                path.rename(backup);other.rename(path)
                try:
                    value=original(stream,**kwargs);actual_loaded.extend(value['value'].tolist());return value
                finally:path.rename(other);backup.rename(path)
            with patch.object(I.torch,'load',malicious_load),self.assertRaisesRegex(ValueError,'descriptor consumption'):
                I.load_tensor_verified(path,sha)
            self.assertEqual(actual_loaded,[-1.]) # Held FD never switched to B.
            self.assertEqual(file_sha(path),sha)

    def test_inplace_alternate_bytes_restore_refused_even_both_hashes_would_pass(self):
        original=torch.load;actual_loaded=[]
        with tempfile.TemporaryDirectory() as tmp:
            path,other,sha=self.files(tmp);expected=path.read_bytes()
            def malicious_load(stream,**kwargs):
                path.write_bytes(other.read_bytes())
                try:
                    value=original(stream,**kwargs);actual_loaded.extend(value['value'].tolist());return value
                finally:path.write_bytes(expected)
            with patch.object(I.torch,'load',malicious_load),self.assertRaisesRegex(ValueError,'descriptor consumption'):
                I.load_tensor_verified(path,sha)
            self.assertEqual(actual_loaded,[-99.]);self.assertEqual(file_sha(path),sha)

    def test_path_replacement_retained_after_consumer_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,other,sha=self.files(tmp)
            with self.assertRaisesRegex(ValueError,'descriptor consumption'):
                with I.bound_descriptor(path,sha):os.replace(other,path)

    def test_symlink_frozen_asset_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,sha=self.files(tmp);alias=Path(tmp)/'alias';alias.symlink_to(path)
            with self.assertRaisesRegex(ValueError,'symlink'):I.load_tensor_verified(alias,sha)

    def test_consumer_exception_does_not_publish_loaded_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,sha=self.files(tmp)
            with self.assertRaisesRegex(RuntimeError,'consumer'):
                with I.bound_descriptor(path,sha):raise RuntimeError('consumer failure')

    @contextmanager
    def synthetic_safetensors(self,descriptor_path,**kwargs):
        self.assertTrue(descriptor_path.startswith('/proc/self/fd/'))
        # Works on CPU hosts without /proc too: use the actual held FD number.
        fd=int(descriptor_path.rsplit('/',1)[1])
        with os.fdopen(os.dup(fd),'rb') as stream:loaded=torch.load(stream,map_location='cpu',weights_only=False)
        class Shard:
            def get_tensor(self,key):return loaded[key]
        yield Shard()

    def test_safetensors_reader_receives_same_verified_inode_fd_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,sha=self.files(tmp)
            with I.safetensor_verified(path,sha,self.synthetic_safetensors) as shard:
                self.assertEqual(shard.get_tensor('value').tolist(),[-1.])

    def test_safetensors_path_swap_restore_cannot_change_expected_tensor(self):
        observed=[]
        with tempfile.TemporaryDirectory() as tmp:
            path,other,sha=self.files(tmp);backup=Path(tmp)/'backup'
            @contextmanager
            def switched_reader(descriptor_path,**kwargs):
                path.rename(backup);other.rename(path)
                try:
                    with self.synthetic_safetensors(descriptor_path,**kwargs) as shard:
                        observed.extend(shard.get_tensor('value').tolist());yield shard
                finally:path.rename(other);backup.rename(path)
            with self.assertRaisesRegex(ValueError,'descriptor consumption'):
                with I.safetensor_verified(path,sha,switched_reader):pass
            self.assertEqual(observed,[-1.]);self.assertEqual(file_sha(path),sha)


if __name__=='__main__':unittest.main()
