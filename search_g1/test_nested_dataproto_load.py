"""Regression for PyTorch 2.4 nested DataProto checkpoint deserialization."""
import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tensordict import TensorDict
from verl.protocol import DataProto


class NestedDataProtoLoadTest(unittest.TestCase):
    def proto(self):
        return DataProto(batch=TensorDict({'tokens':torch.arange(6).reshape(2,3)},batch_size=[2]),
                         non_tensor_batch={'id':np.array(['a','b'],dtype=object)},meta_info={'scope':'fixture'})

    def test_nested_archive_honors_explicit_cpu_and_remaining_tensor(self):
        value={'pending_source':self.proto(),'after':torch.tensor([7.0]),'second':self.proto()}
        stream=io.BytesIO();torch.save(value,stream);stream.seek(0)
        with patch('torch.cuda.is_available',return_value=True):
            restored=torch.load(stream,map_location='cpu',weights_only=False)
        self.assertTrue(torch.equal(restored['pending_source'].batch['tokens'],value['pending_source'].batch['tokens']))
        self.assertTrue(torch.equal(restored['after'],value['after']))
        self.assertEqual(restored['second'].batch['tokens'].device.type,'cpu')
        self.assertFalse(hasattr(torch._utils._thread_local_state,'map_location'))

    def test_direct_pickle_roundtrip(self):
        import pickle
        value=self.proto();restored=pickle.loads(pickle.dumps(value))
        self.assertTrue(torch.equal(restored.batch['tokens'],value.batch['tokens']))
        self.assertEqual(restored.non_tensor_batch['id'].tolist(),['a','b'])
        self.assertFalse(hasattr(torch._utils._thread_local_state,'map_location'))

    def test_inner_error_propagates_and_preserves_outer_mapping(self):
        context=torch._utils._thread_local_state;context.map_location='cpu'
        try:
            with self.assertRaises(Exception):
                self.proto().__setstate__((b'not a torch archive',{},{}))
            self.assertEqual(context.map_location,'cpu')
        finally:
            del context.map_location


if __name__=='__main__':unittest.main()
