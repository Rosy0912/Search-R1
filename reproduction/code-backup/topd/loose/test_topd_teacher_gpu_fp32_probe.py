import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('fp32probe',Path(__file__).with_name('topd-teacher-gpu-fp32-probe.py'));probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class ReleaseTests(unittest.TestCase):
    def fixture(self,missing_api=False,remaining=0,buffer_cuda=False):
        calls=[];memory={'allocated':32*1024*1024,'reserved':32*1024*1024}
        param=SimpleNamespace(device=SimpleNamespace(type='cuda'),dtype='torch.float32');buffer=SimpleNamespace(device=SimpleNamespace(type='cuda'))
        class Model:
            def to(self,device):
                calls.append('model_to_'+device);param.device.type=device
                if not buffer_cuda:buffer.device.type=device
                return self
            def parameters(self):return [param]
            def buffers(self):return [buffer]
        def clear():calls.append('workspace_clear');memory.update(allocated=remaining,reserved=remaining)
        torch=SimpleNamespace(device=lambda value:SimpleNamespace(type=value),_C=SimpleNamespace() if missing_api else SimpleNamespace(_cuda_clearCublasWorkspaces=clear),
            cuda=SimpleNamespace(synchronize=lambda:calls.append('synchronize'),empty_cache=lambda:calls.append('empty_cache'),memory_allocated=lambda:memory['allocated'],memory_reserved=lambda:memory['reserved']))
        return torch,SimpleNamespace(model=Model(),device=SimpleNamespace(type='cuda')),calls,[]
    def invoke(self,fixture,roundtrip='same'):
        torch,teacher,calls,records=fixture
        with patch.object(probe,'gpu_snapshot',return_value={'uuid':'GPU-owned'}),patch.object(probe,'cpu_parameter_sha',return_value=roundtrip):
            result=probe.release_owned_scorer(torch,teacher,'same',records.append,'GPU-owned',{})
        return result
    def test_process_workspace_cleared_before_empty_cache_and_zero_gate(self):
        f=self.fixture();result=self.invoke(f);self.assertEqual(result['allocated_bytes'],0);self.assertEqual(result['reserved_bytes'],0)
        calls=f[2];self.assertLess(calls.index('model_to_cpu'),calls.index('workspace_clear'));self.assertLess(calls.index('workspace_clear'),calls.index('empty_cache'))
        self.assertEqual(f[3][0]['kind'],'workspace_before_clear');self.assertEqual(f[3][0]['allocated_bytes'],33554432);self.assertEqual(f[3][1]['kind'],'workspace_after_clear')
    def test_missing_private_api_fail_closed(self):
        f=self.fixture(missing_api=True)
        with self.assertRaisesRegex(RuntimeError,'lacks private'):self.invoke(f)
        self.assertNotIn('empty_cache',f[2]);self.assertEqual(len(f[3]),1)
    def test_residual32mb_not_exempted(self):
        f=self.fixture(remaining=33554432)
        with self.assertRaisesRegex(RuntimeError,'zero gate retained'):self.invoke(f)
        self.assertEqual(f[3][-1]['allocated_bytes'],33554432)
    def test_cuda_buffer_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'buffers remain'):self.invoke(self.fixture(buffer_cuda=True))
    def test_hash_drift_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'parameters changed'):self.invoke(self.fixture(),roundtrip='different')
    def test_non_fp32_parameters_rejected(self):
        f=self.fixture();parameter=f[1].model.parameters()[0];parameter.dtype='torch.bfloat16'
        with self.assertRaisesRegex(RuntimeError,'parameters changed'):self.invoke(f)
if __name__=='__main__':unittest.main()
