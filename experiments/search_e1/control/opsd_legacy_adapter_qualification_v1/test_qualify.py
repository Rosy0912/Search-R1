"""Two narrow rejection fixtures; no production files, Torch, model or GPU."""
import unittest
from types import SimpleNamespace
import qualify


class QualificationRejections(unittest.TestCase):
    def test_wrong_common_checkpoint_rejected_before_tensor_read(self):
        inputs = dict(schema='optional-L-adapter-only-inputs-v1',
            reader=dict(path='/fixed/reader.py', sha256=qualify.READER_SHA),
            qualification=dict(sha256=qualify.QUALIFICATION_SHA),
            qualification_source=dict(sha256=qualify.QUALIFICATION_SOURCE_SHA))
        inputs.update({k: dict(path=p, sha256=s) for k, (p, s) in qualify.PINS.items()})
        inputs['baseline']['path'] = '/different/CP0/manifest.json'
        with self.assertRaisesRegex(ValueError, 'baseline'):
            qualify.pinned_inputs(inputs)

    def test_one_non_fp32_tensor_rejected(self):
        tensors = {str(k): SimpleNamespace(dtype='float32', device=SimpleNamespace(type='cpu')) for k in range(504)}
        tensors['0'].dtype = 'bfloat16'
        fake_torch = SimpleNamespace(float32='float32', isfinite=lambda t: self.fail('Wrong dtype must reject before finite access'))
        with self.assertRaisesRegex(ValueError, 'finite CPU FP32'):
            qualify.validate_tensors(tensors, fake_torch)


if __name__ == '__main__':
    unittest.main()
