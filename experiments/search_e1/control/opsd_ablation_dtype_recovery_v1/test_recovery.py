"""Two focused CPU regressions; no model generation, CUDA, or production job."""
import hashlib
import json
from pathlib import Path
import unittest
import torch
from controller import rewrite
from recovery_common import same_ref
from trainer_launcher import adapter_values, repaired_setter


def digest(values):
    h = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)], separators=(',', ':')).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class TinyAdapters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(252)])
        for layer in self.layers:
            for name in ('lora_A', 'lora_B'):
                setattr(layer, name, torch.nn.ModuleDict({
                    'default': torch.nn.Linear(1, 1, bias=False, dtype=torch.float32),
                    'outer_old': torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)}))


class RecoveryTests(unittest.TestCase):
    def test_exact_original_torchrun_only_rewrite_preserves_options(self):
        cfg = {'paths': {'python': '/env/python'}}
        ref = {'path': '/old/config.json', 'sha256': 'a' * 64}
        self.assertTrue(same_ref(dict(ref, bytes=987, inode=1234), ref))
        self.assertFalse(same_ref(dict(ref, sha256='d' * 64), ref))
        self.assertFalse(same_ref(dict(ref, path='/another/config.json'), ref))
        cmd = ['/env/python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
               '/old/helper/trainer.py', '--config', ref['path'], '--config-sha256', ref['sha256'],
               '--outer-manifest', '/old/outer.json', '--outer-manifest-sha256', 'b' * 64,
               '--resume-checkpoint', '/old/checkpoint.json', '--resume-checkpoint-sha256', 'c' * 64]
        result, changed = rewrite(cmd, 'training-running', cfg, ref, Path('/old/helper'), Path('/repair/trainer_launcher.py'))
        self.assertTrue(changed)
        self.assertEqual(result[:5] + result[6:], cmd[:5] + cmd[6:])
        self.assertEqual(rewrite(['python', '/old/helper/sampling.py'], 'sampling-running', cfg, ref, Path('/old/helper'), Path('/repair/trainer_launcher.py')),
                         (['python', '/old/helper/sampling.py'], False))
        for index, wrong in [(0, '/wrong/python'), (4, '--nproc_per_node=4'), (5, '/other/trainer.py'), (9, 'd' * 64)]:
            bad = list(cmd); bad[index] = wrong
            with self.assertRaises(ValueError):
                rewrite(bad, 'training-running', cfg, ref, Path('/old/helper'), Path('/repair/trainer_launcher.py'))

    def test_before_set_fp32_exact_current_unchanged_wrong_identity_rejected(self):
        model = TinyAdapters()
        weights = {n: torch.full_like(p, .0001234, dtype=torch.float32) for n, p in adapter_values(model, 'outer_old').items()}
        expected = digest(weights)
        current = digest(adapter_values(model, 'default'))
        sentinel = object(); calls = []; records = []
        def native(m, w, *args, **kwargs):
            calls.append(kwargs.get('adapter_name', args[0] if args else 'default'))
            selected = adapter_values(m, calls[-1])
            with torch.no_grad():
                for name, parameter in selected.items():
                    parameter.copy_(w[name])
            return sentinel
        native(model, weights, adapter_name='outer_old')
        self.assertNotEqual(digest(adapter_values(model, 'outer_old')), expected)
        calls.clear()
        wrapper = repaired_setter(native, digest, expected, records.append)
        self.assertIs(wrapper(model, weights, adapter_name='outer_old'), sentinel)
        self.assertEqual(digest(adapter_values(model, 'outer_old')), expected)
        self.assertEqual(digest(adapter_values(model, 'default')), current)
        self.assertEqual(calls, ['outer_old'])
        self.assertTrue(records[0]['promotion_before_original_setter'])
        calls.clear()
        bad = dict(weights); key = next(iter(bad)); bad[key] = bad[key] + 1.
        with self.assertRaises(ValueError):
            wrapper(model, bad, adapter_name='outer_old')
        self.assertEqual(calls, [])
        # Default setter remains an unmodified passthrough, including return type.
        current_values = {n: p.detach().clone() for n, p in adapter_values(model, 'default').items()}
        self.assertIs(wrapper(model, current_values), sentinel)
        self.assertEqual(calls, ['default'])
        self.assertEqual(len(records), 1)


if __name__ == '__main__':
    unittest.main()
