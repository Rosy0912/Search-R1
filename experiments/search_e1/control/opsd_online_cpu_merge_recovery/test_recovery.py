"""CPU-only boundary tests; actual original merge function with tiny mock models."""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu_merge_recovery_fixture', HERE / 'recovery.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
WAIT_ROOT = Path(os.environ.get('OPSD_ONLINE_TEST_WAIT_ROOT', HERE.parent / 'opsd_online_wait_recovery')).resolve()
spec = importlib.util.spec_from_file_location('cpu_merge_wait_fixture', WAIT_ROOT / 'recovery.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
ORIGINAL_ROOT = Path(os.environ.get('OPSD_ONLINE_TEST_HELPER_ROOT', HERE.parent / 'opsd_online')).resolve()
sys.path.insert(0, str(ORIGINAL_ROOT))
import sampling


class StopHold(BaseException): pass


class OriginalMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.calls = []
        self.source = self.root / 'source'; self.source.mkdir(); (self.source / 'weight.safetensors').write_bytes(b'cpu-source')
        self.adapter = self.root / 'adapter'; self.adapter.mkdir(); (self.adapter / 'adapter_model.safetensors').write_bytes(b'lora')
        self.config = dict(paths=dict(model=str(self.source)), source_model_manifest=dict(sha256='s' * 64))
        self.checkpoint = dict(adapter=dict(path=str(self.adapter), files=[sampling.fileproof(self.adapter / 'adapter_model.safetensors')]),
                               optimizer_updates=11, all_rank_adapter_sha256=['a' * 64] * 8)
        self.ref = dict(path='actual-CP11-manifest.json', sha256='c' * 64)
        self.output = self.root / 'merged'; self.cuda_initialized = False; self.peft_device = 'cpu'
        self.torch = ModuleType('torch'); self.torch.bfloat16 = object()
        self.torch.cuda = SimpleNamespace(is_initialized=lambda: self.cuda_initialized, device_count=lambda: 0)
        case = self
        class Base:
            def parameters(self): return [SimpleNamespace(device=SimpleNamespace(type='cpu'))]
            def save_pretrained(self, output, **kwargs):
                case.calls.append(('save', str(output), kwargs)); Path(output).mkdir(); (Path(output) / 'model.safetensors').write_bytes(b'merged-bf16')
        class AutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                case.calls.append(('base', args, kwargs)); return Base()
        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                case.calls.append(('tokenizer', args, kwargs)); return cls()
            def save_pretrained(self, output): (Path(output) / 'tokenizer.json').write_text('{}')
        class PeftInstance:
            def __init__(self, base): self.base = base
            def parameters(self): return [SimpleNamespace(device=SimpleNamespace(type=case.peft_device))]
            def merge_and_unload(self, *args, **kwargs):
                case.calls.append(('merge', args, kwargs)); return self.base
        class PeftModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                case.calls.append(('peft', args, kwargs)); return PeftInstance(args[0])
        self.PeftModel = PeftModel
        self.transformers = ModuleType('transformers'); self.transformers.AutoModelForCausalLM = AutoModel; self.transformers.AutoTokenizer = AutoTokenizer
        self.peft = ModuleType('peft'); self.peft.PeftModel = PeftModel

    def execute(self):
        source = dict(path=str(self.source), files=[sampling.fileproof(self.source / 'weight.safetensors')])
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), \
             patch.dict(sys.modules, {'torch': self.torch, 'transformers': self.transformers, 'peft': self.peft}), \
             patch.object(sampling, 'bind_model_files', return_value=source):
            return r.original_merge_on_cpu(self.config, self.checkpoint, self.output, self.ref, sampling, self.torch, self.PeftModel)

    def test_calls_actual_original_merge_once_with_identical_bf16_peft_arguments(self):
        descriptor = inspect.getattr_static(self.PeftModel, 'from_pretrained')
        actual_function = sampling.merge_latest
        with patch.object(sampling, 'merge_latest', wraps=actual_function) as call:
            value, proof = self.execute()
            self.assertEqual(call.call_args.args, (self.config, self.checkpoint, self.output, self.ref)); self.assertEqual(call.call_count, 1)
        self.assertIs(inspect.getattr_static(self.PeftModel, 'from_pretrained'), descriptor)
        self.assertEqual(next(c for c in self.calls if c[0] == 'base')[2], dict(torch_dtype=self.torch.bfloat16, device_map=None))
        self.assertEqual(next(c for c in self.calls if c[0] == 'peft')[1][1], str(self.adapter))
        self.assertEqual(next(c for c in self.calls if c[0] == 'peft')[2], {})
        self.assertEqual(next(c for c in self.calls if c[0] == 'merge')[1:], ((), {}))
        self.assertEqual(next(c for c in self.calls if c[0] == 'save')[2], dict(safe_serialization=True))
        self.assertEqual(value['checkpoint'], self.ref); self.assertEqual(value['optimizer_updates'], 11)
        self.assertTrue(proof['original_function_called_once']); self.assertEqual(proof['before'], proof['after'])
        self.assertTrue(r.validate_result(w, self.config, self.checkpoint, self.ref, self.output, value))

    def test_actual_peft_non_cpu_parameter_refuses_before_save_and_restores_descriptor(self):
        self.peft_device = 'cuda'; descriptor = inspect.getattr_static(self.PeftModel, 'from_pretrained')
        with self.assertRaisesRegex(ValueError, 'entirely CPU'): self.execute()
        self.assertFalse(self.output.exists()); self.assertIs(inspect.getattr_static(self.PeftModel, 'from_pretrained'), descriptor)

    def test_cuda_initialized_boundary_refuses_before_original_call(self):
        self.cuda_initialized = True
        with self.assertRaisesRegex(ValueError, 'initialized/visible'): self.execute()
        self.assertEqual(self.calls, []); self.assertFalse(self.output.exists())

    def test_original_model_identity_unchanged_and_tampered_cache_rejected(self):
        value, _ = self.execute(); (self.output / 'model.safetensors').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'Merged cache bytes differ'):
            r.validate_result(w, self.config, self.checkpoint, self.ref, self.output, value)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.output = self.root / 'cache'
        self.identity = dict(pid=300, uid=1005, starttick=1234, run_id=w.RUN_ID)
        self.config = dict(paths=dict(base=str(self.root), python=sys.executable), source_model_manifest=dict(sha256='s' * 64))
        self.cp = dict(adapter=dict(path=str(self.root / 'adapter'), files=[]), optimizer_updates=11, all_rank_adapter_sha256=['a' * 64] * 8)
        self.ref = dict(path=str(self.root / 'checkpoint.json'), sha256='c' * 64)
        self.rc = 0; self.receipt = True; self.wait_count = 0; self.release_calls = 0; self.command = None
        self.gate_calls = []; self.bound_env = None
        self.controller = SimpleNamespace(config=self.config, config_ref=dict(path='actualconfig', sha256=w.CONFIG_SHA),
            run_id=w.RUN_ID, owner=dict(pid=200, uid=1005, starttick=100), known=[], active_jobs=[],
            gate=lambda phase, children=None: self.gate_calls.append(phase) or dict(accepted=True), spawn=self.spawn)
        def admit(*args, **kwargs): return dict(accepted=True)
        admit.require_owned_release = lambda *args, **kwargs: dict(accepted=True, actual_owned_absent=True)
        self.controller.admit = admit
        def release(*args): self.release_calls += 1; return dict(actual_free=True)
        self.frozen = SimpleNamespace(require_gpu_release=release)

    def spawn(self, command, log, phase, env):
        self.command = command; self.bound_env = env; self.assertEqual(phase, 'cpu-merge')
        self.assertEqual(env, dict(CUDA_VISIBLE_DEVICES=''))
        request_path = command[command.index('--request') + 1]; request_sha = command[command.index('--request-sha256') + 1]
        request, _ = w.checked_json(dict(path=request_path, sha256=request_sha))
        log = Path(log).open('x'); log.write('CPU child fixture\n'); log.flush(); self.addCleanup(lambda: None if log.closed else log.close())
        if self.rc == 0 and self.receipt:
            self.output.mkdir(); (self.output / 'model.safetensors').write_bytes(b'tiny CPU merged cache')
            files = [w.fileproof(self.output / 'model.safetensors')]
            result = dict(kind='original-PEFT-CPU-BF16-merge-latest', source_model_identity='s' * 64,
                checkpoint=self.ref, optimizer_updates=11, checkpoint_optimizer_updates=11,
                adapter_tensor_sha256='a' * 64, adapter=self.cp['adapter'], path=str(self.output), files=files,
                sha256=r.original_object_sha(files))
            device = dict(cuda_initialized=False, visible_cuda_devices=0, cuda_visible_devices='')
            w.exclusive_json(request['child_receipt'], dict(accepted=True, run_id=w.RUN_ID, config_sha256=w.CONFIG_SHA,
                owner=self.identity, request=dict(sha256=request_sha), device_proof=dict(before=device, after=device,
                    peft_parameters=[dict(parameter_tensors=1, device_types=['cpu'])],
                    merged_parameters=[dict(parameter_tensors=1, device_types=['cpu'])],
                    original_function_called_once=True, original_bf16_method_and_arguments_unchanged=True), result=result))
        case = self
        class Process:
            def poll(self): return case.rc
            def wait(self): case.wait_count += 1; return case.rc
        self.controller.known.append(self.identity); return Process(), log, self.identity

    def execute(self, sleeper=lambda _: None):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            return r.cpu_boundary(self.controller, self.frozen, w, {}, self.cp, self.output, self.ref, sleeper=sleeper)

    def test_cpu_spawn_forces_empty_cuda_and_actual_zero_readback_retains_cp11(self):
        value = self.execute()
        self.assertEqual(value['optimizer_updates'], 11); self.assertEqual(value['checkpoint'], self.ref)
        self.assertEqual(self.wait_count, 1); self.assertEqual(self.release_calls, 1)
        complete = json.loads(next((self.root / 'receipts' / 'cpu-merge-recovery').glob('*/parent-complete.json')).read_text())
        self.assertTrue(complete['accepted']); self.assertFalse(complete['parent_cpu']['torch_imported'])
        self.assertEqual(complete['actual_terminal']['path'].split('/')[-1], 'actual-terminal.json')
        self.assertEqual(self.command[1], '-B')

    def test_nonzero_preserves_actual_returncode_then_rejects_after_release(self):
        self.rc = 8
        with self.assertRaisesRegex(ValueError, 'nonzero'): self.execute()
        terminal = json.loads(next((self.root / 'receipts' / 'cpu-merge-recovery').glob('*/actual-terminal.json')).read_text())
        self.assertEqual(terminal['actual_wait_returncode'], 8); self.assertEqual(self.release_calls, 1)
        self.assertFalse(list((self.root / 'receipts' / 'cpu-merge-recovery').glob('*/parent-complete.json')))

    def test_rc_zero_missing_receipt_is_rejected_not_faked_from_exit(self):
        self.receipt = False
        with self.assertRaisesRegex(ValueError, 'without its completion receipt'): self.execute()
        self.assertEqual(self.wait_count, 1); self.assertEqual(self.release_calls, 1)

    def test_persistent_owned_release_holds_and_never_accepts(self):
        self.controller.admit.require_owned_release = lambda *args, **kwargs: dict(accepted=False)
        with self.assertRaises(StopHold): self.execute(sleeper=lambda _: (_ for _ in ()).throw(StopHold()))
        self.assertFalse(list((self.root / 'receipts' / 'cpu-merge-recovery').glob('*/parent-complete.json')))

    def test_parent_rejects_visible_cuda_without_spawning(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                r.cpu_boundary(self.controller, self.frozen, w, {}, self.cp, self.output, self.ref)
        self.assertIsNone(self.command)


class IdentityTests(unittest.TestCase):
    def test_wrong_immutable_config_rejects_before_parent_or_gpu_start(self):
        args = SimpleNamespace(resume=True, wait_root=str(WAIT_ROOT), wait_publication='unused', wait_publication_sha256='x',
                               config_sha256='wrong', run_id=w.RUN_ID)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch.object(r, 'load_wait', return_value=(w, {})):
            with self.assertRaisesRegex(ValueError, 'Immutable original config/run'): r.run_parent(args)

    def test_parent_never_allows_already_imported_torch(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch.dict(sys.modules, {'torch': ModuleType('torch')}):
            with self.assertRaisesRegex(ValueError, 'must not import Torch'): r.parent_cpu_only()

    def test_previous_parent_terminal_fresh_absence_and_preservation_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); preserved = w.exclusive_json(root / 'preserved.json', dict(accepted=True, run_id=w.RUN_ID, config_sha256=w.CONFIG_SHA))
            owner = dict(pid=660173, uid=1005, starttick=924349236)
            ref = w.exclusive_json(root / 'terminal.json', dict(schema='opsd-online-cpu-merge-parent-terminal-v1', accepted=True,
                run_id=w.RUN_ID, config_sha256=w.CONFIG_SHA, old_owner=owner, old_source_commit=r.WAIT_COMMIT,
                old_owned_processes=[owner], merge_preservation=preserved))
            proc = root / 'proc'; proc.mkdir()
            self.assertTrue(r.validate_previous(w, ref, proc)[1]['fresh_all_old_pids_absent'])
            (proc / '660173').mkdir()
            with self.assertRaisesRegex(ValueError, 'Previous owned PID present'): r.validate_previous(w, ref, proc)


if __name__ == '__main__': unittest.main()
