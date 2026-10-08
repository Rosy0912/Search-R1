"""Actual collect_outer orchestration on CPU; GPU/tensor interfaces are explicit spies.

Executes the candidate function body, actual2560 pure probability validator,
actual JSON evidence writes/hashes/closure and final manifest. Reconstruction of
GPU tensors and torch serialization are fixtures, not GPU or state qualification.
"""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

from formal_probability_v1 import digest, require, validate_outer
from formal_probability_proofs_v1 import file_record, manifest_records, validate_manifest_closure
from formal_protection_contract_v1 import MODEL
from test_formal_probability_v1 import trajectories, TOLERANCES, SOURCE


class ActualCollectOrchestrationCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.old_env = dict(os.environ); self.addCleanup(self.restore_env)
        os.environ.update(SEARCH_R1_BASELINE_COMMIT=SOURCE, SLURM_JOB_ID='9999',
            SEARCH_R1_BASELINE_RUN_ID='cpu-collect-scope-not-a-job', SEARCH_R1_BASELINE_CONFIG_SHA256='e'*64)
        self.turns = [['CPU declared observed generation']]
        (self.directory/'observed-turns.json').write_text(json.dumps(self.turns))
        def record(name):
            path = self.directory/name; path.write_text('explicit CPU tensor fixture '+name)
            return file_record(path)
        self.raws = [dict(rank=rank, serial=0, **record(f'raw-{rank}.pt')) for rank in range(8)]
        self.raw_receipts = [dict(rank=rank, serial=0, **record(f'raw-{rank}.json')) for rank in range(8)]
        self.refs = [dict(rank=rank, raw=record(f'ref-{rank}.pt'), receipt=record(f'ref-{rank}.json')) for rank in range(8)]
        self.records = trajectories()
        self.calls = []
        self.before_reference = lambda: None
        def reconstruct(batch, turns, directory, binding, source, outer):
            self.calls.append('reconstruct_GPU_interface_spy')
            return self.records, self.raws, self.raw_receipts
        def references(*args):
            self.calls.append('reference_GPU_interface_spy'); self.before_reference()
            return self.refs
        def save_json(value, path):
            Path(path).write_text(json.dumps(value, allow_nan=False))
        def save_tensor_fixture(value, path):
            self.calls.append('torch_serializer_CPU_fixture')
            # Explicit JSON fixture stored at a.pt name for closure control only.
            save_json(value, path)
            return file_record(path)
        def validate(rows, tolerances, pad):
            self.calls.append(('actual_validator_pad', pad))
            return validate_outer(rows, tolerances, pad)
        class Batch:
            def __init__(self, rank=None): self.rank=rank
            def __len__(self): return 2560 if self.rank is None else 320
            def chunk(self, chunks):
                assert chunks == 8
                return [Batch(rank) for rank in range(chunks)]
        self.batch = Batch()
        self.trainer = types.SimpleNamespace(global_steps=26,tokenizer=types.SimpleNamespace(pad_token_id=0,eos_token_id=99),
            config=types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(ref=types.SimpleNamespace(model_path=MODEL))))
        self.binding = dict(probability_identity=dict(pad_token_id=0,eos_token_id=99,tolerances=TOLERANCES,
            acceptance=dict(sha256='a'*64)), production_artifact=dict(sha256='b'*64))
        namespace = dict(os=os, require=require, reconstruct_actions=reconstruct,
            reconstruct_reference_records=references, atomic_json_save=save_json,
            atomic_torch_save=save_tensor_fixture, state_digest=digest,
            scored_batch_state=lambda batch:dict(rows=len(batch),rank=batch.rank),
            validate_outer=validate, actual_driver_boundary=lambda *args:dict(cursor=13312,RNG='post sampling fixture'),
            file_record=file_record,manifest_records=manifest_records,validate_manifest_closure=validate_manifest_closure)
        tree = ast.parse((Path(__file__).parent/'formal_probability_driver_v1.py').read_text())
        functions = [copy.deepcopy(row) for row in tree.body if isinstance(row,ast.FunctionDef)
                     and row.name in ('collect_outer','self_model_path')]
        module = ast.Module(body=functions,type_ignores=[]); ast.fix_missing_locations(module)
        exec(compile(module,'formal_probability_driver_v1.py','exec'),namespace)
        self.collect = namespace['collect_outer']

    def restore_env(self):
        os.environ.clear(); os.environ.update(self.old_env)

    def invoke(self):
        return self.collect(self.trainer,self.batch,self.turns,self.directory,self.binding)

    def test_full_collect_orchestration_success_uses_actual_bound_tokenizer(self):
        path,manifest = self.invoke()
        self.assertEqual(json.loads(path.read_text()),manifest)
        self.assertEqual(manifest['actual_driver_tokenizer_identity'],dict(pad_token_id=0,eos_token_id=99))
        self.assertEqual(manifest['checks']['trajectories'],2560)
        self.assertEqual([row['rows'] for row in manifest['chunks']],[320]*8)
        self.assertEqual(len(validate_manifest_closure(manifest,self.directory)),35)
        self.assertIn(('actual_validator_pad',0),self.calls)

    def test_actual_PAD_changed_refuses_before_reconstruct_or_manifest(self):
        self.trainer.tokenizer.pad_token_id=7
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls,[])
        self.assertFalse((self.directory/'actions.json').exists())
        self.assertFalse((self.directory/'before-adam-manifest.json').exists())

    def test_actual_EOS_changed_refuses_before_reconstruct_or_manifest(self):
        self.trainer.tokenizer.eos_token_id=8
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls,[])
        self.assertFalse((self.directory/'before-adam-manifest.json').exists())

    def test_full_collect_probability_failure_preserves_cause_without_admission(self):
        self.records[2559]['actions'][0]['full_old_log_probs']=[-1.,-3.]
        with self.assertRaises(ValueError): self.invoke()
        cause=json.loads((self.directory/'actions.json').read_text())
        self.assertEqual(cause['records'][2559]['actions'][0]['full_old_log_probs'],[-1.,-3.])
        self.assertTrue((self.directory/'scored-global-batch.pt').is_file())
        self.assertNotIn('reference_GPU_interface_spy',self.calls)
        self.assertFalse((self.directory/'before-adam-manifest.json').exists())

    def test_full_collect_reference_failure_preserves_cause_without_admission(self):
        def fail(): raise ValueError('CPU injected actual reference mismatch')
        self.before_reference=fail
        with self.assertRaises(ValueError): self.invoke()
        self.assertTrue((self.directory/'actions.json').is_file())
        self.assertFalse((self.directory/'before-adam-manifest.json').exists())

    def test_full_collect_changed_original_bytes_fail_closure_without_admission(self):
        self.before_reference=lambda:Path(self.raws[7]['path']).write_text('CPU injected changed raw bytes')
        with self.assertRaises(ValueError): self.invoke()
        self.assertIn('reference_GPU_interface_spy',self.calls)
        self.assertFalse((self.directory/'before-adam-manifest.json').exists())


if __name__=='__main__':
    unittest.main()
