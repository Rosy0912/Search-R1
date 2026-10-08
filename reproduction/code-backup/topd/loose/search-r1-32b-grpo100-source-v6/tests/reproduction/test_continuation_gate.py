"""Exercise actual continuation gate with real tiny serialized CPU checkpoints."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from safetensors.numpy import save_file

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('continuation_gate',ROOT/'reproduction/a800/verify_continuation.py')
gate=importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
PARAM='model.layers.0.self_attn.q_proj.weight'


class ContinuationGate(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.base=self.root/'base'; self.source=self.root/'global_step_100'
        self.target=self.root/'new/actor/global_step_105'
        for path,value in ((self.base,1),(self.source,2),(self.target,3)):
            path.mkdir(parents=True)
            save_file({PARAM:np.full((2,2),value,dtype=np.float32)},str(path/'model.safetensors'))
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{PARAM:'model.safetensors'}}))
            (path/'config.json').write_text('{}')
        self.log=self.root/'segment.log'
        self.log.write_text('SEARCH_R1_SEGMENT fresh\n'+''.join(
            f'step:{i} - critic/rewards/mean:0.400 - actor/lr:0.000\n' for i in range(101,106)))
        def hashes(path):
            return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
        self.manifest=dict(completed_updates=100,optimizer_reset=True,actor_checkpoint=str(self.source),
                           reference_model=str(self.base),acceptance_step=105,run_started=time.time()-10,
                           train_log=str(self.log),checkpoint_dir=str(self.root/'new'),run_id='fresh',
                           source_archive_sha256='f5b9e0d987a7fcb352b1614e9ce4c6b5df1c9c78f3fc83a4fbe247b9f81f7f6a',
                           provenance_verified=True,actor_file_hashes=hashes(self.source),reference_file_hashes=hashes(self.base))

    def tearDown(self): self.temp.cleanup()

    def test_five_real_updates_and_delta_pass(self):
        result=gate.verify(self.manifest)
        self.assertEqual(result['accepted_updates'],5)
        self.assertEqual(result['changed_elements'],4)

    def test_stale_log_header_and_old_metrics_rejected(self):
        for text in (self.log.read_text().replace('fresh','old'),
                     self.log.read_text()+'step:100 - critic/rewards/mean:0.400\n'):
            self.log.write_text(text)
            with self.assertRaises(AssertionError): gate.verify(self.manifest)

    def test_source_or_reference_change_rejected(self):
        for path in (self.source,self.base):
            file=path/'config.json'; original=file.read_text();file.write_text('{"changed":true}')
            with self.assertRaises(AssertionError): gate.verify(self.manifest)
            file.write_text(original)

    def test_missing_update_duplicate_and_nan_rejected(self):
        original=self.log.read_text()
        for text in (original.replace('step:103','missing:103'),
                     original+'step:103 - critic/rewards/mean:0.500\n',original.replace('0.400','nan')):
            self.log.write_text(text)
            with self.assertRaises(AssertionError): gate.verify(self.manifest)

    def test_unchanged_and_nonfinite_weights_rejected(self):
        for value in (2,float('nan')):
            save_file({PARAM:np.full((2,2),value,dtype=np.float32)},str(self.target/'model.safetensors'))
            with self.assertRaises(AssertionError): gate.verify(self.manifest)


if __name__=='__main__': unittest.main()
