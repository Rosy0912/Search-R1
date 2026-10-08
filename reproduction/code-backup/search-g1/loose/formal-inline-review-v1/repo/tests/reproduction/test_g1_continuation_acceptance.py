"""Real CPU safetensor deltas and exact201..205 acceptance; no training claims."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from reproduction.a800.prepare_run import digest
from reproduction.a800.runtime import accept_five_updates


class ContinuationAcceptance(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        for n in ('data','logs','base','checkpoint200','checkpoint205'):(self.base/n).mkdir()
        counts={'nq':10,'trivia':9,'pop':9,'hotpot':9,'2wiki':9,'musique':9,'bamboogle':9}
        (self.base/'data/audit.json').write_text(json.dumps({'validation_source_counts':counts}))
        (self.base/'data/validation-64.parquet').write_bytes(b'bound CPU fixture')
        for name,value in [('base',0.),('checkpoint200',2.),('checkpoint205',3.)]:
            p=self.base/name;save_file({'weight':torch.tensor([value])},str(p/'weights.safetensors'))
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'weights.safetensors'}}))
        self.checkpoint=self.base/'checkpoint205'
        proof=dict(run_id='new',step=205,finite_all_saved_tensors=True,
            files_sha256={n:digest(self.checkpoint/n) for n in ('weights.safetensors','model.safetensors.index.json')})
        (self.checkpoint/'complete.json').write_text(json.dumps(proof))
        run=dict(run_id='new',completed_updates=200,acceptance_update=205,
            initial_model=str(self.base/'base'),actor_checkpoint=str(self.base/'checkpoint200'),
            actor_checkpoint_sha256={n:digest(self.base/'checkpoint200'/n) for n in ('weights.safetensors','model.safetensors.index.json')},
            initial_probe_version='new:200:uuid',input_sha256={n:digest(self.base/'data'/n) for n in ('audit.json','validation-64.parquet')})
        (self.base/'logs/run-manifest.json').write_text(json.dumps(run))
        metrics={n:.1 for n in ('critic/rewards/mean','actor/pg_loss','actor/entropy_loss','actor/kl_loss','actor/ppo_kl','actor/grad_norm')}
        self.records=[dict(step=i,probe_version='new:200:uuid',metrics=copy.deepcopy(metrics)) for i in range(201,206)]
        self.records[-1]['metrics'].update({'val/evaluated_count/'+k:v for k,v in counts.items()})
        self.env=patch.dict(os.environ,{'SEARCH_G1_BASE':str(self.base)});self.env.start();self.addCleanup(self.env.stop)

    def test_fresh205_delta_accepted(self):
        result=accept_five_updates(self.checkpoint,self.records)
        self.assertEqual(result['changed_parameters'],['weight'])
        self.assertTrue((self.base/'logs/five-updates-acceptance.json').exists())

    def test_rejects_stale_version_wrong_sequence_and_nonfinite(self):
        for kind in ('version','step','nonfinite','coverage'):
            records=copy.deepcopy(self.records)
            if kind=='version':records[0]['probe_version']='old:150:uuid'
            if kind=='step':records[0]['step']=1
            if kind=='nonfinite':records[1]['metrics']['actor/grad_norm']=float('nan')
            if kind=='coverage':records[-1]['metrics']['val/evaluated_count/nq']=9
            with self.subTest(kind=kind),self.assertRaises(AssertionError):accept_five_updates(self.checkpoint,records)
        self.assertFalse((self.base/'logs/five-updates-acceptance.json').exists())

    def test_unchanged_source200_rejected_even_if_different_from_base(self):
        save_file({'weight':torch.tensor([3.])},str(self.base/'checkpoint200/weights.safetensors'))
        # A legitimate unchanged-parameter source is still rejected even though
        # both source and output differ from the original KL base.
        p=self.base/'logs/run-manifest.json';run=json.loads(p.read_text())
        run['actor_checkpoint_sha256']['weights.safetensors']=digest(self.base/'checkpoint200/weights.safetensors')
        p.write_text(json.dumps(run))
        with self.assertRaisesRegex(AssertionError,'no parameter update'):accept_five_updates(self.checkpoint,self.records)

    def test_source200_tamper_rejected(self):
        save_file({'weight':torch.tensor([4.])},str(self.base/'checkpoint200/weights.safetensors'))
        with self.assertRaisesRegex(AssertionError,'Source checkpoint changed'):accept_five_updates(self.checkpoint,self.records)

    def test_checkpoint_tamper_rejected(self):
        save_file({'weight':torch.tensor([4.])},str(self.checkpoint/'weights.safetensors'))
        with self.assertRaisesRegex(AssertionError,'checkpoint changed'):accept_five_updates(self.checkpoint,self.records)


if __name__=='__main__':unittest.main()
