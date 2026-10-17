import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import base_contracts as b
from .contracts import IDS_SHA, sha, validate_milestone


class BaseContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.template=dict(protocol='tree_seven_milestone_v1',group='original',outer=200,
            base=str(root),backend='slurm_lumia',canonical_tokenizer='/bound/base',
            training_config_sha256='a'*64,data_root='/bound/data',data_manifest_sha256='b'*64,
            final_content_identity='c'*64,ordered_ids_sha256=IDS_SHA,
            retriever_receipt='/bound/retriever.json',retriever_receipt_sha256='d'*64)
        self.pp=root/'reference.json';self.pp.write_text(json.dumps(self.template))
        self.plan={k:self.template[k] for k in b.INHERITED}
        self.plan.update(protocol='base_seven_repeats_v1',group='base',outer=0,
            baseline_model=b.MODEL,model_revision=b.REVISION,model='/bound/base',model_files=dict(b.FILES),
            checkpoint_identity=b.model_identity(),protocol_identity=b.protocol_identity(self.template),
            protocol_template_config_sha256='a'*64,template_plan=str(self.pp),template_plan_sha256=sha(self.pp),
            evaluation_code_commit='e'*40,evaluation_run_id='f'*32,repeat_index=1,repeat_count=3)
        self.plan['job']='zhangjunming-ttt-e-base-r1-'+b.model_identity()[:4]
        self.plan['output']=str(b.output_root(self.plan)/'output')

    def test_three_unique_outputs_one_model_not_milestone(self):
        paths=[]
        for index in (1,2,3):
            p=copy.deepcopy(self.plan);p['repeat_index']=index
            p['job']=p['job'].replace('-r1-','-r'+str(index)+'-');p['output']=str(b.output_root(p)/'output')
            paths.append(b.validate_base_plan(p,self.template))
        self.assertEqual(len(set(paths)),3)
        with self.assertRaises(ValueError):validate_milestone('base',0)

    def test_rejects_other_weights_and_science_or_data(self):
        for k,v in [('model_revision','wrong'),('model_files',{}),('model','/unbound'),
                    ('data_root','/other'),('final_content_identity','0'*64),('ordered_ids_sha256','0'*64),
                    ('retriever_receipt','other'),('protocol_template_config_sha256','0'*64),
                    ('checkpoint_identity','0'*64),('protocol_identity','0'*64)]:
            with self.subTest(k=k):
                p=copy.deepcopy(self.plan);p[k]=v
                with self.assertRaises(ValueError):b.validate_base_plan(p,self.template)

    def test_rejects_training_spoof_and_protocol_override(self):
        for field,value in [('training_config_sha256','x'),('checkpoint_manifest','x'),
                            ('inference_model','x'),('science_override',{}),('repeat',{})]:
            p=copy.deepcopy(self.plan);p[field]=value
            with self.assertRaises(ValueError):b.validate_base_plan(p,self.template)
        for field,value in [('group','Search-R1'),('outer',200),('protocol','tree_seven_milestone_v1')]:
            p=copy.deepcopy(self.plan);p[field]=value
            with self.assertRaises(ValueError):b.validate_base_plan(p,self.template)

    def test_rejects_replay_or_unrequested_repeat(self):
        for index in (0,4,True,'1'):
            p=copy.deepcopy(self.plan);p['repeat_index']=index
            with self.assertRaises(ValueError):b.validate_base_plan(p,self.template)
        for k,v in [('repeat_count',5),('output','/elsewhere/output'),('job','other'),
                    ('evaluation_run_id','not-uuid'),('evaluation_code_commit','main')]:
            p=copy.deepcopy(self.plan);p[k]=v
            with self.assertRaises(ValueError):b.validate_base_plan(p,self.template)

    def test_rejects_unrelated_reference_and_tamper_before_inference(self):
        for field,value in [('group','C1-root-only'),('outer',150),('backend','pj_pt'),('repeat',{})]:
            t=dict(self.template);t[field]=value
            with self.assertRaises(ValueError):b.validate_base_plan(self.plan,t)
        self.pp.write_text(self.pp.read_text()+' ')
        with patch.object(b,'validate_model_files') as check:
            with self.assertRaises(ValueError):b.inspect_base(self.plan)
            check.assert_not_called()

    def test_actual_weight_validation_failure_prevents_reference_or_gpu_init(self):
        with patch.object(b,'validate_model_files',side_effect=ValueError('actual bytes differ')) as check:
            with self.assertRaisesRegex(ValueError,'actual bytes differ'):b.inspect_base(self.plan)
            check.assert_called_once_with(self.plan)


if __name__=='__main__':unittest.main()
