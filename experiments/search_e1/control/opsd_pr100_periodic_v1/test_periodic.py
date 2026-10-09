"""Five focused CPU checks for the new boundary/template contract only."""
import json
from pathlib import Path
import tempfile
import unittest
import prepare
import prepare_pipeline
import supervisor
from utils import fileproof,Hold

class PeriodicContract(unittest.TestCase):
    def test_exact_ten_boundaries(self):
        self.assertEqual(prepare.BOUNDARIES,list(range(10,101,10)))
        self.assertEqual(prepare.scientific()['versions'],['raw_bf16'])
        self.assertEqual(prepare.scientific()['total_primary_slots'],512)
        self.assertEqual(prepare.scientific()['repetition_penalty'],1.05)
    def binding(self,n):
        template={'completed_outers':n,'training':{'config':{'path':'config','sha256':'same100'},'terminal':None,'boundary_checkpoint':None,'boundary':None},
            'scientific':prepare.scientific(),'gates':{'memory_observer':{'path':self.binding_path}}}
        status={'completed_outers':n,'phase':'TRAIN_COMPLETE' if n==100 else 'TRAIN_BOUNDARY','latest_checkpoint':{'path':str(n),'sha256':str(n)}}
        return template,status
    def test_bind10_raw_not_initial(self):
        with tempfile.TemporaryDirectory() as d:
            self.binding_path=str(Path(d)/'observer.json');Path(self.binding_path).write_text('{}\n')
            template,status=self.binding(10);actual=supervisor.bind_evaluation(template,{'path':'real0','sha256':'wait0'},status,{'path':'boundary10','sha256':'bound'})
            self.assertEqual(actual['training']['boundary_checkpoint'],status['latest_checkpoint']);self.assertIsNone(template['training']['boundary_checkpoint'])
            self.assertEqual(actual['training']['config']['sha256'],'same100')
    def test_bind100_final_same_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            self.binding_path=str(Path(d)/'observer.json');Path(self.binding_path).write_text('{}\n')
            template,status=self.binding(100);actual=supervisor.bind_evaluation(template,{'path':'real0','sha256':'wait0'},status,{'path':'boundary100','sha256':'bound'})
            self.assertEqual(actual['scientific'],prepare.scientific());self.assertEqual(actual['completed_outers'],100)
    def test_wrong_boundary_rejected(self):
        self.binding_path='/not/read';template,status=self.binding(10);status['completed_outers']=20
        with self.assertRaises(Hold):supervisor.bind_evaluation(template,{},status,{})
    def test_templates_independent_and_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);manifest=p/'manifest.json';manifest.write_text(json.dumps({'files':{'memory_observer.py':{'sha256':'observer','bytes':1}}})+'\n')
            train={'schema':'opsd-pr100-config-v1','variant':'PR','run_id':'a'*32,'study_id':'study','scientific':{'outer_iterations':100,'component_clip':10},
                'helper_source_manifest':fileproof(manifest),'owner':{'uid':1005,'gpu_uuids':list(range(8))},
                'paths':{'base':str(p/'training'),'output_root':str(p/'training/runs'),'helper_root':str(p/'training-source'),'python':'python','retriever_url':'unchanged'},
                'resources':{'shared_gpu_lease_paths':['a','b','c','d','e'],'study_lease_path':'study.lock'},
                'gates':{'retriever_admission':{'path':'oracle','sha256':'oracle'},'memory_binding_path':str(p/'binding')}}
            path=p/'training.json';path.write_text(json.dumps(train)+'\n')
            result=prepare_pipeline.make(fileproof(path),p/'periodic',p/'eval-source',p/'pipeline-template.json')
            value=json.loads(Path(result['pipeline_template']['path']).read_text())
            self.assertFalse(result['ready']);self.assertFalse(result['gpu_started']);self.assertEqual(value['state'],'NOT_READY')
            configs=[json.loads(Path(x['template']['path']).read_text()) for x in value['evaluations']]
            self.assertEqual(len({x['run_id'] for x in configs}),10)
            for x in configs:prepare.validate_config(x);self.assertIsNone(x['training']['boundary_checkpoint'])

if __name__=='__main__':unittest.main()
