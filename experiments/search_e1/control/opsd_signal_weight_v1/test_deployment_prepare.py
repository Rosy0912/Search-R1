"""Tiny CPU fixtures for immutable four-group deployment; no production assets."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PACKAGE='_opsd_signal_deployment_fixtures'
package=types.ModuleType(PACKAGE);package.__path__=[str(HERE)];sys.modules[PACKAGE]=package
deploy=importlib.import_module(PACKAGE+'.deployment_prepare')
cp=importlib.import_module(PACKAGE+'.checkpoint')


class DeploymentPreparation(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)/'remote-study'
        self.root.mkdir();self.helper=self.root/'source'/'fixed-commit'/'opsd_signal_weight_v1';self.helper.mkdir(parents=True)
        source=self.helper/'synthetic.py';source.write_text('"""Only a source-identity fixture."""\n')
        self.helper_manifest=cp.exclusive_json(self.helper/'source-manifest.json',{'files':{'synthetic.py':cp.fileproof(source)}})
        fixtures=self.root/'fixtures';fixtures.mkdir()
        payloads=[]
        for index in range(12):
            p=fixtures/('source-'+str(index));p.write_text('tiny synthetic source identity '+str(index));payloads.append(cp.fileproof(p))
        model=cp.exclusive_json(fixtures/'source-manifest.json',dict(accepted=True,source_grpo_step=100,files=payloads))
        initial=cp.exclusive_json(fixtures/'initial-manifest.json',dict(accepted=True,tensors=504,zero_B_tensors=252,
            dtype='float32',adapter_tensor_sha256=deploy.INITIAL_TENSOR_SHA,source_model_identity=model['sha256']))
        cohorts=cp.exclusive_json(fixtures/'cohorts.json',dict(outer_count=25,questions_per_outer=512,
            selected_unique_question_count=12800,validation_id_overlap=0,validation_exact_question_text_overlap=0))
        runtime=cp.exclusive_json(fixtures/'runtime.json',{'fixture_only':True})
        train=fixtures/'train';train.write_text('tiny train identity\n');validation=fixtures/'validation';validation.write_text('tiny validation identity\n')
        self.assets_value=dict(schema='opsd-signal-assets-v1',accepted=True,runtime_files=305,cuda_initialized=False,
            local_large_product_bytes=0,source_model_manifest=model,runtime_manifest=runtime,
            common_initial_adapter_manifest=initial,common_cohort_manifest=cohorts,
            train_identity=cp.fileproof(train),validation_identity=cp.fileproof(validation))
        self.assets=cp.exclusive_json(fixtures/'assets.json',self.assets_value)
        self.commit='a'*40
        self.pub=cp.exclusive_json(fixtures/'publication.json',dict(accepted=True,branch='search-e1',
            source_commit=self.commit,remote_readback_sha=self.commit,source_manifest_sha256=self.helper_manifest['sha256']))
        goal=fixtures/'goal.md';goal.write_text('Explicit four-group25-outer execution; four-card accumulation8.\n');self.goal=cp.fileproof(goal)
        self.resources=dict(accepted=True,cluster='HKUST',gpu_uuids=list(deploy.GPU_UUIDS),physical_gpus=[4,5,6,7],
            uid=1001,checked_unix=12345,scope='Synthetic lifecycle fixture, never a GPU acceptance')
        self.stack=ExitStack();self.addCleanup(self.stack.close);self.addCleanup(self.temporary.cleanup)
        self.stack.enter_context(patch.object(deploy,'STUDY',str(self.root)))
        self.stack.enter_context(patch.object(deploy.os,'getuid',return_value=1001))
        self.stack.enter_context(patch.dict(deploy.os.environ,{'CUDA_VISIBLE_DEVICES':''}))
        self.stack.enter_context(patch.object(deploy,'sys',types.SimpleNamespace(modules={})))
        self.fresh=self.stack.enter_context(patch.object(deploy,'fresh_resources',return_value=self.resources))
        self.runtime=self.stack.enter_context(patch.object(deploy,'verify_runtime',return_value={'fixture_only':True}))
        self.common=self.stack.enter_context(patch.object(deploy,'verify_common',return_value={'fixture_only':True}))
        self.oracle=self.stack.enter_context(patch.object(deploy.gates,'retriever_admission',return_value={'owner':{'pid':1749337}}))
        storage=deploy.gates.storage_admission
        self.stack.enter_context(patch.object(deploy.gates,'storage_admission',side_effect=lambda c,r:
            storage(c,r,disk_reader=lambda p:types.SimpleNamespace(free=200<<30))))

    def invoke(self):
        return deploy.prepare_deployment(self.assets,self.pub,self.commit,str(self.helper),self.goal,str(self.root))

    def test_four_immutable_configs_have_same_science_remote_outputs_and_reused_ids(self):
        first=self.invoke();self.assertTrue(first['prepared']);self.assertFalse(first['already_prepared'])
        self.assertEqual([g['variant'] for g in first['groups']],list(deploy.VARIANTS));ids=[]
        frozen={}
        for group in first['groups']:
            config=json.loads(Path(group['config']['path']).read_text());ids.append(config['run_id'])
            self.assertEqual(config['scientific']['gradient_accumulation_full_batch_per_rank'],8)
            self.assertEqual(config['scientific']['global_batch'],32);self.assertEqual(config['scientific']['outer_iterations'],25)
            self.assertEqual(config['sampling']['waves'],[[0,1,2,3],[4,5,6,7]])
            self.assertEqual(config['evaluation']['completed_outers'],[15,25])
            self.assertEqual(config['common_cohort_manifest'],self.assets_value['common_cohort_manifest'])
            storage=json.loads(Path(config['gates']['storage']['path']).read_text())
            self.assertEqual(storage['backup_policy']['durable_target'],deploy.BACKUP)
            self.assertIn('Planned',storage['backup_policy']['synchronization_status'])
            self.assertTrue(storage['backup_policy']['large_outputs_remote_only'])
            self.assertFalse(Path(config['paths']['output_root']).exists(),'Parent controller requires a fresh absent output root')
            frozen[group['config']['path']]=Path(group['config']['path']).read_bytes()
        self.assertEqual(len(set(ids)),4);self.assertEqual(self.runtime.call_count,4);self.assertEqual(self.common.call_count,4)
        second=self.invoke();self.assertTrue(second['already_prepared']);self.assertFalse(second['fresh_initial_resource_checked'])
        self.assertEqual(first['study_manifest'],second['study_manifest']);self.assertEqual(self.fresh.call_count,1)
        for path,raw in frozen.items():self.assertEqual(Path(path).read_bytes(),raw)
        self.assertTrue(first['no_training_started']);self.assertEqual(first['large_products_local_bytes'],0)

    def test_partial_preparation_reuses_four_run_ids_and_preserves_original_snapshot(self):
        original=deploy.verify_common;calls=[]
        def fault(config):
            calls.append(config['variant'])
            if len(calls)==2:raise RuntimeError('Synthetic interrupted preparation')
            return original(config)
        with patch.object(deploy,'verify_common',side_effect=fault):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):self.invoke()
        identity_raw=(self.root/'receipts/deployment-identity.json').read_bytes()
        resource_raw=(self.root/'receipts/deployment-resource-initial.json').read_bytes()
        first_raw=(self.root/'group-G_POS/receipts/config.json').read_bytes()
        self.fresh.return_value=dict(self.resources,checked_unix=99999)
        result=self.invoke();self.assertTrue(result['accepted'])
        self.assertEqual((self.root/'receipts/deployment-identity.json').read_bytes(),identity_raw)
        self.assertEqual((self.root/'receipts/deployment-resource-initial.json').read_bytes(),resource_raw)
        self.assertEqual((self.root/'group-G_POS/receipts/config.json').read_bytes(),first_raw)

    def test_input_or_config_changes_cannot_overwrite_prepared_identity(self):
        first=self.invoke();path=Path(first['groups'][0]['config']['path']);raw=path.read_bytes()
        altered=json.loads(raw);altered['scientific']['learning_rate']=5e-6
        path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError,'Refuse overwrite'):self.invoke()
        self.assertEqual(json.loads(path.read_text())['scientific']['learning_rate'],5e-6,'Original damaged evidence must not be repaired silently')
        goal=Path(self.goal['path']);goal.write_text('Different goal\n');self.goal=cp.fileproof(goal)
        with self.assertRaisesRegex(ValueError,'deployment inputs changed'):self.invoke()

    def test_source_payload_tamper_rejected_before_configs(self):
        source=json.loads(Path(self.assets_value['source_model_manifest']['path']).read_text())['files'][0]
        Path(source['path']).write_text('tampered')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.invoke()
        self.assertFalse((self.root/'group-G_POS').exists())

    def test_missing_remote_readback_rejected_before_deployment_identity(self):
        value=json.loads(Path(self.pub['path']).read_text());value['remote_readback_sha']='b'*40
        Path(self.pub['path']).write_text(json.dumps(value));self.pub=cp.fileproof(self.pub['path'])
        with self.assertRaisesRegex(ValueError,'pushed and read back'):self.invoke()
        self.assertFalse((self.root/'receipts/deployment-identity.json').exists())

    def test_no_torch_or_cuda_visibility_or_wrong_uid_permitted(self):
        with patch.object(deploy,'sys',types.SimpleNamespace(modules={'torch':object()})):
            with self.assertRaisesRegex(ValueError,'must not import Torch'):self.invoke()
        with patch.dict(deploy.os.environ,{'CUDA_VISIBLE_DEVICES':'4'}):
            with self.assertRaisesRegex(ValueError,'empty CUDA visibility'):self.invoke()
        with patch.object(deploy.os,'getuid',return_value=1005):
            with self.assertRaisesRegex(ValueError,'authorized HKUST4 UID'):self.invoke()
        self.assertFalse((self.root/'receipts/deployment-identity.json').exists())

    def test_fresh_assigned_cards_ignore_others_reject_owned_foreign_and_low_free(self):
        config=deploy.template(str(self.root/'group-G_POS'),'G_POS','study',str(self.root))
        cards=[dict(index=i,uuid='other-'+str(i),actual_free_mib=0) for i in range(4)]
        cards+=[dict(index=i+4,uuid=u,actual_free_mib=45000) for i,u in enumerate(deploy.GPU_UUIDS)]
        apps=[dict(pid=99,gpu_uuid='other-0',used_mib=48000),dict(pid=1749337,gpu_uuid=deploy.GPU_UUIDS[-1],used_mib=738)]
        real_fresh=ORIGINAL_FRESH
        with patch.object(deploy.gates,'inventory',side_effect=lambda:(cards,apps)):
            accepted=real_fresh(config,{})
            self.assertEqual([d['index'] for d in accepted['devices']],[4,5,6,7])
            apps.append(dict(pid=123,gpu_uuid=deploy.GPU_UUIDS[0],used_mib=1))
            with self.assertRaisesRegex(ValueError,'non-oracle'):real_fresh(config,{})
            apps.pop();cards[4]['actual_free_mib']=20000
            with self.assertRaises(deploy.gates.Hold):real_fresh(config,{})


ORIGINAL_FRESH=deploy.fresh_resources
if __name__=='__main__':unittest.main()
