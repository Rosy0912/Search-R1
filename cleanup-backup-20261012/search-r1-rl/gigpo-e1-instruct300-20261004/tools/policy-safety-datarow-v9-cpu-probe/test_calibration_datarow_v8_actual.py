"""Actual DataProto/select_rows and one_outer CPU boundary regression.

Run with the bound source52 on PYTHONPATH and --runtime containing the v8
driver. Genuine DataProto, TensorDict, repeat/pop/select_rows/union/chunk and
the complete candidate one_outer function body execute on real CPU tensors.
Model generation, reference/actor RPC, reward/advantage and text recording are
explicit CPU fixtures. No CUDA/Ray/FSDP is initialized and no Adam is executed.
This script also runs on the existing target Torch2.4 environment; local2.8
results are not reported as target2.4 or GPU results.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import tensordict
from verl import DataProto
from search_r1.baselines.rollout import SearchRollout, select_rows
from search_r1.baselines.recovery import atomic_torch_save

RUNTIME = None


def function_from_driver(name, **extra):
    path = RUNTIME/'calibration_driver_v1.py'
    tree = ast.parse(path.read_text())
    function = next(row for row in tree.body if isinstance(row,ast.FunctionDef) and row.name == name)
    namespace = dict(np=np,torch=torch,Path=Path,SCOPE='GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION',
                     require=lambda value,message: value or fail(message))
    namespace.update(extra)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def fail(message):
    raise ValueError(message)


def batch512():
    ids = torch.arange(512,dtype=torch.long).reshape(-1,1).repeat(1,3)
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(3,dtype=torch.long).expand_as(ids).clone(),
                index=np.asarray(['q'+str(i) for i in range(512)],dtype=object),
                fixture_question=np.asarray(list(range(512)),dtype=object))


class ActualDataProtoBoundary(unittest.TestCase):
    def ready_batch(self):
        data=DataProto.from_single_dict(batch512()).repeat(5,interleave=True)
        data.non_tensor_batch['audit_trajectory_row']=np.arange(2560,dtype=object)
        return data

    def test_actual_repeat_keeps_integer_audit_rows_and_five_groups(self):
        data=self.ready_batch(); data.check_consistency()
        rows=data.non_tensor_batch['audit_trajectory_row']
        self.assertEqual(rows.dtype,object)
        self.assertTrue(all(type(value) is int for value in rows))
        self.assertEqual(rows.tolist(),list(range(2560)))
        self.assertEqual(data.non_tensor_batch['fixture_question'].tolist(),
                         [i for i in range(512) for _ in range(5)])

    def test_real_select_preserves_reorder_duplicate_and_all_fields(self):
        data=self.ready_batch()
        order=[2559,0,100,0,17,2558]
        selected=select_rows(data,order); selected.check_consistency()
        for key,value in selected.non_tensor_batch.items():
            self.assertEqual(value.dtype,object)
            self.assertEqual(value.tolist(),data.non_tensor_batch[key][order].tolist())
        for key,value in selected.batch.items():
            self.assertTrue(torch.equal(value,data.batch[key][order]))
        self.assertEqual(selected.non_tensor_batch['audit_trajectory_row'].tolist(),order)

    def test_real_padded_duplicate_selection_keeps_dtype_and_lengths(self):
        data=self.ready_batch()
        order=list(range(13))+[0,1,2]
        padded=select_rows(data,order)
        chunks=padded.chunk(chunks=8)
        self.assertEqual([len(chunk) for chunk in chunks],[2]*8)
        self.assertEqual([x for chunk in chunks for x in chunk.non_tensor_batch['audit_trajectory_row']],order)

    def test_original_int64_boundary_reproduces_assertion(self):
        data=self.ready_batch()
        data.non_tensor_batch['audit_trajectory_row']=np.arange(2560,dtype=np.int64)
        with self.assertRaisesRegex(AssertionError,'dtype=object'):
            select_rows(data,list(range(2560)))

    def test_wrong_metadata_length_is_still_rejected(self):
        data=self.ready_batch(); data.non_tensor_batch['audit_trajectory_row']=np.arange(2559,dtype=object)
        with self.assertRaisesRegex(AssertionError,'length'):
            data.check_consistency()

    def test_wrong_metadata_type_is_still_rejected(self):
        data=self.ready_batch(); data.non_tensor_batch['audit_trajectory_row']=list(range(2560))
        with self.assertRaises(AssertionError): data.check_consistency()


class CompleteOneOuterCPU(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        self.calls=[]; self.generated_batch=None; self.last_alignment=None
        self.modules={}
        self.original_advance=SearchRollout.advance
        self.addCleanup(self.restore_advance)
        def module(name,**members):
            value=types.ModuleType(name); value.__dict__.update(members); self.modules[name]=value
        module('verl.trainer.ppo.ray_trainer',compute_data_metrics=lambda *a,**kw:{},reduce_metrics=lambda value:value)
        class GenerationConfig:
            def __init__(self,**kw): self.__dict__.update(kw)
        class Manager:
            def __init__(self,*args): self.tokenizer=args[0];self.config=args[2]
        module('search_r1.llm_agent.generation',GenerationConfig=GenerationConfig,LLMGenerationManager=Manager)
        class Trace:
            def __init__(self,*args): pass
        module('search_r1.baselines.trajectory_records',RolloutTrace=Trace,
               record_scored_batch=lambda *args:self.calls.append('CPU_text_record_fixture'))
        self.patch=patch.dict(sys.modules,self.modules);self.patch.start();self.addCleanup(self.patch.stop)
        self.source_order=list(reversed(range(2560)))
        self.poison=None

        class RPC:
            def calibration_phase(inner,phase): self.calls.append(('CPU_phase_RPC_fixture',phase))
            def compute_log_prob(inner,data):
                self.calls.append('CPU_old_RPC_fixture')
                return DataProto.from_dict(tensors={'old_log_probs':torch.full((len(data),2),-.5)})
            def compute_ref_log_prob(inner,data):
                data.check_consistency(); self.calls.append('CPU_ref_RPC_fixture')
                return DataProto.from_dict(tensors={'ref_log_prob':torch.full((len(data),2),-.6)})
            def update_actor(inner,data):
                self.calls.append('FORBIDDEN_ADAM_RPC');raise AssertionError('No CPU Adam allowed')
        rpc=RPC()
        def generate(manager,gen_batch,qa_batch,metrics):
            self.calls.append('CPU_full2560_generate_fixture')
            qa_batch.check_consistency()
            self.assertEqual(len(qa_batch),2560)
            self.assertTrue(all(type(value) is int for value in qa_batch.non_tensor_batch['audit_trajectory_row']))
            if self.poison=='int64': qa_batch.non_tensor_batch['audit_trajectory_row']=np.arange(2560,dtype=np.int64)
            if self.poison=='length': qa_batch.non_tensor_batch['audit_trajectory_row']=np.arange(2559,dtype=object)
            selected=select_rows(gen_batch,self.source_order)
            prompts=selected.batch['input_ids']
            responses=torch.tensor([[10,11]]*2560,dtype=torch.long)
            ids=torch.cat((prompts,responses),dim=1)
            generated=DataProto.from_dict(tensors=dict(prompts=prompts,responses=responses,input_ids=ids,
                attention_mask=torch.ones_like(ids),position_ids=torch.arange(5).expand_as(ids).clone(),
                step_ids=torch.zeros_like(responses)),meta_info={'temperature':1})
            self.generated_batch=generated
            return generated,self.source_order
        def advantage(data,metrics):
            self.calls.append('CPU_GiGPO_advantage_fixture')
            data.batch['advantages']=torch.ones((len(data),2))
            data.batch['policy_weights']=torch.full((len(data),2),.5)
            return data
        def balance(data,metrics):
            self.calls.append('actual_DataProto_length_balance_fixture')
            # Use actual select_rows to simulate the source's row permutation.
            selected=select_rows(data,list(reversed(range(2560))))
            data.batch,data.non_tensor_batch=selected.batch,selected.non_tensor_batch
            data.check_consistency()
        def loss_mask(data,metrics): data.batch['loss_mask']=torch.ones((len(data),2));return data,metrics
        def alignment(trainer,run,turns,binding,data):
            self.calls.append('CPU_probability_probe_fixture');data.check_consistency()
            self.last_alignment=data
            self.assertEqual(data.non_tensor_batch['audit_trajectory_row'].tolist(),list(range(2560)))
            self.assertEqual(data.non_tensor_batch['fixture_question'].tolist(),[i for i in range(512) for _ in range(5)])
            self.assertEqual(data.batch['input_ids'][:,0].tolist(),[i for i in range(512) for _ in range(5)])
            self.assertEqual([len(chunk) for chunk in data.chunk(chunks=8)],[320]*8)
            return {'CPU_fixture_only_not_probability_acceptance':True}
        config=types.SimpleNamespace(algorithm=types.SimpleNamespace(no_think_rl=False),
            actor_rollout_ref=types.SimpleNamespace(actor=types.SimpleNamespace(use_kl_loss=True)),
            retriever=types.SimpleNamespace(url='http://127.0.0.1:18081'),force_final_answer=False)
        sampler=types.SimpleNamespace(cursor=13312,state_dict=lambda:dict(cursor=13312))
        self.trainer=types.SimpleNamespace(config=config,global_steps=25,tokenizer=object(),actor_rollout_wg=rpc,
            ref_policy_wg=rpc,recovery_sampler=sampler,baseline_controller=types.SimpleNamespace(generate=generate,
            compute_advantages=advantage),reward_fn=lambda data:torch.ones((len(data),2)),
            _balance_batch=balance,_create_loss_mask=loss_mask)
        self.fn=function_from_driver('one_outer',install_training_prefix_observer=lambda run:(self.original_advance,[]),
            probability_and_alignment=alignment,json_save=lambda path,value:Path(path).write_text(json.dumps(value)))

    def restore_advance(self): SearchRollout.advance=self.original_advance

    def invoke(self):
        return self.fn(self.trainer,self.run,{'calibration_phase':'probability_probe_only'},iter([batch512()]))

    def test_complete_real_one_outer_512repeat_select_balance_old_ref_adv_probe(self):
        self.assertIsNone(self.invoke())
        self.assertEqual(self.trainer.global_steps,26)
        self.assertTrue((self.run/'actual-rollout-2560.pt').is_file())
        saved=torch.load(self.run/'actual-rollout-2560.pt',map_location='cpu',weights_only=False)
        self.assertEqual(saved['non_tensors']['audit_trajectory_row'].dtype,object)
        self.assertEqual(saved['non_tensors']['audit_trajectory_row'].tolist(),list(range(2560)))
        self.assertTrue(torch.equal(saved['tensors']['input_ids'],self.last_alignment.batch['input_ids']))
        self.assertEqual(self.calls.count('CPU_old_RPC_fixture'),1)
        self.assertEqual(self.calls.count('CPU_ref_RPC_fixture'),1)
        self.assertEqual(self.calls.count('CPU_probability_probe_fixture'),1)
        self.assertNotIn('FORBIDDEN_ADAM_RPC',self.calls)
        self.assertIs(SearchRollout.advance,self.original_advance)

    def test_one_outer_int64_poison_refused_before_old_ref_or_Adam(self):
        self.poison='int64'
        with self.assertRaisesRegex(AssertionError,'dtype=object'): self.invoke()
        self.assertNotIn('CPU_old_RPC_fixture',self.calls)
        self.assertNotIn('CPU_ref_RPC_fixture',self.calls)
        self.assertNotIn('FORBIDDEN_ADAM_RPC',self.calls)
        self.assertIs(SearchRollout.advance,self.original_advance)

    def test_one_outer_wrong_metadata_length_refused(self):
        self.poison='length'
        with self.assertRaises((AssertionError,IndexError)): self.invoke()
        self.assertNotIn('CPU_old_RPC_fixture',self.calls)
        self.assertNotIn('FORBIDDEN_ADAM_RPC',self.calls)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True);parser.add_argument('--json-out')
    args=parser.parse_args();RUNTIME=Path(args.runtime).resolve()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                             for cls in (ActualDataProtoBoundary,CompleteOneOuterCPU))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raw=(RUNTIME/'calibration_driver_v1.py').read_bytes()
    out=dict(status='PASS' if result.wasSuccessful() else 'FAIL',scope='ACTUAL_DATAPROTO_SELECT_ROWS_AND_ONE_OUTER_CPU_ONLY',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        torch_version=torch.__version__,numpy_version=np.__version__,tensordict_version=tensordict.__version__,
        actual_protocol_path=sys.modules[DataProto.__module__].__file__,
        actual_select_rows_path=sys.modules[select_rows.__module__].__file__,
        runtime_driver_path=str(RUNTIME/'calibration_driver_v1.py'),runtime_driver_sha256=hashlib.sha256(raw).hexdigest(),
        actual_source_protocol=True,actual_full_one_outer_function_body=True,
        CPU_RPC_generation_advantage_reference_text_fixtures=True,
        actual_CUDA_initialized=torch.cuda.is_initialized(),actual_GPU_acceptance=False,
        actual_Adam_writes=0,Ray_cluster_created=False)
    if args.json_out: Path(args.json_out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)
