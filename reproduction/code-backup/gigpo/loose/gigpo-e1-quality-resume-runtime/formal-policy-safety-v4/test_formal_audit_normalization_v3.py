"""Actual source52 fit through actual v3 generate metadata adapter on CPU.

Real DataProto/TensorDict/source52 fit statements/RolloutTrace and the candidate
OuterProbabilityHooks.generate method body run. Actor/ref/generation/prod-role
RPCs are explicit CPU fixtures, no CUDA/Ray/Adam or probability acceptance.
"""
import ast
import copy
import os
from pathlib import Path
import random
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

from formal_audit_normalization_v1 import normalize_exact_audit_rows
from formal_probability_v1 import require,processed_tokens_without_right_padding
import fit_dataproto_cpu_fixture_v3 as fixture


def actual_v3_generate_body():
    path=Path(__file__).with_name('formal_probability_driver_v1.py')
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='OuterProbabilityHooks')
    fn=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='generate')
    namespace=dict(normalize_exact_audit_rows=normalize_exact_audit_rows,require=require,Path=Path,os=os,
        processed_tokens_without_right_padding=processed_tokens_without_right_padding)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['generate']


class Source52WithExternalV3(fixture.ActualFitBoundaryFixture):
    def setUp(self):
        super().setUp()
        self.original_fixture_generate=self.trainer.baseline_controller.generate
        method=actual_v3_generate_body()
        self.hook=types.SimpleNamespace(trainer=self.trainer,last_completed=25,
            original_generate=self.original_fixture_generate)
        self.actor_role_calls=[]
        self.trainer.actor_rollout_wg.production_begin_outer=lambda outer: [dict(rank=rank) for rank in range(8)]
        self.trainer.actor_rollout_wg.production_end_sampling=lambda outer:self.actor_role_calls.append(('CPU_end_sampling_fixture',outer))
        self.env=patch.dict(os.environ,{'SEARCH_R1_BASELINE_OUTPUT':str(Path(self.temp.name)/'v3_cpu_probability_directory')})
        self.env.start();self.addCleanup(self.env.stop)
        self.trainer.baseline_controller.generate=lambda *args:method(self.hook,*args)
        self.assertIn('dtype=np.int64',fixture.SOURCE.read_text())

    def test_actual_source52_fit_reproduces_failure_before_old_ref(self):
        # Independent bare source baseline proves why this external adapter is
        # required. The qualified A baseline uses this normalization alone.
        self.trainer.baseline_controller.generate=self.original_fixture_generate
        super().test_actual_source52_fit_reproduces_failure_before_old_ref()

    def test_actual_v3_wrapper_rejects_preexisting_wrong_source_row_order(self):
        wrapped=self.trainer.baseline_controller.generate
        def corrupt(manager,gen_batch,batch,metrics):
            batch.non_tensor_batch['audit_trajectory_row']=np.arange(2560,dtype=np.int64)[::-1]
            return wrapped(manager,gen_batch,batch,metrics)
        self.trainer.baseline_controller.generate=corrupt
        with self.assertRaisesRegex(ValueError,'values/order'):
            self.candidate()(self.trainer,self.qa,self.manager,self.config)
        self.assertNotIn('CPU_generation_fixture',self.calls)
        self.assertNotIn('CPU_old_RPC_fixture',self.calls)

    def test_actual_v3_wrapper_rejects_wrong_source_dtype_before_generation(self):
        wrapped=self.trainer.baseline_controller.generate
        def corrupt(manager,gen_batch,batch,metrics):
            batch.non_tensor_batch['audit_trajectory_row']=np.arange(2560,dtype=np.int32)
            return wrapped(manager,gen_batch,batch,metrics)
        self.trainer.baseline_controller.generate=corrupt
        with self.assertRaisesRegex(ValueError,'int64'):
            self.candidate()(self.trainer,self.qa,self.manager,self.config)
        self.assertNotIn('CPU_generation_fixture',self.calls)


class AuditOnlyNeutrality(unittest.TestCase):
    def raw_batch(self):
        from verl import DataProto
        data=DataProto.from_dict(tensors={'input_ids':torch.arange(5120).reshape(2560,2)},
            non_tensors={'index':np.asarray(['q'+str(i//5) for i in range(2560)],dtype=object),
                         'anchors':np.asarray(['anchor']*2560,dtype=object)})
        data.non_tensor_batch['audit_trajectory_row']=np.arange(2560,dtype=np.int64)
        return data

    def test_only_exact_audit_representation_changes_and_all_RNG_stays(self):
        data=self.raw_batch();array=data.non_tensor_batch['audit_trajectory_row']
        tensors={key:value.clone() for key,value in data.batch.items()}
        metadata={key:value for key,value in data.non_tensor_batch.items() if key!='audit_trajectory_row'}
        python_rng=random.getstate();numpy_rng=copy.deepcopy(np.random.get_state());torch_rng=torch.get_rng_state().clone()
        result=normalize_exact_audit_rows(data)
        self.assertEqual(result['rows'],2560);self.assertEqual(array.dtype,np.int64)
        self.assertEqual(data.non_tensor_batch['audit_trajectory_row'].dtype,object)
        self.assertTrue(all(type(value) is int for value in data.non_tensor_batch['audit_trajectory_row']))
        self.assertEqual(data.non_tensor_batch['audit_trajectory_row'].tolist(),array.tolist())
        self.assertTrue(all(data.non_tensor_batch[key] is value for key,value in metadata.items()))
        self.assertTrue(all(torch.equal(data.batch[key],value) for key,value in tensors.items()))
        self.assertEqual(random.getstate(),python_rng)
        now=np.random.get_state();self.assertEqual(now[0],numpy_rng[0]);self.assertTrue(np.array_equal(now[1],numpy_rng[1]));self.assertEqual(now[2:],numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_rng));data.check_consistency()

    def test_array_type_dtype_shape_length_duplicate_and_reorder_never_repaired(self):
        for kind in ('list','float','object','short','2d','duplicate','reorder','other_metadata'):
            with self.subTest(kind=kind):
                data=self.raw_batch();row=data.non_tensor_batch['audit_trajectory_row']
                if kind=='list':data.non_tensor_batch['audit_trajectory_row']=row.tolist()
                elif kind=='float':data.non_tensor_batch['audit_trajectory_row']=row.astype(float)
                elif kind=='object':data.non_tensor_batch['audit_trajectory_row']=row.astype(object)
                elif kind=='short':data.non_tensor_batch['audit_trajectory_row']=row[:-1]
                elif kind=='2d':data.non_tensor_batch['audit_trajectory_row']=row.reshape(1280,2)
                elif kind=='duplicate':row[0]=1
                elif kind=='reorder':data.non_tensor_batch['audit_trajectory_row']=row[::-1]
                elif kind=='other_metadata':data.non_tensor_batch['index']=np.arange(2560,dtype=np.int64)
                with self.assertRaises((ValueError,AssertionError)):normalize_exact_audit_rows(data)


if __name__=='__main__':unittest.main(verbosity=2)
