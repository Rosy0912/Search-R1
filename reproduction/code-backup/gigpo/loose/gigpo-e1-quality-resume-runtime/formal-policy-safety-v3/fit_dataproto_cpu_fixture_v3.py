"""CPU execution of the real fit batch/trace/select/old/reference source path.

Only model generation and the actor/ref RPC are explicit CPU fixtures. The
source52 and candidate fit statements, DataProto, TensorDict, RolloutTrace,
select_rows, tensor conversions/unions and actual chunk boundaries execute.
No CUDA/Ray/FSDP/native or production acceptance is claimed by these tests.
"""
import ast
import contextlib
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

import numpy as np
import torch
from verl import DataProto
from search_r1.baselines.rollout import select_rows
from search_r1.baselines.trajectory_records import RolloutTrace

BASE='52befe968c856162d1bc79cd0171bb38309f4f9e'
REPO=Path(__import__(DataProto.__module__,fromlist=["__file__"]).__file__).resolve().parents[1]
SOURCE=REPO/'verl/trainer/ppo/ray_trainer.py'


def fit_through_reference(text,path):
    """Extract unchanged actual fit source statements through ref RPC boundary."""
    tree=ast.parse(text)
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RayPPOTrainer')
    fit=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='fit')
    loop=next(node for node in ast.walk(fit) if isinstance(node,ast.For)
              and isinstance(node.target,ast.Name) and node.target.id=='batch_dict')
    start=next(index for index,node in enumerate(loop.body)
               if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='batch')
    selected=[]
    for node in loop.body[start:]:
        selected.append(node)
        if isinstance(node,ast.With) and any(isinstance(item.context_expr,ast.Call)
            and item.context_expr.args and isinstance(item.context_expr.args[0],ast.Constant)
            and item.context_expr.args[0].value=='step' for item in node.items):
            reference=next(index for index,item in enumerate(node.body) if isinstance(item,ast.If)
                and isinstance(item.test,ast.Attribute) and item.test.attr=='use_reference_policy')
            node.body=node.body[:reference+1]
            break
    function=ast.FunctionDef(name='actual_fit_batch_through_reference',
        args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=name) for name in
            ('self','batch_dict','generation_manager','gen_config')],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),
        body=[ast.Assign(targets=[ast.Name(id='metrics',ctx=ast.Store())],value=ast.Dict(keys=[],values=[])),
              ast.Assign(targets=[ast.Name(id='timing_raw',ctx=ast.Store())],value=ast.Dict(keys=[],values=[]))]
             +selected+[ast.Return(value=ast.Name(id='batch',ctx=ast.Load()))],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace=dict(DataProto=DataProto,np=np,torch=torch,Path=Path,_timer=lambda *args:contextlib.nullcontext())
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[function.name]


class ActualFitBoundaryFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.calls=[];self.poison=None;self.order=list(reversed(range(2560)))
        self.manager=types.SimpleNamespace()
        ids=torch.arange(512,dtype=torch.long).reshape(-1,1).repeat(1,3)
        self.qa=dict(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(3).expand_as(ids).clone(),
                     index=np.asarray(['q'+str(i) for i in range(512)],dtype=object),
                     original_question=np.asarray(list(range(512)),dtype=object))
        self.generated=None
        def generate(manager,gen_batch,batch,metrics):
            self.calls.append('CPU_generation_fixture')
            self.assertIsInstance(manager.rollout_trace,RolloutTrace)
            self.assertEqual(manager.rollout_trace.ids,[f'q{i}' for i in range(512) for _ in range(5)])
            self.assertEqual(batch.non_tensor_batch['audit_trajectory_row'].tolist(),list(range(2560)))
            if self.poison=='int64':batch.non_tensor_batch['audit_trajectory_row']=np.arange(2560,dtype=np.int64)
            elif self.poison=='length':batch.non_tensor_batch['audit_trajectory_row']=np.arange(2559,dtype=object)
            selected=select_rows(gen_batch,self.order)
            prompts=selected.batch['input_ids'];responses=torch.tensor([[8,9]]*2560)
            input_ids=torch.cat((prompts,responses),dim=1)
            self.generated=DataProto.from_dict(tensors=dict(prompts=prompts,responses=responses,input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),position_ids=torch.arange(5).expand_as(input_ids).clone()))
            return self.generated,self.order
        def old(batch):
            self.calls.append('CPU_old_RPC_fixture')
            self.assertIs(batch,self.generated)
            return DataProto.from_dict(tensors={'old_log_probs':torch.full((len(batch),2),-.5)})
        def reference(batch):
            self.calls.append('CPU_ref_RPC_fixture');batch.check_consistency()
            self.assertEqual(batch.non_tensor_batch['audit_trajectory_row'].dtype,object)
            self.assertEqual(batch.non_tensor_batch['audit_trajectory_row'].tolist(),list(range(2560)))
            self.assertEqual(batch.non_tensor_batch['uid'].tolist(),batch.non_tensor_batch['index'].tolist())
            self.assertEqual(batch.batch['input_ids'][:,0].tolist(),[i for i in range(512) for _ in range(5)])
            self.assertEqual([len(chunk) for chunk in batch.chunk(chunks=8)],[320]*8)
            return DataProto.from_dict(tensors={'ref_log_prob':torch.full((len(batch),2),-.6)})
        def balance(batch,metrics):
            selected=select_rows(batch,list(reversed(range(len(batch)))))
            batch.batch,batch.non_tensor_batch=selected.batch,selected.non_tensor_batch
        self.trainer=types.SimpleNamespace(global_steps=26,use_reference_policy=True,
            config=types.SimpleNamespace(do_search=True,
                actor_rollout_ref=types.SimpleNamespace(rollout=types.SimpleNamespace(n_agent=5,n=1)),
                trainer=types.SimpleNamespace(trajectory_logging=True,default_local_dir=str(Path(self.temp.name)/'checkpoints'))),
            baseline_controller=types.SimpleNamespace(generate=generate),
            actor_rollout_wg=types.SimpleNamespace(compute_log_prob=old),
            ref_policy_wg=types.SimpleNamespace(compute_ref_log_prob=reference),_balance_batch=balance)
        self.config=types.SimpleNamespace(max_start_length=2048)

    def candidate(self):return fit_through_reference(SOURCE.read_text(),SOURCE)

    def test_candidate_actual_fit_trace_select_old_reference_continuation(self):
        batch=self.candidate()(self.trainer,self.qa,self.manager,self.config)
        batch.check_consistency()
        self.assertEqual(self.calls,['CPU_generation_fixture','CPU_old_RPC_fixture','CPU_ref_RPC_fixture'])
        self.assertTrue(all(type(value) is int for value in batch.non_tensor_batch['audit_trajectory_row']))
        self.assertEqual(batch.non_tensor_batch['audit_trajectory_row'].tolist(),list(range(2560)))
        self.assertEqual(len(batch),2560)
        self.assertEqual(batch.batch['old_log_probs'].dtype,torch.float32)
        self.assertEqual(batch.batch['ref_log_prob'].dtype,torch.float32)

    def test_actual_source52_fit_reproduces_failure_before_old_ref(self):
        original=subprocess.check_output(['git','show',BASE+':verl/trainer/ppo/ray_trainer.py'],cwd=REPO,text=True)
        function=fit_through_reference(original,'<actual-git-source52-fit>')
        with self.assertRaisesRegex(AssertionError,'dtype=object'):
            function(self.trainer,self.qa,self.manager,self.config)
        self.assertNotIn('CPU_old_RPC_fixture',self.calls);self.assertNotIn('CPU_ref_RPC_fixture',self.calls)

    def test_candidate_still_rejects_wrong_non_tensor_dtype(self):
        self.poison='int64'
        with self.assertRaisesRegex(AssertionError,'dtype=object'):
            self.candidate()(self.trainer,self.qa,self.manager,self.config)
        self.assertNotIn('CPU_old_RPC_fixture',self.calls)

    def test_candidate_still_rejects_wrong_metadata_length(self):
        self.poison='length'
        with self.assertRaises((IndexError,AssertionError)):
            self.candidate()(self.trainer,self.qa,self.manager,self.config)
        self.assertNotIn('CPU_old_RPC_fixture',self.calls)


