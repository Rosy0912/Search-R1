"""Meaningful CPU autograd fixtures; no CUDA, model download or HTTP."""
import ast
import contextlib
import copy
import hashlib
import os
from pathlib import Path
import types
import unittest
import weakref
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint
from objective import (encoded_pair, pair_objective, trajectory_batch_objective,
                       validate_component_clip, vocabulary_metrics)

ROOT = Path(os.environ.get('OPSD_ONLINE_TEST_RUNTIME', str(Path(__file__).resolve().parents[1] / 'opsd_debug_frozen_runtime_20261008/runtime'))).resolve()
FROZEN_SHA = {'search_e1/core.py':'0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc',
              'search_e1/opsd.py':'61df9cc630bca7466f3edfb06da18ad92ddff1c713bf7c6c4a104ba88672c037'}


def require_actual_frozen_runtime():
    for relative,wanted in FROZEN_SHA.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=wanted:
            raise ValueError('CPU fixture runtime is not actual frozen scientific source: '+relative)
    return ROOT


def native_functions():
    require_actual_frozen_runtime()
    namespace = dict(torch=torch, checkpoint=checkpoint)
    for file, name in [('search_e1/core.py', 'clipped_forward_kl'), ('search_e1/opsd.py', 'pair_loss')]:
        node = next(n for n in ast.parse((ROOT / file).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT/file), 'exec'), namespace)
    return namespace['pair_loss']


class TinyTransformer(torch.nn.Module):
    def __init__(self, owner):
        super().__init__()
        object.__setattr__(self, 'owner', weakref.ref(owner))
    def forward(self, input_ids, use_cache=False):
        owner = self.owner()
        x = owner.embedding(input_ids)
        # Causal context makes ordinary/privileged prefixes observably different.
        x = x.cumsum(dim=1) / torch.arange(1, x.shape[1]+1, device=x.device)[None,:,None]
        if owner.adapter_enabled: x = x + x @ owner.adapter
        return types.SimpleNamespace(last_hidden_state=x)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        self.head = torch.nn.Linear(8, 128, bias=False)
        self.embedding.requires_grad_(False); self.head.requires_grad_(False)
        self.adapter = torch.nn.Parameter(torch.randn(8, 8)*0.02)
        self.adapter_enabled = True
        self.base_model = TinyTransformer(self)
    def get_base_model(self): return self
    def get_output_embeddings(self): return self.head
    @contextlib.contextmanager
    def disable_adapter(self):
        old = self.adapter_enabled; self.adapter_enabled = False
        try: yield
        finally: self.adapter_enabled = old


def example():
    response = list(range(2, 78))
    return dict(prefixes=[[1,3],[1,9,11,13]], response=response,
                mask=[i % 7 != 1 for i in range(len(response))],
                offsets=[[i,i+1] for i in range(len(response))])


class ObjectiveTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(42); torch.set_num_threads(1)
    def test_clip_explicit_and_null(self):
        for bad in ['UNRESOLVED', '10', True, 0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): validate_component_clip(bad)
        self.assertIsNone(validate_component_clip(None))
        self.assertEqual(validate_component_clip(10), 10.)

    def test_clip_is_per_vocabulary_upper_only_and_teacher_detached(self):
        s = torch.tensor([[0., -20., 1.]], requires_grad=True)
        t = torch.tensor([[0., 5., 1.]], requires_grad=True)
        result = vocabulary_metrics(s, t, .1)
        q = torch.log_softmax(t.detach(), -1).exp()
        contribution = q*(torch.log_softmax(t.detach(),-1)-torch.log_softmax(s,-1))
        torch.testing.assert_close(result['post'], contribution.clamp(max=.1).sum(-1))
        self.assertTrue((contribution<0).any()); self.assertGreater(result['clipped_count'].item(), 0)
        result['post'].sum().backward()
        self.assertIsNone(t.grad); self.assertTrue(torch.isfinite(s.grad).all())

    def test_trajectory_equal_weight_differs_from_token_weight(self):
        means = trajectory_batch_objective([10., 1000.], [10, 1000])
        self.assertEqual(means, 1.)
        self.assertEqual(trajectory_batch_objective([10., 2000.], [10, 1000]), 1.5)
        self.assertNotEqual(1.5, (10+2000)/(10+1000))
        with self.assertRaises(ValueError): trajectory_batch_objective([1.],[0])

    def checkpoint_case(self, wanted):
        model = TinyModel(); direct = copy.deepcopy(model)
        # deepcopy's weakref must point to the copied owner.
        object.__setattr__(direct.base_model, 'owner', weakref.ref(direct))
        records=[]; encoded=example()
        loss, stats=pair_objective(model,encoded,.002,32,3,records.append,wanted)
        loss.backward()
        ref=native_functions()(direct,(encoded['prefixes'],encoded['response'],encoded['mask']),.002,32)
        ref.backward()
        torch.testing.assert_close(loss,ref,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(model.adapter.grad,direct.adapter.grad,rtol=1e-5,atol=1e-7)
        self.assertEqual(len(records),sum(encoded['mask']))
        self.assertEqual(len({r['token_index'] for r in records}),len(records))
        self.assertEqual(sum('teacher_top32_ids' in r for r in records),len(wanted))
        self.assertAlmostEqual(sum(r['global_batch_contribution'] for r in records),loss.item()/3,places=6)
        self.assertAlmostEqual(stats['post_loss'],loss.item(),places=6)
        self.assertTrue(all(r['loss_scope']=='actual_training_forward_before_optimizer' for r in records))
        self.assertIsNone(model.head.weight.grad); self.assertIsNone(model.embedding.weight.grad)

    def test_checkpoint_top_first_then_off_backward_matches_frozen(self):
        self.checkpoint_case([0])
    def test_checkpoint_top_last_after_off_backward_matches_frozen(self):
        self.checkpoint_case([75])

    def test_null_clip_gradient_matches_direct_forward_kl(self):
        model=TinyModel(); direct=copy.deepcopy(model)
        object.__setattr__(direct.base_model,'owner',weakref.ref(direct))
        encoded=example()
        loss,_=pair_objective(model,encoded,None,32,1)
        loss.backward()
        with patch('torch.utils.checkpoint.checkpoint', side_effect=lambda fn,*args,**kw:fn(*args)):
            exact,_=pair_objective(direct,encoded,None,32,1)
        exact.backward()
        torch.testing.assert_close(loss,exact)
        torch.testing.assert_close(model.adapter.grad,direct.adapter.grad)

    def test_teacher_distribution_fixed_under_student_adapter_changes(self):
        model=TinyModel(); records=[]
        pair_objective(model,example(),None,32,1,records.append,[0])
        before=next(r for r in records if r['token_index']==0)
        with torch.no_grad(): model.adapter.add_(.3)
        records=[]; pair_objective(model,example(),None,32,1,records.append,[0])
        after=next(r for r in records if r['token_index']==0)
        self.assertEqual(before['teacher_top32_logp'],after['teacher_top32_logp'])
        self.assertEqual(before['teacher_top32_ids'],after['teacher_top32_ids'])
        self.assertNotEqual(before['student_top32_logp'],after['student_top32_logp'])

    def test_nonfinite_distribution_refuses_loss_before_backward(self):
        model=TinyModel()
        with torch.no_grad(): model.head.weight[0,0]=float('nan')
        records=[]
        with self.assertRaises(FloatingPointError): pair_objective(model,example(),None,32,1,records.append)
        self.assertIsNone(model.adapter.grad); self.assertEqual(records,[])

    def test_native_encoding_context_and_mask_provenance(self):
        # A tokenizer fixture keeps exact span boundaries visible, no independent truncation.
        class Tokenizer:
            def __call__(self,text,**kw): return dict(input_ids=list(range(len(text))),offset_mapping=[[i,i+1] for i in range(len(text))])
            def apply_chat_template(self,messages,**kw): return [1]*(2 if messages=='ordinary' else 4)
        core=types.SimpleNamespace(prompt=lambda q,r:'ordinary' if r is None else 'teacher',
            response_encoding=lambda tok,response,spans:(list(range(len(response))),[any(a<=i<b for a,b in spans) for i in range(len(response))]))
        pair=dict(question='q',question_id='q1',student='abcde',reference='correct',student_policy_spans=[[0,2],[4,5]],reference_reward=1,student_reward=0)
        item,reason=encoded_pair(Tokenizer(),pair,core,9)
        self.assertIsNone(reason); self.assertEqual(item['mask'],[True,True,False,False,True])
        self.assertEqual(encoded_pair(Tokenizer(),pair,core,8),(None,'context_limit'))
        bad=dict(pair,student_policy_spans=[[4,9]])
        with self.assertRaises(ValueError): encoded_pair(Tokenizer(),bad,core)
        with self.assertRaises(ValueError): encoded_pair(Tokenizer(),dict(pair,reference_reward=0),core)


if __name__=='__main__': unittest.main()
