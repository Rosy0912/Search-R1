"""Affected original actor AST and passive lifecycle; CPU fixtures, not FSDP."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import test_native_input_adapter as adapter

P=adapter.P
path=Path(__file__).with_name('single_window_runner.py')
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compare_native_boundaries')
scope=dict(torch=torch,require=P.require,tensor_digest=P.tensor_digest)
receipt=dict(bound_model_fields=dict(num_hidden_layers=36,hidden_size=2048))
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
finalizer_node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='finalize_native_boundaries')
exec(compile(ast.Module(body=[finalizer_node],type_ignores=[]),str(path),'exec'),scope)
finalize=scope['finalize_native_boundaries']
compare=scope['compare_native_boundaries']

class Tracker:
    def __init__(self,actor,refuse=False):self.actor=actor;self.events=[];self.refuse=refuse
    def prepare_actor_call(self,batch,mapping,start,call):
        self.events.append(('prepare',call,self.actor.entries,tuple(batch),start))
        assert set(batch)=={'input_ids','responses','attention_mask','position_ids'}
        assert mapping['effective_mask'].shape==batch['responses'].shape
    def check_errors(self):
        self.events.append(('after',self.actor.entries))
        if self.refuse:raise ValueError('deferred passive fixture failure')
    def clear_step(self):self.events.append(('clear',self.actor.entries))

class Tests(unittest.TestCase):
    def test_finalizer_refuses_deferred_incomplete_receipt_and_preserves_successful_result(self):
        outcomes=[{'fixture':'not actual runtime'}]
        calls=[]
        bad=dict(observer_evidence_complete=False,errors=['deferred fixture'])
        observer=SimpleNamespace(finalize=lambda value:calls.append(value) or bad,check_errors=lambda:None)
        with self.assertRaisesRegex(ValueError,'boundary evidence refused'):finalize(observer,outcomes)
        self.assertIs(calls[0],outcomes)
        good=dict(observer_evidence_complete=True,errors=[])
        observer.finalize=lambda value:good
        self.assertIs(finalize(observer,outcomes),good)

    def test_native_two_micro_calls_are_not_repeated_or_reordered_by_passive_lifecycle(self):
        actor=adapter.Actor();data=adapter.raw(6);co=adapter.Coordinator();t=Tracker(actor)
        with adapter.Recorder(actor,data,co,None,[],None,None,SimpleNamespace(check=lambda _:None),boundary=t) as rec:
            result=actor.compute_log_prob(adapter.Proto(data))
        self.assertEqual(actor.entries,2);self.assertEqual(actor.actor_module.calls,2);self.assertEqual(rec.calls,2)
        self.assertEqual(tuple(result.shape),(6,3))
        self.assertEqual(t.events,[('prepare',0,0,('responses','input_ids','attention_mask','position_ids'),0),('after',1),('clear',1),('prepare',1,1,('responses','input_ids','attention_mask','position_ids'),4),('after',2),('clear',2)])
        self.assertEqual(len(actor.actor_module._forward_hooks),0)
        self.assertEqual([item['native_forward_receipt']['original_actor_calls'] for item in rec.records],[1,1])
    def test_observer_failure_is_deferred_until_first_native_returns_and_prevents_second_native(self):
        actor=adapter.Actor();data=adapter.raw(6);co=adapter.Coordinator();t=Tracker(actor,True);original=actor._forward_micro_batch
        with self.assertRaisesRegex(ValueError,'deferred passive fixture failure'):
            with adapter.Recorder(actor,data,co,None,[],None,None,SimpleNamespace(check=lambda _:None),boundary=t):
                actor.compute_log_prob(adapter.Proto(data))
        self.assertEqual(actor.entries,1);self.assertEqual(actor.actor_module.calls,1)
        self.assertEqual(t.events[-1],('after',1));self.assertEqual(co.labels[-1],'actor same-call anchors 000')
        self.assertEqual(actor._forward_micro_batch,original);self.assertEqual(len(actor.actor_module._forward_hooks),0)
    def test_comparison_reports_first_observed_difference_without_mutating_vectors_or_causing_forward(self):
        left={'a':{f'boundary{i}':torch.zeros(2048,dtype=torch.bfloat16) for i in range(74)}}
        right={'a':{k:v.clone() for k,v in left['a'].items()}}
        right['a']['boundary3'][17]=.125
        original={k:P.tensor_digest(v) for k,v in right['a'].items()}
        result=compare(receipt,left,receipt,right,[dict(anchor_id='a',status='SAME_CAUSAL_CONTEXT_LOGITS_OBSERVED')])
        self.assertEqual(result[0]['first_nonidentical_observed_boundary'],'boundary3')
        self.assertEqual(result[0]['boundary_count'],74);self.assertEqual(result[0]['boundaries'][3]['max_abs'],.125)
        self.assertFalse(result[0]['unique_kernel_cause_proven']);self.assertFalse(result[0]['probability_accepted'])
        self.assertEqual(original,{k:P.tensor_digest(v) for k,v in right['a'].items()})
        with self.assertRaisesRegex(ValueError,'labels/count differ'):
            compare(receipt,left,receipt, {'a':{}},[dict(anchor_id='a',status='SAME_CAUSAL_CONTEXT_LOGITS_OBSERVED')])
    def test_explicit_early_finish_is_absent_not_zero_fabricated_and_observed_missing_refuses(self):
        result=compare(receipt, {}, receipt, {},[dict(anchor_id='a',status='absent_native_early_finish')])
        self.assertEqual(result[0]['status'],'absent_native_early_finish')
        with self.assertRaisesRegex(ValueError,'Missing actual paired'):
            compare(receipt, {}, receipt, {},[dict(anchor_id='a',status='SAME_CAUSAL_CONTEXT_LOGITS_OBSERVED')])

if __name__=='__main__':unittest.main(verbosity=2)
