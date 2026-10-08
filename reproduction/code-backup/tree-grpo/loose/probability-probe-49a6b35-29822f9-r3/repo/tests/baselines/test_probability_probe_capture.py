"""CPU only: original pinned normalization AST, mock actor mapping, no GPU assets."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
MODULE=ROOT/'search_r1/baselines/probability_probe_capture.py'
spec=importlib.util.spec_from_file_location('probability_probe_capture',MODULE)
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
ORIGINAL=ROOT/'verl/utils/torch_functional.py'
tree=ast.parse(ORIGINAL.read_text())
nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('logprobs_from_logits','logprobs_from_logits_naive','gather_from_labels')]
scope={'torch':torch,'F':F,'FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE':False,'__name__':'bound_original_normalization_cpu'}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ORIGINAL),'exec'),scope)
NATIVE=scope['logprobs_from_logits']
BRANCH=dict(source_sha256=P.file_sha(ORIGINAL),module=NATIVE.__module__,flash_ce_available=False)


def batch():
    ids=torch.tensor([[0,1,2,3,4,0],[0,5,6,7,8,9]])
    return dict(input_ids=ids,position_ids=torch.tensor([[9,0,1,2,3,9],[9,0,1,2,3,4]]),
        attention_mask=torch.tensor([[0,1,1,1,1,0],[0,1,1,1,1,1]]),responses=ids[:,-3:],
        loss_mask=torch.tensor([[0,1,0],[1,0,1]]))


class Model(torch.nn.Module):
    def __init__(self,dtype):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.,dtype=dtype));self.calls=0
    def forward(self,input_ids,position_ids,attention_mask,use_cache):
        self.calls+=1
        value=torch.arange(16,dtype=self.weight.dtype).reshape(1,1,16)
        logits=(value*input_ids.unsqueeze(-1).to(self.weight.dtype)/13+position_ids.unsqueeze(-1).to(self.weight.dtype)/7)
        return SimpleNamespace(logits=logits)


class Actor:
    def __init__(self,remove_padding=True,dtype=torch.float64):
        self.use_remove_padding=remove_padding;self.actor_module=Model(dtype);self.entries=0
        self.actor_optimizer=SimpleNamespace(step=lambda: (_ for _ in ()).throw(AssertionError('optimizer forbidden')))
    def _forward_micro_batch(self,data,temperature):
        self.entries+=1;ids=data['input_ids'];r=data['responses'].shape[1]
        if self.use_remove_padding:
            valid=data['attention_mask'].flatten().bool();tokens=ids.flatten()[valid]
            output=self.actor_module(input_ids=tokens[None,:],position_ids=data['position_ids'].flatten()[valid][None,:],attention_mask=None,use_cache=False)
            logits=output.logits.squeeze(0);logits.div_(temperature)
            original=NATIVE(logits,torch.roll(tokens,-1));full=torch.zeros_like(ids,dtype=original.dtype)
            full.flatten()[valid]=original;current=full[:,-r-1:-1]
        else:
            output=self.actor_module(input_ids=ids,position_ids=data['position_ids'],attention_mask=data['attention_mask'],use_cache=False)
            logits=output.logits;logits.div_(temperature)
            current=NATIVE(logits[:,-r-1:-1],data['responses'])
        return torch.zeros_like(current),current


class Tests(unittest.TestCase):
    def test_packed_predictor_indices_use_all_attention_including_masked_observation(self):
        mapping=P.response_prediction_map(batch(),True)
        self.assertEqual(mapping['rows'].tolist(),[0,1,1])
        self.assertEqual(mapping['response_positions'].tolist(),[1,0,2])
        self.assertEqual(mapping['logit_indices'].tolist(),[2,5,7])
        self.assertEqual(mapping['labels'].tolist(),[4,7,9])
        self.assertEqual(mapping['expected_logit_rows'],9)

    def test_dense_indices_keep_full_buffer_width(self):
        mapping=P.response_prediction_map(batch(),False)
        self.assertEqual(mapping['logit_indices'].tolist(),[3,8,10])
        self.assertEqual(mapping['expected_logit_rows'],12)

    def test_original_packed_entry_called_once_and_same_logits_match_normalization(self):
        actor=Actor();state=actor.actor_module.state_dict()['weight'].clone()
        captured=P.capture_original_call(actor,batch(),2.)
        result=P.compare_same_logits(captured,NATIVE,BRANCH,1)
        self.assertEqual(actor.entries,1);self.assertEqual(actor.actor_module.calls,1)
        self.assertTrue(result['metadata']['selected_native_exactly_matches_original'])
        self.assertEqual(result['metadata']['selected_native_vs_original_max_abs_delta'],0.)
        self.assertEqual(result['metadata']['effective_response_token_occurrences'],3)
        self.assertTrue(torch.equal(actor.actor_module.weight,state))
        self.assertEqual(len(actor.actor_module._forward_hooks),0)
        self.assertFalse(result['metadata']['weights_independently_verified'])
        self.assertFalse(result['metadata']['acceptance'])

    def test_original_dense_entry_normalization_agrees_at_same_labels(self):
        actor=Actor(False);captured=P.capture_original_call(actor,batch(),1.)
        result=P.compare_same_logits(captured,NATIVE,BRANCH,2)
        self.assertTrue(result['metadata']['selected_native_exactly_matches_original'])

    def test_bfloat16_logits_dtype_and_actual_normalization_difference_recorded(self):
        actor=Actor(dtype=torch.bfloat16);captured=P.capture_original_call(actor,batch(),1.)
        result=P.compare_same_logits(captured,NATIVE,BRANCH)
        self.assertEqual(result['metadata']['original_forward_logits_dtype'],'torch.bfloat16')
        self.assertEqual(result['metadata']['original_actor_log_probs_dtype'],'torch.bfloat16')
        self.assertEqual(result['metadata']['normalization_actual_output_dtypes']['fp32_logsoftmax'],['torch.float32'])
        self.assertTrue(result['metadata']['selected_native_exactly_matches_original'])
        self.assertTrue(bool((result['token_values']['original_dtype_logsoftmax']!=result['token_values']['fp32_logsoftmax']).any()))
        self.assertEqual(actor.actor_module.calls,1)

    def test_prediction_attention_padding_refused_not_skipped_or_shifted(self):
        value=batch();value['attention_mask'][0,3]=0
        with self.assertRaisesRegex(ValueError,'attention-masked native prediction'):
            P.response_prediction_map(value,True)

    def test_suffix_token_mutation_refused(self):
        value=batch();value['responses']=value['responses'].clone();value['responses'][1,2]=10
        with self.assertRaisesRegex(ValueError,'exact input suffix'):P.response_prediction_map(value,True)

    def test_forward_hook_cleanup_on_original_entry_exception(self):
        actor=Actor()
        def failure(data,temperature):raise RuntimeError('synthetic entry failure')
        actor._forward_micro_batch=failure
        with self.assertRaisesRegex(RuntimeError,'synthetic entry failure'):P.capture_original_call(actor,batch(),1.)
        self.assertEqual(len(actor.actor_module._forward_hooks),0)

    def test_temperature_nonfinite_refused(self):
        with self.assertRaises(ValueError):P.capture_original_call(Actor(),batch(),float('inf'))

    def test_micro4_2_1_preserves_original_rows_and_partial_group(self):
        plan=P.micro_plan([[10,11,12,13],[20]])
        self.assertEqual([call['rows'] for call in plan[4]],[[10,11,12,13],[20]])
        self.assertEqual([call['rows'] for call in plan[2]],[[10,11],[12,13],[20]])
        self.assertEqual([call['rows'] for call in plan[1]],[[10],[11],[12],[13],[20]])

    def test_all_rank_call_alignment_explicitly_excludes_padding_duplicates(self):
        plans=[P.micro_plan([[0,1,2,3]]) for _ in range(7)]+[P.micro_plan([[0,1,2,3],[4,5,6,7]])]
        aligned=P.synchronize_plans(plans)
        for micro in (4,2,1):self.assertEqual(len(set(len(plan[micro]) for plan in aligned)),1)
        self.assertEqual(sum(call['exclude_duplicate'] for call in aligned[0][1]),4)
        self.assertFalse(any(call['exclude_duplicate'] for call in aligned[7][1]))
        self.assertEqual(len(plans[0][1]),4)

    def test_actual_native_availability_and_source_sha_refused_when_misreported(self):
        capture=P.capture_original_call(Actor(),batch(),1.)
        with self.assertRaisesRegex(ValueError,'actual imported availability'):
            P.compare_same_logits(capture,NATIVE,dict(BRANCH,flash_ce_available=True))
        with self.assertRaisesRegex(ValueError,'source binding changed'):
            P.compare_same_logits(capture,NATIVE,dict(BRANCH,source_sha256='0'*64))

    def test_source_binding_matches_original_parent_sha_and_rejects_change(self):
        result=P.bind_sources([{'path':str(ORIGINAL),'sha256':P.file_sha(ORIGINAL)}])
        self.assertEqual(result[0]['sha256'],'21eedf934c643b6c0a6809df89915c17b80796254920dcd4cbe2ec5850a7f575')
        with self.assertRaises(ValueError):P.bind_sources([{'path':str(ORIGINAL),'sha256':'0'*64}])

    def test_real_forward_attribute_full_digest_not_parameter_sample(self):
        actor=Actor();target=dict(logical_name='weight',module=actor.actor_module,attribute='weight',
            expected_cast_digest=P.tensor_digest(actor.actor_module.weight))
        with P.EffectiveWeightTrace([target]) as trace:P.capture_original_call(actor,batch(),1.)
        receipt=trace.receipt()
        self.assertEqual(receipt['logical_tensor_count'],1)
        self.assertEqual(receipt['records']['weight']['device'],'cpu')
        self.assertFalse(receipt['acceptance'])
        self.assertEqual(len(actor.actor_module._forward_pre_hooks),0)

    def test_forward_weight_difference_refused_and_hooks_removed(self):
        actor=Actor();expected=P.tensor_digest(actor.actor_module.weight)
        with torch.no_grad():actor.actor_module.weight.add_(1)
        target=dict(logical_name='weight',module=actor.actor_module,attribute='weight',expected_cast_digest=expected)
        with self.assertRaisesRegex(ValueError,'differs from full HF cast'):
            with P.EffectiveWeightTrace([target]):P.capture_original_call(actor,batch(),1.)
        self.assertEqual(actor.actor_module.calls,1) # Refusal deferred past whole native forward.
        self.assertEqual(actor.entries,1)
        self.assertEqual(len(actor.actor_module._forward_pre_hooks),0)

    def test_unobserved_forward_weight_coverage_refused(self):
        module=Model(torch.float64);target=dict(logical_name='unused.weight',module=module,attribute='weight',
            expected_cast_digest=P.tensor_digest(module.weight))
        with self.assertRaisesRegex(ValueError,'Incomplete effective forward'):
            with P.EffectiveWeightTrace([target]):pass

    def test_tied_forward_storage_alias_refused_even_if_values_equal(self):
        first,second=Actor(),Actor()
        targets=[dict(logical_name=str(i)+'.weight',module=actor.actor_module,attribute='weight',
            expected_cast_digest=P.tensor_digest(actor.actor_module.weight),expected_storage_alias_group='expected_tied')
            for i,actor in enumerate((first,second))]
        with self.assertRaisesRegex(ValueError,'storage alias differs'):
            with P.EffectiveWeightTrace(targets):
                P.capture_original_call(first,batch(),1.);P.capture_original_call(second,batch(),1.)

    def test_tied_forward_storage_alias_and_bfloat16_bytes_hash(self):
        first,second=Actor(dtype=torch.bfloat16),Actor(dtype=torch.bfloat16)
        second.actor_module.weight=first.actor_module.weight
        targets=[dict(logical_name=str(i)+'.weight',module=actor.actor_module,attribute='weight',
            expected_cast_digest=P.tensor_digest(actor.actor_module.weight),expected_storage_alias_group='expected_tied')
            for i,actor in enumerate((first,second))]
        with P.EffectiveWeightTrace(targets) as trace:
            P.capture_original_call(first,batch(),1.);P.capture_original_call(second,batch(),1.)
        self.assertEqual(trace.receipt()['logical_tensor_count'],2)
        self.assertFalse(trace.receipt()['tied_alias_identity_proven'])
        self.assertEqual(trace.receipt()['address_observations']['expected_tied']['scope'],
                         'observed_address_only; different forward/gather times may reuse addresses')

    def test_alias_exit_failure_cannot_publish_complete_receipt(self):
        first,second=Actor(),Actor()
        targets=[dict(logical_name=str(i)+'.weight',module=actor.actor_module,attribute='weight',
            expected_cast_digest=P.tensor_digest(actor.actor_module.weight),expected_storage_alias_group='address_group')
            for i,actor in enumerate((first,second))]
        trace=P.EffectiveWeightTrace(targets)
        with self.assertRaisesRegex(ValueError,'storage alias differs'):
            with trace:P.capture_original_call(first,batch(),1.);P.capture_original_call(second,batch(),1.)
        self.assertFalse(trace.completed_successfully)
        with self.assertRaisesRegex(ValueError,'successful complete trace exit'):trace.receipt()

    def test_original_forward_failure_after_all_weight_observation_cannot_publish_receipt(self):
        actor=Actor();target=dict(logical_name='weight',module=actor.actor_module,attribute='weight',expected_cast_digest=P.tensor_digest(actor.actor_module.weight))
        trace=P.EffectiveWeightTrace([target])
        original_entry=actor._forward_micro_batch
        def failure_after_model(data,temperature):
            original_entry(data,temperature)
            raise RuntimeError('original forward failure after model observation')
        actor._forward_micro_batch=failure_after_model
        with self.assertRaisesRegex(RuntimeError,'original forward failure'):
            with trace:
                P.capture_original_call(actor,batch(),1.)
        self.assertEqual(len(trace.records),1)
        self.assertFalse(trace.completed_successfully)
        with self.assertRaisesRegex(ValueError,'successful complete trace exit'):trace.receipt()

    def test_in_context_receipt_is_not_success_and_context_reuse_resets_evidence(self):
        actor=Actor();target=dict(logical_name='weight',module=actor.actor_module,attribute='weight',expected_cast_digest=P.tensor_digest(actor.actor_module.weight))
        trace=P.EffectiveWeightTrace([target])
        with trace:
            P.capture_original_call(actor,batch(),1.)
            with self.assertRaises(ValueError):trace.receipt()
        self.assertTrue(trace.completed_successfully)
        with self.assertRaisesRegex(ValueError,'Incomplete effective forward'):
            with trace:pass
        self.assertFalse(trace.completed_successfully)
        with self.assertRaises(ValueError):trace.receipt()


if __name__=='__main__':
    torch.set_num_threads(4)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
    require_no_cuda=not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PROBE_SOURCE_CORE_CPU_PASS' if result.wasSuccessful() and require_no_cuda else 'FAILED',
        tests_run=result.testsRun,cuda_initialized=torch.cuda.is_initialized(),real_model_loaded=False,remote_calls=False,
        native_source_sha256=P.file_sha(ORIGINAL),module_sha256=P.file_sha(MODULE),test_sha256=P.file_sha(__file__),
        acceptance=False,scope='Original normalization AST on CPU plus mock actor; not real FSDP/GPU/weight/RNG/resource acceptance'),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and require_no_cuda else 1)
