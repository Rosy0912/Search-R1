"""Tiny CPU contracts only; no real PT/HF, CUDA, vLLM, Ray or job execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from prepare_window_inputs import prepare_payload,position_report_identity,original_tokenizer_ids
from single_window_runner import true_generated_alignment,compare_window_anchors
from window_contracts import validate_inputs,json_payload_sha


class MetadataAndPromptTests(unittest.TestCase):
    def plan(self):
        return json.loads(Path(__file__).with_name('WINDOW_INPUTS_PLAN.json').read_bytes())

    def payload(self):
        return dict(input_ids=torch.tensor([[0,1,2,3,0],[1,2,3,4,5]]),
            attention_mask=torch.tensor([[0,1,1,1,0],[1,1,1,1,1]]),
            position_ids=torch.tensor([[0,0,1,2,3],[0,1,2,3,4]]),
            responses=torch.tensor([[3,0],[4,5]]),distribution={'temperature':1.})

    def record(self):
        return dict(rank=0,world_size=8,generation_call=5,saved_rows=2,local_rows=2,
                    optimizer_committed=0,distribution={'temperature':1.})

    def test_bound_actual_cpu_metadata_plan_keeps_fixed_scope(self):
        value=validate_inputs(self.plan())
        self.assertEqual([r['local_rows'] for r in value['ranks']],[320]*8)
        self.assertEqual([r['expected_actor_micro4_calls'] for r in value['ranks']],[80]*8)
        self.assertEqual(value['global_requests'],2560)
        self.assertEqual(value['maximum_new_emitted_tokens'],1280000)
        self.assertFalse(value['actual_tensor_rows_verified'])

    def test_reject_plan_changes_and_false_preparation_claim(self):
        for change in ('rows','order','response','shape','claimed','escape'):
            value=self.plan()
            if change=='rows':value['ranks'][0]['local_rows']=160
            if change=='order':value['ranks'][0]['original_physical_rows'].reverse()
            if change=='response':value['ranks'][0]['source_response_width']=499
            if change=='shape':value['ranks'][0]['source_full_input_identity']['input_ids']['shape'][1]+=1
            if change=='claimed':value['actual_tensor_rows_verified']=True
            if change=='escape':value['hf_config_file']['path']='/another/cluster/config.json'
            with self.subTest(change=change),self.assertRaises(ValueError):validate_inputs(value)

    def test_complete_prompt_order_leftpad_and_position_identity(self):
        value,meta=prepare_payload(self.payload(),self.record(),0,0)
        self.assertEqual(value['batch']['input_ids'].tolist(),[[0,1,2],[1,2,3]])
        self.assertEqual(meta['original_physical_rows'],[0,1])
        self.assertEqual(meta['prompt_lengths'],[2,3])
        original=position_report_identity(self.payload()['input_ids'])
        altered=self.payload()['input_ids'];altered[0,1]=7
        self.assertNotEqual(original,position_report_identity(altered))

    def test_invalid_suffix_dtype_or_prompt_position_refuses(self):
        for change in ('suffix','dtype','position','internalpad','distribution'):
            value=self.payload()
            if change=='suffix':value['responses'][0,0]=9
            if change=='dtype':value['position_ids']=value['position_ids'].float()
            if change=='position':value['position_ids'][0,1]=3
            if change=='internalpad':value['attention_mask'][1,:3]=torch.tensor([1,0,1])
            if change=='distribution':value['distribution']={'temperature':.7}
            with self.subTest(change=change),self.assertRaises(ValueError):prepare_payload(value,self.record(),0,0)

    def test_canonical_json_hash_binds_order_and_values(self):
        self.assertEqual(json_payload_sha({'a':1,'b':2}),json_payload_sha({'b':2,'a':1}))
        self.assertNotEqual(json_payload_sha([1,2]),json_payload_sha([2,1]))

    def test_actual_bos_none_is_preserved_and_not_guessed(self):
        value={'bos_token_id':None,'eos_token_id':151645,'pad_token_id':151643}
        self.assertEqual(original_tokenizer_ids(value),value)
        for altered in ({'bos_token_id':None,'pad_token_id':151643},
                        {'bos_token_id':None,'eos_token_id':None,'pad_token_id':151643},
                        {'bos_token_id':False,'eos_token_id':151645,'pad_token_id':151643}):
            with self.assertRaises(ValueError):original_tokenizer_ids(altered)

    def test_bound_hf110_config_omits_default_bos(self):
        import hashlib
        payload = (Path(__file__).parent/'fixtures/hf110-config-metadata.json').read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
            '879f8585cd1807a067222aa146212f59537cc498f8f9e9d3adc8c791739a7fd0')
        config = json.loads(payload)
        self.assertNotIn('bos_token_id', config)
        self.assertEqual(original_tokenizer_ids(config),
            dict(pad_token_id=151643, eos_token_id=151645, bos_token_id=None))
        for key in ('pad_token_id', 'eos_token_id'):
            altered = dict(config); del altered[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                original_tokenizer_ids(altered)


class ActualOutputAndAnchorTests(unittest.TestCase):
    def raw(self):
        responses=torch.zeros((1,500),dtype=torch.int64);responses[0,:2]=torch.tensor([2,3])
        attention=torch.zeros((1,503),dtype=torch.int64);attention[0,:5]=1
        logprobs=torch.zeros((1,500));logprobs[0,:2]=torch.tensor([-.5,-1.5])
        receipt={'generated':[dict(input_row=0,token_ids=[2,3],true_response_length=2,
            logprobs=[{'2':{'logprob':-.5}},{'3':{'logprob':-1.5}}])]}
        return receipt,SimpleNamespace(batch=dict(responses=responses,attention_mask=attention,__sampling_log_probs=logprobs))

    def test_real_observer_logprobs_schema_and_true_length(self):
        receipt,raw=self.raw();mask,sampled=true_generated_alignment(receipt,raw,0)
        self.assertEqual(int(mask.sum()),2);self.assertEqual(sampled[mask].tolist(),[-.5,-1.5])

    def test_actual_observer_receipt_to_alignment_and_strict_json(self):
        from test_window_observer import ObserverTest,group,model_input,completion
        from search_r1.baselines.probability_probe_contracts import strict_json_bytes
        fixture=ObserverTest();model,observer=fixture.observer([
            dict(id='a',rank=0,input_row=0,response_offset=0)])
        observer.observe_prompts([[1,2]]);observer.register_request('real9',[1,2],0)
        with observer:
            native=model_input([group(40,[1,2],[],0,True)],{'real9':[40]},[1],[2])
            actual,_=fixture.run_step(observer,model,native,[[1,2,3,4]],{40:3})
            observer.observe_generated([completion('real9',[1,2],actual.outputs[0].samples,'stop')])
        receipt=observer.receipt();self.assertTrue(receipt['observer_evidence_complete'])
        strict_json_bytes(receipt)  # the real produced object must publish unchanged.
        value=actual.outputs[0].samples[0].logprobs[3].logprob
        responses=torch.zeros((1,500),dtype=torch.long);responses[0,0]=3
        attention=torch.zeros((1,502),dtype=torch.long);attention[0,:3]=1
        returned=torch.zeros((1,500));returned[0,0]=value
        raw=SimpleNamespace(batch=dict(responses=responses,attention_mask=attention,__sampling_log_probs=returned))
        mask,sampled=true_generated_alignment(receipt,raw,0)
        self.assertEqual(int(mask.sum()),1);self.assertEqual(float(sampled[0,0]),value)

    def test_mask_scalar_and_padded_suffix_changes_refuse(self):
        for change in ('mask','scalar','pad','length'):
            receipt,raw=self.raw()
            if change=='mask':raw.batch['attention_mask'][0,5]=1
            if change=='scalar':raw.batch['__sampling_log_probs'][0,0]=-.6
            if change=='pad':raw.batch['responses'][0,7]=8
            if change=='length':receipt['generated'][0]['true_response_length']=1
            with self.subTest(change=change),self.assertRaises(ValueError):true_generated_alignment(receipt,raw,0)

    def anchors(self):
        head=torch.tensor([2.,1.,0.],dtype=torch.bfloat16)
        policy=head+2
        scalar=float(torch.log_softmax(head.float(),-1)[1])
        observed={'anchors':[dict(id='a',status='observed',input_row=0,response_offset=0,request_id='real0',
            context_token_ids=[4,5],context_positions=[0,1],target_token_id=1,sampled_logprob=scalar)]}
        actor=[dict(anchor_id='a',status='ORIGINAL_ACTOR_SAME_CALL_LOGITS_CAPTURED',input_row=0,response_offset=0,
            context_token_ids=[4,5],context_positions=[0,1],target_token=1,native_logprob=scalar)]
        return observed,{'a':dict(head_logits=head,sampler_input_logits=head.clone())},actor,{'a':policy}

    def test_constant_logits_shift_is_observed_separately(self):
        row=compare_window_anchors(*self.anchors())[0]
        self.assertEqual(row['full_vocab_actor_minus_vllm_head']['max_abs'],2.)
        self.assertEqual(row['full_vocab_actor_minus_vllm_head_after_constant']['max_abs'],0.)
        self.assertEqual(row['fp32_sampler_input_minus_actual_sampled'],0.)
        self.assertFalse(row['probability_accepted']);self.assertFalse(row['training_accepted'])

    def test_finite_numeric_mismatch_kept_without_probability_acceptance(self):
        args=self.anchors();args[3]['a'][1]+=1
        row=compare_window_anchors(*args)[0]
        self.assertGreater(row['full_vocab_actor_minus_vllm_head_after_constant']['max_abs'],.01)
        self.assertFalse(row['probability_accepted'])

    def test_context_position_target_or_shape_mismatch_refuses(self):
        for change in ('context','position','target','shape','nonfinite'):
            args=self.anchors()
            if change=='context':args[2][0]['context_token_ids']=[4,6]
            if change=='position':args[2][0]['context_positions']=[0,2]
            if change=='target':args[2][0]['target_token']=2
            if change=='shape':args[3]['a']=torch.ones(4)
            if change=='nonfinite':args[3]['a'][0]=float('inf')
            with self.subTest(change=change),self.assertRaises(ValueError):compare_window_anchors(*args)

    def test_early_native_finish_does_not_fabricate_anchor(self):
        observed={'anchors':[dict(id='a',status='absent_native_early_finish',input_row=0,response_offset=17,
            request_id='real0',true_response_length=3,finish_reason='stop')]}
        actor=[dict(anchor_id='a',status='ABSENT_EARLY_FINISH',input_row=0,response_offset=17)]
        result=compare_window_anchors(observed,{},actor,{})
        self.assertEqual(result[0]['status'],'absent_native_early_finish')
        with self.assertRaises(ValueError):compare_window_anchors(observed,{},actor,{'a':torch.ones(3)})


if __name__=='__main__':unittest.main()
