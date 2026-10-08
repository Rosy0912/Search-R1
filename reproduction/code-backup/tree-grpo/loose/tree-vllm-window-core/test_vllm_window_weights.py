import copy
import unittest
import torch
from vllm_window_weights import compare_tp1_weights,qwen2_tp1_mapping


class SameWindowWeights(unittest.TestCase):
    def setUp(self):
        self.config=dict(layers=2,tensor_parallel_size=1,expected_dtype=torch.bfloat16,
                         tie_word_embeddings=True,vocab_size=8,hidden_size=4,
                         intermediate_size=6,q_rows=4,kv_rows=2,expected_device='cpu')
        self.source={}
        for item in qwen2_tp1_mapping(2):
            for name in item.sources:
                if name in self.source:continue
                if name in ('model.embed_tokens.weight','lm_head.weight'): shape=(8,4)
                elif name=='model.norm.weight' or name.endswith('layernorm.weight'):shape=(4,)
                elif '.self_attn.' in name:
                    n=4 if '.q_proj.' in name or '.o_proj.' in name else 2
                    shape=(n,) if name.endswith('bias') else (n,4)
                elif name.endswith('mlp.down_proj.weight'):shape=(4,6)
                else:shape=(6,4)
                # Distinct rows/keys expose swaps and accidental broadcasting.
                n=torch.tensor(shape).prod().item()
                self.source[name]=(torch.arange(n).reshape(shape).float()+len(self.source)*100)/7
        self.source['lm_head.weight']=self.source['model.embed_tokens.weight'].clone()
        self.actual={item.target:torch.cat([self.source[k] for k in item.sources],0).bfloat16()
                     if len(item.sources)>1 else self.source[item.sources[0]].bfloat16().clone()
                     for item in qwen2_tp1_mapping(2)}
        self.actual['lm_head.weight']=self.actual['model.embed_tokens.weight']

    def run_check(self): return compare_tp1_weights(self.source,self.actual,**self.config)
    def test_complete_packing_values_alias_and_no_acceptance(self):
        before={k:v.clone() for k,v in self.actual.items()}
        r=self.run_check();self.assertTrue(r['all_parameter_values_match']);self.assertEqual(r['serialized_source_keys'],27)
        self.assertEqual(r['serialized_target_keys'],17);self.assertEqual(r['different_elements'],0)
        self.assertTrue(r['same_window_tied_alias']);self.assertFalse(r['buffers_compared']);self.assertFalse(r['probability_accepted'])
        self.assertFalse(r['actual_device_is_cuda']);self.assertEqual(r['actual_parameter_device'],'cpu')
        self.assertFalse(r['live_window_provenance_proven_by_this_function']);self.assertFalse(r['historical_3268_identity_proven'])
        for k,v in before.items():self.assertTrue(torch.equal(v,self.actual[k]))
    def test_qkv_bias_wrong_order_is_detected(self):
        k='model.layers.0.self_attn.qkv_proj.bias'
        self.actual[k]=torch.cat([self.source['model.layers.0.self_attn.'+v+'_proj.bias'] for v in ('q','v','k')]).bfloat16()
        r=self.run_check();self.assertFalse(r['all_parameter_values_match']);self.assertGreater(r['different_elements'],0)
    def test_gate_up_wrong_order_is_detected(self):
        k='model.layers.0.mlp.gate_up_proj.weight'
        self.actual[k]=torch.cat([self.source['model.layers.0.mlp.'+v+'_proj.weight'] for v in ('up','gate')]).bfloat16()
        self.assertFalse(self.run_check()['all_parameter_values_match'])
    def test_equal_value_copy_is_not_tied_alias(self):
        self.actual['lm_head.weight']=self.actual['model.embed_tokens.weight'].clone()
        with self.assertRaisesRegex(ValueError,'alias'):self.run_check()
    def test_tied_source_divergence_is_refused(self):
        self.source['lm_head.weight'].add_(1)
        with self.assertRaisesRegex(ValueError,'different values'):self.run_check()
    def test_source_missing_and_extra_are_refused(self):
        key='model.norm.weight';value=self.source.pop(key)
        with self.assertRaisesRegex(ValueError,'coverage'):self.run_check()
        self.source[key]=value;self.source['unreviewed.weight']=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'coverage'):self.run_check()
    def test_actual_missing_or_extra_is_refused(self):
        value=self.actual.pop('model.norm.weight')
        with self.assertRaisesRegex(ValueError,'coverage'):self.run_check()
        self.actual['model.norm.weight']=value;self.actual['unreviewed.weight']=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'coverage'):self.run_check()
    def test_source_half_does_not_hide_master_identity(self):
        self.source['model.norm.weight']=self.source['model.norm.weight'].half()
        with self.assertRaisesRegex(ValueError,'master dtype'):self.run_check()
    def test_bad_live_dtype_or_shape_cannot_broadcast(self):
        key='model.norm.weight';self.actual[key]=torch.ones(1,dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'shape/dtype'):self.run_check()
        self.actual[key]=self.source[key].half()
        with self.assertRaisesRegex(ValueError,'shape/dtype'):self.run_check()
    def test_nonfinite_source_and_live_are_refused(self):
        key='model.norm.weight';self.source[key][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):self.run_check()
        self.source[key][0]=1;self.actual[key][0]=float('inf')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):self.run_check()
    def test_tp2_and_bool_tp_are_refused(self):
        for value in (2,True):
            self.config['tensor_parallel_size']=value
            with self.assertRaisesRegex(ValueError,'TP1'):self.run_check()
    def test_untied_separate_equal_values_are_valid(self):
        self.config['tie_word_embeddings']=False;self.actual['lm_head.weight']=self.actual['model.embed_tokens.weight'].clone()
        r=self.run_check();self.assertTrue(r['all_parameter_values_match']);self.assertFalse(r['same_window_tied_alias'])
    def test_fp32_runtime_configuration_is_explicit(self):
        self.config['expected_dtype']=torch.float32
        self.actual={item.target:torch.cat([self.source[k] for k in item.sources],0) if len(item.sources)>1 else self.source[item.sources[0]].clone() for item in qwen2_tp1_mapping(2)}
        self.actual['lm_head.weight']=self.actual['model.embed_tokens.weight'];self.assertTrue(self.run_check()['all_parameter_values_match'])
    def test_offloaded_cpu_cannot_claim_live_cuda(self):
        self.config['expected_device']='cuda:0'
        with self.assertRaisesRegex(ValueError,'window device'):self.run_check()
    def test_real_declared_key_count(self):
        plan=qwen2_tp1_mapping(36)
        self.assertEqual(len(plan),255);self.assertEqual(len({k for i in plan for k in i.sources}),435)

if __name__=='__main__':unittest.main(verbosity=2)
