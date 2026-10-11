"""Native EOS/string-stop metadata contract, no GPU speed/equivalence claim."""
import copy
import unittest
from search_r1.baselines.tree_raw_tokens import validate_native_emission

class Check(unittest.TestCase):
    def emission(self):
        return dict(true_emitted_length=2,raw_emitted_token_ids=[10,11],raw_selected_logprobs=[-1.,-2.],
            finished=True,finish_reason='stop',stop_reason='</search>',
            native_returned_text='<search>x</search>',
            sampling_params=dict(stop=['</search>','</answer>'],include_stop_str_in_output=True),
            native_stopping=dict(protocol='vllm_native_eos_and_stop_set_v1',
                source='detached_sampling_params_clone_native_update_from_generation_config_v0_6_3',
                primary_eos_source='native_input_preprocessor_get_eos_token_id_no_lora',
                generation_config_sha256='a'*64,primary_eos_token_id=151645,
                generation_config_eos_token_ids=[151645],sampling_stop_token_ids=[],
                sampling_all_stop_token_ids=[151645],effective_stop_token_ids=[151645],
                sampling_stop_strings=['</search>','</answer>'],include_stop_str_in_output=True))
    def test_bound_string_stop_keeps_original_ids_and_mu(self):
        e=self.emission(); self.assertEqual(validate_native_emission(e,151643,151645),([10,11],2,[-1.,-2.]))
    def test_missing_or_mutated_binding_refused(self):
        for edit in [lambda e:e.update(stop_reason='other'),
            lambda e:e['native_stopping'].update(include_stop_str_in_output=False),
            lambda e:e.pop('native_returned_text'),
            lambda e:e.update(finish_reason='length')]:
            e=self.emission();edit(e)
            with self.assertRaises(ValueError):validate_native_emission(e,151643,151645)
    def test_original_eos_contract_remains_valid(self):
        e=self.emission();e.update(stop_reason=None,raw_emitted_token_ids=[10,151645])
        e['sampling_params']=dict(stop=[],include_stop_str_in_output=False)
        e['native_stopping'].pop('sampling_stop_strings');e['native_stopping'].pop('include_stop_str_in_output')
        self.assertEqual(validate_native_emission(e,151643,151645)[0],[10,151645])
if __name__=='__main__':unittest.main()
