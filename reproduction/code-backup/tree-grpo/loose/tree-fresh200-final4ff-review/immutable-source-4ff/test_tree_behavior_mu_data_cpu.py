"""New collector contracts on tiny CPU tensors, not real DataProto/Ray/assets."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_tree_raw_tokens_cpu import Tokenizer, TinyProto, params, ENV, source_functions
from search_r1.baselines.rollout import SearchRollout, State, object_array
from search_r1.baselines.tree_raw_tokens import (FIELD as RAW_FIELD, capture_native_emissions,
                                               sampled_action, fingerprint)
from search_r1.baselines.tree_behavior_mu import (PROTOCOL, TENSOR, FIELD, META, PROVENANCE_META,
    validate_options, runtime_provenance, compose_behavior_mu_rows, validate_behavior_mu_batch)


class BehaviorMuDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name)
        payload = b'{"synthetic_cpu_tokenizer":true}'
        (path/'tokenizer.json').write_bytes(payload)
        self.sha = hashlib.sha256(payload).hexdigest()
        self.tok = Tokenizer(path)
        self.manager = NS(tokenizer=self.tok,
            config=NS(max_prompt_length=4096, max_start_length=2048),
            tensor_fn=NS(create_position_ids=lambda mask: (mask.cumsum(-1)-1)*mask))
        self.env = patch.dict(os.environ, ENV); self.env.start()
        self.verl = ModuleType('verl'); self.verl.DataProto = TinyProto
        self.fake = patch.dict(sys.modules, {'verl': self.verl}); self.fake.start()
        self.engine = SearchRollout(self.manager, token_preservation={'tokenizer_sha256': self.sha},
                                   behavior_mu={'enabled': True, 'protocol': PROTOCOL})
        self.engine.output_meta = {}

    def tearDown(self):
        self.fake.stop(); self.env.stop(); self.tmp.cleanup()

    def state(self, ids=None, obs=None, call='fixture-native-call', source=0):
        ids = ids or [1, 2, 3, 151645]
        obs = [10] if obs is None else obs
        p = params(); p.logits_processors = None
        out = NS(token_ids=ids,
            logprobs=[{token: NS(logprob=-.25*(i+1))} for i, token in enumerate(ids)],
            finish_reason='stop' if ids[-1] == 151645 else 'length', stop_reason=None)
        request = NS(request_id='fixture-request', finished=True, outputs=[out], prompt_token_ids=[8])
        record = capture_native_emissions([request], p, call, 0)[0]
        self.assertEqual(len(record['sampling_params']), 24)
        self.assertEqual(record['sampling_params']['best_of'], 1)
        self.assertEqual(record['sampling_params']['_real_n'], 1)
        self.assertIsNone(record['sampling_params']['logits_processors'])
        text = self.tok.decode(ids, skip_special_tokens=True)
        from search_r1.baselines.tree_raw_tokens import canonical_text
        text = canonical_text(text); legacy = self.tok.encode(text, add_special_tokens=False)
        action, ledger = sampled_action(self.tok, record, ids+[151643]*(500-len(ids)),
                                       legacy, text, self.engine.tokenizer_identity)
        state = State(source, [8], [8]+action+obs, response=action+obs,
                      model_mask=[1]*len(action)+[0]*len(obs),
                      steps=[0]*len(action)+[-1]*len(obs))
        state.raw_ledger = [dict(turn=0, action=ledger, observation_token_ids=obs.copy(),
            observation_text='synthetic-observation', generation_context_token_ids=[8],
            generation_context_attention=[1], generation_context_positions=[0])]
        return state

    def output(self, states=None):
        return self.engine.compose(states or [self.state()])[0]

    def validate(self, data):
        return validate_behavior_mu_batch(data, expected_provenance=runtime_provenance(os.environ))

    def test_native_best_of_is_recorded_without_defaulting(self):
        out = NS(token_ids=[151645], logprobs=[{151645: NS(logprob=-.25)}],
                 finish_reason='stop', stop_reason=None)
        request = NS(request_id='best-of-fixture', finished=True, outputs=[out], prompt_token_ids=[8])
        settings = params()
        captured = capture_native_emissions([request], settings, 'best-of-capture', 0)[0]
        self.assertEqual(captured['sampling_params']['best_of'], 1)
        self.assertEqual(len(captured['sampling_params']), 24)
        for actual in (None, 2, 1., True):
            settings.best_of = actual
            with self.subTest(actual=actual), self.assertRaisesRegex(ValueError, 'best_of'):
                capture_native_emissions([request], settings, 'bad-best-of', 0)

    def test_missing_native_best_of_is_not_inferred_from_n1(self):
        settings = params(); del settings.best_of
        out = NS(token_ids=[151645], logprobs=[{151645: NS(logprob=-.25)}],
                 finish_reason='stop', stop_reason=None)
        request = NS(request_id='missing-best-of', finished=True, outputs=[out], prompt_token_ids=[8])
        with self.assertRaisesRegex(ValueError, 'best_of'):
            capture_native_emissions([request], settings, 'missing-best-of', 0)

    def test_best_of_ledger_missing_and_resealed_distribution_drift_refuse(self):
        for actual in ('missing', None, 2, 1., True):
            state = self.state()
            action = state.raw_ledger[0]['action']
            settings = action['native_emission']['sampling_params']
            if actual == 'missing': del settings['best_of']
            else: settings['best_of'] = actual
            # Re-seal the actual source action to exercise the sampling contract,
            # rather than relying only on generic ledger hash drift rejection.
            action['action_binding_sha256'] = fingerprint({k: v for k, v in action.items()
                if not k.startswith('prefix_scan_') and k != 'action_binding_sha256'})
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                self.output([state])
        data = self.output()
        data.non_tensor_batch[RAW_FIELD][0]['raw_ledger'][0]['action']['native_emission']['sampling_params']['best_of'] = 2
        with self.assertRaises(ValueError): self.validate(data)

    def test_native_context_processor_controls_and_real_n_cannot_be_hidden(self):
        out = NS(token_ids=[151645], logprobs=[{151645: NS(logprob=-.25)}],
                 finish_reason='stop', stop_reason=None)
        request = NS(request_id='native-control-fixture', finished=True, outputs=[out], prompt_token_ids=[8])
        for key, actual in (('truncate_prompt_tokens',1),('guided_decoding',{'json':{}}),
                            ('logit_bias',{151645:1.}),('allowed_token_ids',[151645]),('_real_n',2)):
            settings=params();setattr(settings,key,actual)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,key):
                capture_native_emissions([request],settings,'bad-native-control',0)
        for key in ('truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids','_real_n'):
            settings=params();delattr(settings,key)
            with self.subTest(missing=key),self.assertRaisesRegex(ValueError,key):
                capture_native_emissions([request],settings,'missing-native-control',0)

    def test_native_control_missing_or_resealed_mu_binding_drift_refuses(self):
        for key, actual in (('truncate_prompt_tokens',1),('guided_decoding',{'json':{}}),
                            ('logit_bias',{151645:1.}),('allowed_token_ids',[151645]),('_real_n',2)):
            for missing in (False,True):
                state=self.state();action=state.raw_ledger[0]['action'];settings=action['native_emission']['sampling_params']
                if missing:del settings[key]
                else:settings[key]=actual
                action['action_binding_sha256']=fingerprint({k:v for k,v in action.items()
                    if not k.startswith('prefix_scan_') and k!='action_binding_sha256'})
                with self.subTest(key=key,missing=missing),self.assertRaises(ValueError):self.output([state])
        data=self.output()
        data.non_tensor_batch[RAW_FIELD][0]['raw_ledger'][0]['action']['native_emission']['sampling_params']['allowed_token_ids']=[151645]
        with self.assertRaises(ValueError):self.validate(data)

    def test_actual_facade_native_entry_refuses_initial_controls_before_generate(self):
        native=source_functions('verl/third_party/vllm/vllm_v_0_6_3/llm.py','LLM',
            ['generate_with_emission_metadata'],{})['generate_with_emission_metadata']
        for key,value in (('truncate_prompt_tokens',1),('guided_decoding',{'json':{}}),
                          ('logit_bias',{151645:1.}),('allowed_token_ids',[151645]),('_real_n',2)):
            for missing in (False,True):
                calls=[]
                def fake_native(**kwargs):
                    calls.append(kwargs)
                    # An old engine might clear a processor-construction input.
                    # This callback must never receive unsupported initial values.
                    setattr(kwargs['sampling_params'],key,None)
                facade=NS(generate=fake_native)
                settings=params()
                if missing:delattr(settings,key)
                else:setattr(settings,key,value)
                with self.subTest(key=key,missing=missing),self.assertRaisesRegex(ValueError,key):
                    native(facade,sampling_params=settings)
                self.assertEqual(calls,[],'Initial unsupported settings reached the native entrypoint')

    def test_actual_compose_emits_exact_fp32_and_zero_observation_padding(self):
        data = self.output([self.state(), self.state([7, 151645], call='second-call', obs=[])])
        self.assertEqual(data.batch[TENSOR].dtype, torch.float32)
        self.assertEqual(data.batch[TENSOR].tolist(), [[-.25, -.5, -.75, 0.], [-.25, 0., 0., 0.]])
        receipt = self.validate(data)
        self.assertEqual(receipt['leaf_effective_token_occurrences'], 4)
        self.assertFalse(receipt['probability_accepted'])

    def test_default_off_omits_mu_and_enabled_requires_raw_protocol(self):
        old = SearchRollout(self.manager); old.output_meta = {}
        data = old.compose([State(0, [8], [8, 10], response=[10], model_mask=[1], steps=[0])])[0]
        self.assertNotIn(TENSOR, data.batch); self.assertNotIn(FIELD, data.non_tensor_batch)
        self.assertNotIn(META, data.meta_info)
        self.assertIsNone(validate_options({'enabled': False, 'protocol': PROTOCOL}))
        for options in ({'enabled': True, 'protocol': PROTOCOL},
                        {'enabled': 'true', 'protocol': PROTOCOL},
                        {'enabled': True, 'protocol': 'wrong'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SearchRollout(self.manager, behavior_mu=options)

    def test_missing_raw_scalar_or_partial_mu_fields_refuse(self):
        for key in ('sampled', 'binding', 'protocol', 'raw'):
            data = self.output()
            if key == 'sampled': del data.batch[TENSOR]
            if key == 'binding': del data.non_tensor_batch[FIELD]
            if key == 'protocol': del data.meta_info[META]
            if key == 'raw': del data.non_tensor_batch[RAW_FIELD]
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(data)
        state = self.state(); state.raw_ledger[0]['action']['selected_logprobs'].pop()
        with self.assertRaises(ValueError): self.output([state])

    def test_scalar_mutation_and_non_fp32_native_scalar_refuse(self):
        data = self.output(); data.batch[TENSOR][0, 1] = -.625
        with self.assertRaises(ValueError): self.validate(data)
        data = self.output(); data.batch[TENSOR] = data.batch[TENSOR].double()
        with self.assertRaises(ValueError): self.validate(data)
        state = self.state(); action = state.raw_ledger[0]['action']
        action['native_emission']['raw_selected_logprobs'][0] = -.1
        action['selected_logprobs'][0] = -.1
        action['action_binding_sha256'] = fingerprint({k:v for k,v in action.items()
            if not k.startswith('prefix_scan_') and k != 'action_binding_sha256'})
        with self.assertRaises(ValueError): self.output([state])

    def test_observation_padding_probability_and_loss_mask_cannot_become_actor_tokens(self):
        for change in ('observation_scalar', 'padding_scalar', 'loss_mask', 'nan_observation'):
            data = self.output([self.state(), self.state([7, 151645], obs=[], call='second-call')])
            if change == 'observation_scalar': data.batch[TENSOR][0, 3] = -.25
            if change == 'padding_scalar': data.batch[TENSOR][1, 3] = -.25
            if change == 'nan_observation': data.batch[TENSOR][0, 3] = float('nan')
            if change == 'loss_mask':
                data.batch['loss_mask'] = data.batch['info_mask'][:, -4:].clone()
                data.batch['loss_mask'][0, 3] = 1
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(data)

    def test_joint_row_shuffle_valid_but_binding_only_shuffle_refuses(self):
        data = self.output([self.state(), self.state([7, 151645], obs=[], call='second-call', source=1)])
        before = self.validate(data)
        order = np.array([1, 0])
        data.batch = {key: value[order] for key, value in data.batch.items()}
        data.non_tensor_batch = {key: value[order] for key, value in data.non_tensor_batch.items()}
        self.assertEqual(before, self.validate(data))
        data.non_tensor_batch[FIELD] = data.non_tensor_batch[FIELD][order]
        with self.assertRaises(ValueError): self.validate(data)

    def test_branch_two_roots_six_leaves_preserves_physical_vs_inherited_counts(self):
        roots = [self.state(call='root-call-0'), self.state(call='root-call-1', source=1)]
        leaves = [leaf for root in roots for leaf in (root, root.branch(), root.branch())]
        data = self.output(leaves); receipt = self.validate(data)
        self.assertEqual(receipt['rows'], 6)
        self.assertEqual(receipt['leaf_effective_token_occurrences'], 18)
        self.assertEqual(receipt['unique_sampled_occurrences'], 6)
        self.assertEqual(receipt['inherited_leaf_occurrences'], 12)
        child = leaves[1]; child.raw_ledger[0]['action']['selected_logprobs'][0] = -99.
        self.assertEqual(roots[0].raw_ledger[0]['action']['selected_logprobs'][0], -.25)
        with self.assertRaises(ValueError): self.output(leaves)

    def test_actual_settings_greedy_temperature_filters_processors_and_stop_refuse(self):
        for key, changed in (('temperature', 0.), ('top_p', .9), ('top_k', 5), ('min_p', .01),
                             ('presence_penalty', .1), ('frequency_penalty', .1),
                             ('repetition_penalty', 1.1), ('n', 2), ('ignore_eos', True),
                             ('stop_token_ids', [10]), ('stop', ['x']), ('seed', 42),
                             ('logits_processors', ['unsupported'])):
            state = self.state(); action = state.raw_ledger[0]['action']
            action['native_emission']['sampling_params'][key] = changed
            action['action_binding_sha256'] = fingerprint({k:v for k,v in action.items()
                if not k.startswith('prefix_scan_') and k != 'action_binding_sha256'})
            with self.subTest(key=key), self.assertRaises(ValueError): self.output([state])
        state = self.state(); del state.raw_ledger[0]['action']['native_emission']['sampling_params']['logits_processors']
        with self.assertRaises(ValueError): self.output([state])

    def test_identity_prefix_positions_and_turn_mask_mutations_refuse(self):
        for change in ('runtime', 'position', 'prefix', 'turn', 'attention', 'info'):
            data = self.output()
            if change == 'runtime': data.meta_info[PROVENANCE_META]['SLURM_JOB_ID'] = 'another-job'
            if change == 'position': data.batch['position_ids'][0, 1] = 10
            if change == 'prefix': data.batch['input_ids'][0, 0] = 9
            if change == 'turn': data.batch['step_ids'][0, 0] = 1
            if change == 'attention': data.batch['attention_mask'][0, 1] = 0
            if change == 'info': data.batch['info_mask'][0, 1] = 0
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(data)
        data = self.output(); other = dict(ENV, SLURM_JOB_ID='other-job')
        with self.assertRaises(ValueError): validate_behavior_mu_batch(data, expected_provenance=other)

    def test_added_beta_advantages_weights_and_correct_loss_mask_do_not_rebind_mu(self):
        data = self.output(); expected = self.validate(data)
        data.batch['actor_old_log_probs'] = torch.ones_like(data.batch[TENSOR]) * -7.
        data.batch['old_log_probs'] = data.batch[TENSOR].clone()
        data.batch['advantages'] = torch.ones_like(data.batch[TENSOR])
        data.batch['policy_weights'] = torch.ones_like(data.batch[TENSOR]) / 3.
        data.batch['loss_mask'] = data.batch['info_mask'][:, -data.batch[TENSOR].shape[1]:].clone()
        self.assertEqual(expected, self.validate(data))
        data.batch['actor_old_log_probs'][0, 0] = -8.
        self.assertEqual(expected, self.validate(data))

    def test_standalone_assembly_matches_actual_compose_without_mutating_projection(self):
        states = [self.state()]; data = self.output(states)
        before = copy.deepcopy(data.non_tensor_batch[RAW_FIELD][0])
        tensor, bindings = compose_behavior_mu_rows(states, data.non_tensor_batch[RAW_FIELD],
            response_width=data.batch['responses'].shape[1], expected_provenance=dict(ENV))
        self.assertTrue(torch.equal(tensor, data.batch[TENSOR]))
        self.assertEqual(bindings[0], data.non_tensor_batch[FIELD][0])
        self.assertEqual(before, data.non_tensor_batch[RAW_FIELD][0])


if __name__ == '__main__':
    unittest.main()
