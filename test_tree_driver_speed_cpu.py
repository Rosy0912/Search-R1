"""Scoped Tree proof reuse on CPU; no model, service or GPU dependencies."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import search_r1.baselines.tree_raw_tokens as raw
from search_r1.baselines.rollout import State
from test_tree_raw_tokens_cpu import Tokenizer, params, ENV


def native_record(ids):
    stopping=dict(protocol='vllm_native_eos_and_stop_set_v1',primary_eos_token_id=151645,
        source='detached_sampling_params_clone_native_update_from_generation_config_v0_6_3',
        primary_eos_source='native_input_preprocessor_get_eos_token_id_no_lora',
        generation_config_sha256=raw.fingerprint({}),generation_config_eos_token_ids=[],
        sampling_stop_token_ids=[],sampling_all_stop_token_ids=[151645],effective_stop_token_ids=[151645])
    output=NS(token_ids=ids,logprobs=[{token:NS(logprob=-.25*(i+1))} for i,token in enumerate(ids)],
        finish_reason='stop' if ids[-1]==151645 else 'length',stop_reason=None)
    request=NS(request_id='cpu-native',finished=True,outputs=[output],prompt_token_ids=[8])
    with patch.dict('os.environ',ENV):
        return raw.capture_native_emissions([request],params(),'cpu-speed',0,native_stopping=stopping)[0]


class DriverSpeedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();path=Path(self.temp.name)
        (path/'tokenizer.json').write_bytes(b'{"fixture":"cpu-speed"}')
        self.tok=Tokenizer(path)
        self.tok.parts.update({200:b'\xe4',201:b'\xb8\xad',202:b'"\n\\\x00',203:b'\xff\xfe',
            204:b'\xe2',205:b'\x82',206:b'\xac'})
        self.identity=raw.tokenizer_identity(self.tok,hashlib.sha256((path/'tokenizer.json').read_bytes()).hexdigest())

    def tearDown(self):self.temp.cleanup()

    def state(self,ids=None):
        ids=ids or [1,2,3,151645];record=native_record(ids)
        text=raw.canonical_text(self.tok.decode(ids,skip_special_tokens=True))
        action,ledger=raw.sampled_action(self.tok,record,ids+[151643]*(500-len(ids)),
            self.tok.encode(text,add_special_tokens=False),text,self.identity,
            response_mask=[1]*len(ids)+[0]*(500-len(ids)))
        state=State(0,[8],[8]+action,response=action.copy(),model_mask=[1]*len(action),steps=[0]*len(action))
        state.raw_ledger=[dict(turn=0,action=ledger,observation_token_ids=[],observation_text='',
            generation_context_token_ids=[8],generation_context_attention=[1],generation_context_positions=[0])]
        return state

    def item(self,state,projection):
        return NS(batch=dict(prompts=torch.tensor(state.prompt),responses=torch.tensor(state.response),
            attention_mask=torch.ones(len(state.prompt)+len(state.response),dtype=torch.long),
            info_mask=torch.tensor([1]*len(state.prompt)+state.model_mask)),
            non_tensor_batch={raw.FIELD:projection},meta_info={'tree_raw_token_protocol':raw.PROTOCOL})

    def test_scope_reuses_sampling_inherited_proof_and_preserves_output(self):
        actual=self.tok.backend_tokenizer.to_str
        with patch.object(self.tok.backend_tokenizer,'to_str',wraps=actual) as backend, \
                patch.object(raw,'_scan_utf8_prefixes',wraps=raw._scan_utf8_prefixes) as scan:
            with raw.tokenizer_validation_scope(self.tok,self.identity):
                state=self.state();child=state.branch()
                first=raw.trajectory_projection(state,4096)
                second=raw.trajectory_projection(child,4096)
                spans=raw.semantic_spans(self.item(child,second),self.tok)
                self.assertEqual(first,second);self.assertEqual(spans,['<answer>Paris</answer>'])
                self.assertEqual(first['policy_response_token_ids'],[1,2,3])
                self.assertEqual(child.raw_ledger[0]['action']['selected_logprobs'],[-.25,-.5,-.75])
                self.assertEqual(scan.call_count,1)
            self.assertEqual(backend.call_count,2)
        self.assertEqual(raw.trajectory_projection(state,4096),first)

    def test_incremental_prefix_hash_matches_full_json_for_invalid_utf8_and_escapes(self):
        ids=[200,151643,201,202,203,204,205,206,204,151645]
        state=self.state(ids);proof=state.raw_ledger[0]['action']['utf8_boundary_proof']
        expected=[raw.fingerprint(self.tok.decode(ids[:end],skip_special_tokens=True)) for end in range(1,len(ids)+1)]
        self.assertEqual(proof['native_prefix_agreement_sha256'],raw.fingerprint(expected))
        self.assertEqual(proof['native_prefix_agreement_count'],len(ids))
        self.assertTrue(proof['invalid_original_utf8_events']);self.assertTrue(proof['pending_prefix_events'])
        raw.validate_action_utf8_proof(state.raw_ledger[0]['action'],self.tok)

    def test_cached_content_and_resealed_semantic_mutations_refuse(self):
        with raw.tokenizer_validation_scope(self.tok,self.identity):
            state=self.state();raw.trajectory_projection(state,4096)
            for reseal in (False,True):
                changed=state.branch();action=changed.raw_ledger[0]['action']
                action['canonical_semantic_token_ids']=self.tok.encode('<answer>WRONG</answer>')
                action['legacy_reward_action_text']='<answer>WRONG</answer>'
                if reseal:
                    action['action_binding_sha256']=raw.fingerprint({k:v for k,v in action.items()
                        if not k.startswith('prefix_scan_') and k!='action_binding_sha256'})
                with self.assertRaises(ValueError):raw.trajectory_projection(changed,4096)
            changed=state.branch();changed.raw_ledger[0]['action']['canonical_semantic_token_ids'][0]=True
            with self.assertRaises(ValueError):raw.trajectory_projection(changed,4096)

    def test_live_backend_and_runtime_mutation_reject_reused_context(self):
        with raw.tokenizer_validation_scope(self.tok,self.identity) as context:state=self.state()
        for attribute,value in (('backend_normalizer',dict(type='Lowercase')),('split_special_tokens',True)):
            with self.subTest(attribute=attribute), patch.object(self.tok,attribute,value,create=True):
                with self.assertRaises(ValueError):
                    with raw.tokenizer_validation_scope(self.tok,self.identity,_context=context):
                        raw.trajectory_projection(state,4096)
        with patch.dict(self.tok.parts,{2:'WRONG'}),self.assertRaises(ValueError):
            with raw.tokenizer_validation_scope(self.tok,self.identity,_context=context):raw.trajectory_projection(state,4096)
        with patch.object(self.tok,'decode',lambda ids,**kwargs:'WRONG'),self.assertRaises(ValueError):
            with raw.tokenizer_validation_scope(self.tok,self.identity,_context=context):raw.trajectory_projection(state,4096)

    def test_new_sampling_keeps_every_native_prefix_agreement_check(self):
        original=self.tok.decode
        def changed(ids,skip_special_tokens=False):
            if skip_special_tokens and list(ids)==[1]:return 'wrong-prefix'
            return original(ids,skip_special_tokens)
        with patch.object(self.tok,'decode',changed),self.assertRaises(ValueError):self.state()

    def test_five_long_leaves_microcost_reports_real_reuse_counts(self):
        state=self.state([10]*497+[200,201,151645]);leaves=[state.branch() for _ in range(5)]
        actual=self.tok.backend_tokenizer.to_str;timings={};counts={};outputs={}
        for scoped in (False,True):
            label='scoped' if scoped else 'separate_calls'
            with patch.object(self.tok.backend_tokenizer,'to_str',wraps=actual) as backend, \
                    patch.object(raw,'_scan_utf8_prefixes',wraps=raw._scan_utf8_prefixes) as scan:
                started=time.process_time()
                if scoped:
                    with raw.tokenizer_validation_scope(self.tok,self.identity):
                        outputs[label]=[raw.trajectory_projection(leaf,4096) for leaf in leaves]
                else:outputs[label]=[raw.trajectory_projection(leaf,4096) for leaf in leaves]
                timings[label]=time.process_time()-started
                counts[label]=dict(backend_serializations=backend.call_count,utf8_scans=scan.call_count)
        self.assertEqual(outputs['scoped'],outputs['separate_calls'])
        self.assertEqual(counts['separate_calls'],dict(backend_serializations=5,utf8_scans=5))
        self.assertEqual(counts['scoped'],dict(backend_serializations=2,utf8_scans=1))
        print('TREE_DRIVER_CPU_MICROCOST='+json.dumps(dict(cpu_seconds=timings,counts=counts,synthetic_only=True)),flush=True)


if __name__=='__main__':unittest.main()
