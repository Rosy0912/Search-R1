"""Synthetic CPU contracts and actual source call-path fixtures; no services/assets."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType,SimpleNamespace as NS
import unittest
from unittest.mock import patch
from typing import Any,List,Tuple
from collections import defaultdict
import string

import numpy as np
import torch

from search_r1.baselines.tree_raw_tokens import (PROTOCOL,FIELD,EMISSION,capture_native_emissions,capture_native_sampling_settings,
    sampled_action,canonical_text,trajectory_projection,semantic_spans,tokenizer_identity,
    BYTELEVEL_DECODER_STATE,BYTELEVEL_INVERSE,validate_action_utf8_proof,fingerprint)
from search_r1.baselines.rollout import State,SearchRollout,object_array


def source_functions(path,klass,names,namespace):
    tree=ast.parse(Path(path).read_text())
    functions=[n for node in tree.body for n in (node.body if isinstance(node,ast.ClassDef) and node.name==klass else [])
               if isinstance(n,ast.FunctionDef) and n.name in names] if klass else [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(functions)==len(names)
    module=ast.Module(body=functions,type_ignores=[]);ast.fix_missing_locations(module)
    exec(compile(module,path,'exec'),namespace)
    return {name:namespace[name] for name in names}


class Tokenizer:
    pad_token_id=151643;eos_token_id=151645;bos_token_id=None;pad_token='<pad>'
    clean_up_tokenization_spaces=False
    all_special_ids=[151643,151645,152000]
    def __init__(self,path):
        self.name_or_path=str(path)
        self.parts={1:'<answer>',2:'Paris',3:'</answer>tail',4:'</an',5:'swer>',6:'later',
            7:'<search>capital</search>extra',8:'提示',9:'中文',10:'X',11:'Y'}
        self.enc={};self.next=100
        self.backend_padding=None;self.backend_truncation=None;self.backend_normalizer=None
        self.backend_tokenizer=NS(decoder=NS(__getstate__=lambda:json.dumps(BYTELEVEL_DECODER_STATE).encode()),
            to_str=self.backend_state)
    def backend_state(self):
        return json.dumps(dict(version='1.0',model=dict(type='SyntheticFixedByteVocabulary',
            byte_tokens=[1000+byte for byte in range(256)],pieces={str(k):(v if isinstance(v,bytes) else v.encode()).hex()
                for k,v in self.parts.items()}),normalizer=self.backend_normalizer,pre_tokenizer=None,
            post_processor=None,decoder=BYTELEVEL_DECODER_STATE,
            added_tokens=[dict(id=token,special=True) for token in self.all_special_ids],
            padding=self.backend_padding,truncation=self.backend_truncation))
    def convert_ids_to_tokens(self,token):
        piece=bytes([token-1000]) if 1000<=token<1256 else self.parts[token]
        data=piece if isinstance(piece,bytes) else piece.encode('utf-8')
        alphabet={byte:character for character,byte in BYTELEVEL_INVERSE.items()}
        return ''.join(alphabet[byte] for byte in data)
    def decode(self,ids,skip_special_tokens=False):
        ids=ids.tolist() if hasattr(ids,'tolist') else ids
        result=bytearray()
        for token in ids:
            if token in (self.pad_token_id,self.eos_token_id,152000):
                result.extend(b'' if skip_special_tokens else
                    ('<eos>' if token==self.eos_token_id else '<special>').encode())
            else:
                piece=bytes([token-1000]) if 1000<=token<1256 else self.parts[token]
                result.extend(piece if isinstance(piece,bytes) else piece.encode('utf-8'))
        return bytes(result).decode('utf-8',errors='replace')
    def batch_decode(self,rows,**kwargs):return [self.decode(row,**kwargs) for row in rows]
    def encode(self,text,add_special_tokens=False):
        return [1000+byte for byte in text.encode('utf-8')]
    def __call__(self,texts,**kwargs):
        rows=[self.encode(text) for text in texts];width=max(1,max(map(len,rows)))
        return {'input_ids':torch.tensor([row+[self.pad_token_id]*(width-len(row)) for row in rows])}


def params():
    return NS(n=1,best_of=1,_real_n=1,temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,frequency_penalty=0.,
        repetition_penalty=1.,min_tokens=0,max_tokens=500,ignore_eos=False,stop=None,stop_token_ids=[],
        include_stop_str_in_output=False,logprobs=1,prompt_logprobs=None,seed=None,logits_processors=None,
        truncate_prompt_tokens=None,guided_decoding=None,logit_bias=None,allowed_token_ids=None)


ENV={'SEARCH_R1_BASELINE_COMMIT':'a'*40,'SEARCH_R1_BASELINE_CONFIG_SHA256':'b'*64,
     'SEARCH_R1_BASELINE_DATA_IDENTITY':'c'*64,'SEARCH_R1_MODEL_IDENTITY':'d'*64,
     'SEARCH_R1_BASELINE_RUN_ID':'fixture-only','SLURM_JOB_ID':'0'}


def emission(ids,prompt=None,call='cpu-fixture',row=0):
    out=NS(token_ids=ids,logprobs=[{token:NS(logprob=-.25*(i+1))} for i,token in enumerate(ids)],
           finish_reason='stop' if ids and ids[-1]==151645 else 'length',stop_reason=None)
    request=NS(request_id='real-'+str(row),finished=True,outputs=[out],prompt_token_ids=prompt or [8])
    with patch.dict(os.environ,ENV):return capture_native_emissions([request],params(),call,0)[0]


class TinyProto:
    def __init__(self,batch,non_tensor_batch=None,meta_info=None):
        self.batch=batch;self.non_tensor_batch=non_tensor_batch or {};self.meta_info=meta_info or {}
    def __len__(self):return len(next(iter(self.batch.values())))
    @classmethod
    def from_dict(cls,tensors=None,non_tensors=None,meta_info=None):return cls(tensors,non_tensors,meta_info)
    def __getitem__(self,i):
        return NS(batch={k:v[i] for k,v in self.batch.items()},
            non_tensor_batch={k:v[i] for k,v in self.non_tensor_batch.items()},meta_info=self.meta_info)


class RawTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();path=Path(self.tmp.name)
        (path/'tokenizer.json').write_text('{"fixture":"synthetic"}')
        self.tok=Tokenizer(path);self.sha=hashlib.sha256((path/'tokenizer.json').read_bytes()).hexdigest()
        self.identity=tokenizer_identity(self.tok,self.sha)
        self.pinned_parts=copy.deepcopy(self.tok.parts)
        self.verl=ModuleType('verl');self.verl.DataProto=TinyProto
        self.fake_verl=patch.dict(sys.modules,{'verl':self.verl});self.fake_verl.start()
        self.ns={'torch':torch,'re':re,'List':List,'Any':Any,'Tuple':Tuple,'Dict':dict}
        self.methods=source_functions('search_r1/llm_agent/generation.py','LLMGenerationManager',
            ['_postprocess_responses','_process_next_obs','postprocess_predictions','execute_predictions'],self.ns)
        self.manager=NS(tokenizer=self.tok,config=NS(no_think_rl=False,max_obs_length=500,max_start_length=2048,
            max_prompt_length=4096,max_response_length=500,max_turns=4,num_gpus=1,force_final_answer=False))
        self.manager._batch_tokenize=lambda texts:self.tok(texts)['input_ids']
        for name,method in self.methods.items():setattr(self.manager,name,method.__get__(self.manager))
        self.manager.batch_search=lambda queries:['fixture-doc']*len(queries)
    def tearDown(self):self.fake_verl.stop();self.tmp.cleanup()
    def action(self,ids):
        # Test-native pieces are deliberately changed by several UTF8 fixtures.
        # Pin their declared fixed vocabulary before sampling the new fixture.
        if self.tok.parts!=self.pinned_parts:
            self.identity=tokenizer_identity(self.tok,self.sha);self.pinned_parts=copy.deepcopy(self.tok.parts)
        row=ids+[self.tok.pad_token_id]*(500-len(ids))
        legacy_ids,text=self.manager._postprocess_responses(torch.tensor([row]))
        return sampled_action(self.tok,emission(ids),row,legacy_ids[0][legacy_ids[0]!=self.tok.pad_token_id].tolist(),text[0],self.identity)
    def state(self,ids=None):
        action,ledger=self.action(ids or [1,2,3,self.tok.eos_token_id])
        state=State(0,[8],[8]+action,response=action.copy(),model_mask=[1]*len(action),steps=[0]*len(action))
        state.raw_ledger=[dict(turn=0,action=ledger,observation_token_ids=[],observation_text='',
            generation_context_token_ids=[8],generation_context_attention=[1],generation_context_positions=[0])]
        return state
    def item(self,state):
        projection=trajectory_projection(state,4096)
        return NS(batch=dict(prompts=torch.tensor(state.prompt),responses=torch.tensor(state.response),
            attention_mask=torch.ones(len(state.prompt)+len(state.response),dtype=torch.long),
            info_mask=torch.tensor([1]*len(state.prompt)+state.model_mask)),
            non_tensor_batch={FIELD:projection},meta_info={'tree_raw_token_protocol':PROTOCOL})
    def engine(self,raw_ids):
        manager=self.manager;calls=[]
        def generate(batch):
            calls.append(batch)
            records=[]
            for i in range(len(batch)):
                prompt=batch.batch['input_ids'][i][batch.batch['attention_mask'][i].bool()].tolist()
                records.append(emission(raw_ids,prompt,call='fixture-call-'+str(len(calls)),row=i))
            rows=torch.tensor([raw_ids+[self.tok.pad_token_id]*(500-len(raw_ids))]*len(batch))
            return TinyProto(dict(responses=rows),{EMISSION:object_array(records)})
        manager.actor_rollout_wg=NS(generate_sequences=generate)
        manager.tensor_fn=NS(create_position_ids=lambda mask:(mask.cumsum(-1)-1).clamp_min(0))
        engine=SearchRollout(manager,token_preservation={'tokenizer_sha256':self.sha})
        return engine,calls

    def test_whole_token_tag_tail_and_raw_scalar_prefix(self):
        action,ledger=self.action([1,2,3,151645])
        self.assertEqual(action,[1,2,3]);self.assertEqual(ledger['selected_logprobs'],[-.25,-.5,-.75])
        self.assertEqual(ledger['canonical_text'],'<answer>Paris</answer>')
        self.assertEqual(ledger['indivisible_extra_text'],'tail')
        self.assertNotEqual(action,ledger['canonical_semantic_token_ids'])

    def test_split_tags_search_priority_and_multiple_tag_rule_match_actual_old_source(self):
        action,ledger=self.action([1,2,4,5,6,151645]);self.assertEqual(action,[1,2,4,5])
        self.tok.parts[30]='<answer>A</answer><search>q</search>tail'
        _,ledger=self.action([30,151645])
        self.assertEqual(ledger['canonical_text'],'<answer>A</answer><search>q</search>')
        before=self.manager.postprocess_predictions([ledger['canonical_text']])
        after=self.manager.postprocess_predictions([canonical_text(self.tok.decode([30],skip_special_tokens=True))])
        self.assertEqual(before,after)

    def test_chinese_internal_special_eos_only_and_no_tag_keep_real_ids(self):
        action,ledger=self.action([9,152000,151645]);self.assertEqual(action,[9,152000,151645])
        self.assertEqual(ledger['canonical_text'],'中文')
        action,ledger=self.action([151645]);self.assertEqual(action,[151645]);self.assertTrue(ledger['empty_canonical'])
        self.assertEqual(ledger['canonical_semantic_token_ids'],[])

    def test_utf8_and_nonmonotonic_decode_refuse_without_substitute_tokens(self):
        for mode in ('replacement','nonmonotonic'):
            original=self.tok.decode
            def changed(ids,skip_special_tokens=False):
                if skip_special_tokens and list(ids)==[1]:return '\ufffd' if mode=='replacement' else 'not-a-prefix'
                return original(ids,skip_special_tokens)
            with patch.object(self.tok,'decode',changed),self.assertRaises(ValueError):self.action([1,2,3,151645])

    def test_split_utf8_2_3_4_byte_characters_and_all_original_scalars(self):
        for text in ('é','中','😀','é中😀'):
            with self.subTest(text=text):
                data=text.encode('utf-8');ids=list(range(200,200+len(data)))
                self.tok.parts.update({token:bytes([byte]) for token,byte in zip(ids,data)})
                action,ledger=self.action(ids+[151645])
                self.assertEqual(action,ids+[151645]);self.assertEqual(ledger['canonical_text'],text)
                self.assertEqual(ledger['selected_logprobs'],emission(action)['raw_selected_logprobs'])
                proof=ledger['utf8_boundary_proof']
                self.assertTrue(proof['pending_prefix_events']);self.assertTrue(proof['final_utf8_complete'])
                self.assertEqual(proof['native_prefix_agreement_count'],len(action))
                self.assertEqual(ledger['whole_prefix_decode'],self.tok.decode(action,skip_special_tokens=True))
                validate_action_utf8_proof(ledger,self.tok)

    def test_fixed_bytelevel_alphabet_against_independent_literal_token_pieces(self):
        # Literal ByteLevel spellings, independent of the fixture encoder.
        pieces={200:'Ã',201:'©',202:'ä',203:'¸',204:'Ń',205:'ð',206:'Ł',207:'ĺ',208:'Ģ'}
        data={200:b'\xc3',201:b'\xa9',202:b'\xe4',203:b'\xb8',204:b'\xad',
              205:b'\xf0',206:b'\x9f',207:b'\x98',208:b'\x80'}
        self.tok.parts.update(data)
        original=self.tok.convert_ids_to_tokens
        with patch.object(self.tok,'convert_ids_to_tokens',side_effect=lambda token:pieces.get(token) or original(token)):
            action,ledger=self.action(list(pieces)+[151645])
        self.assertEqual(ledger['canonical_text'],'é中😀');self.assertEqual(action,list(pieces)+[151645])
        self.assertEqual({c:BYTELEVEL_INVERSE[c] for c in ('Ġ','Ċ','Ā','ġ','Ģ','Ł','ł','Ń')},
                         {'Ġ':32,'Ċ':10,'Ā':0,'ġ':127,'Ģ':128,'Ł':159,'ł':160,'Ń':173})

    def test_every_legal_partition_of_single_unicode_char_is_complete_at_same_text(self):
        for text in ('é','中','😀'):
            data=text.encode('utf-8')
            for mask in range(1<<(len(data)-1)):
                ends=[i+1 for i in range(len(data)-1) if mask&(1<<i)]+[len(data)]
                starts=[0]+ends[:-1];ids=list(range(200,200+len(ends)))
                self.tok.parts.update({token:data[start:end] for token,start,end in zip(ids,starts,ends)})
                with self.subTest(text=text,ends=ends):
                    action,ledger=self.action(ids+[151645])
                    self.assertEqual(action,ids+[151645]);self.assertEqual(ledger['canonical_text'],text)
                    validate_action_utf8_proof(ledger,self.tok)

    def test_tag_and_utf8_share_original_token_retains_until_complete_boundary(self):
        self.tok.parts.update({200:b'<answer>Paris</answer>\xe4',201:b'\xb8',202:b'\xad',203:'later'})
        action,ledger=self.action([200,201,202,203,151645])
        self.assertEqual(action,[200,201,202]);self.assertEqual(ledger['canonical_text'],'<answer>Paris</answer>')
        self.assertEqual(ledger['indivisible_extra_text'],'中')
        self.assertEqual(ledger['utf8_boundary_proof']['complete_original_token_boundaries'],[3,4,5])
        self.assertEqual(ledger['selected_logprobs'],[-.25,-.5,-.75])
        validate_action_utf8_proof(ledger,self.tok)

    def test_utf8_search_priority_tool_and_legacy_reward_text_unchanged(self):
        self.tok.parts.update({200:b'<answer>A</answer><search>\xe4',201:b'\xb8\xad</search>tail'})
        action,ledger=self.action([200,201,151645])
        self.assertEqual(action,[200,201])
        self.assertEqual(ledger['canonical_text'],'<answer>A</answer><search>中</search>')
        # Canonical truncation is Search-first; the original tool parser still
        # selects the first complete tag, which is the answer in this example.
        self.assertEqual(self.manager.postprocess_predictions([ledger['canonical_text']]),(['answer'],['A']))
        self.assertEqual(ledger['legacy_reward_action_text'],ledger['canonical_text'])
        self.assertEqual(ledger['indivisible_extra_text'],'tail')
        validate_action_utf8_proof(ledger,self.tok)

    def test_special_token_between_utf8_bytes_is_skipped_only_in_semantic_decode(self):
        self.tok.parts.update({200:b'\xe4',201:b'\xb8\xad'})
        action,ledger=self.action([200,152000,201,151645])
        self.assertEqual(action,[200,152000,201,151645]);self.assertEqual(ledger['canonical_text'],'中')
        self.assertEqual(ledger['utf8_boundary_proof']['complete_original_token_boundaries'],[3,4])
        self.assertEqual([x['pending_utf8_hex'] for x in ledger['utf8_boundary_proof']['pending_prefix_events']],['e4','e4'])

    def test_invalid_terminal_utf8_and_real_replacement_fail_closed(self):
        for data in (b'\xff',b'\x80',b'\xc0\xaf',b'\xed\xa0\x80',b'\xf4\x90\x80\x80',
                     b'\xe4X',b'\xe4',b'\xf0\x9f',b'\xef\xbf\xbd',b'</answer>\xe4'):
            with self.subTest(data=data):
                self.tok.parts[200]=data
                with self.assertRaises(ValueError):self.action([200,151645])

    def test_unsupported_decoder_cleanup_and_original_piece_mismatch_refuse(self):
        for state in ({'type':'Metaspace'},dict(BYTELEVEL_DECODER_STATE,use_regex=True),
                      dict(BYTELEVEL_DECODER_STATE,trim_offsets=0)):
            decoder=NS(__getstate__=lambda:json.dumps(state).encode())
            with self.subTest(state=state),patch.object(self.tok.backend_tokenizer,'decoder',decoder):
                with self.assertRaises(ValueError):tokenizer_identity(self.tok,self.sha)
                with self.assertRaises(ValueError):self.action([1,2,3,151645])
        with patch.object(self.tok,'clean_up_tokenization_spaces',True),self.assertRaises(ValueError):
            self.action([1,2,3,151645])
        with patch.object(self.tok,'convert_ids_to_tokens',return_value='中'),self.assertRaises(ValueError):
            self.action([1,2,3,151645])
        with patch.object(self.tok,'convert_ids_to_tokens',return_value='X'),self.assertRaises(ValueError):
            self.action([1,2,3,151645])

    def test_pending_native_output_and_frozen_special_identity_cannot_be_suppressed(self):
        self.tok.parts.update({200:b'\xe4',201:b'\xb8\xad'})
        original=self.tok.decode
        def suppressed(ids,skip_special_tokens=False):
            if skip_special_tokens and list(ids)==[200]:return ''
            return original(ids,skip_special_tokens)
        with patch.object(self.tok,'decode',suppressed),self.assertRaises(ValueError):self.action([200,201,151645])
        with patch.object(self.tok,'all_special_ids',[151643,151645]),self.assertRaises(ValueError):
            self.action([200,201,151645])
        _,ledger=self.action([200,201,151645])
        def changed_full(ids,skip_special_tokens=False):
            if skip_special_tokens and list(ids)==[200,201,151645]:return 'wrong-full-text'
            return original(ids,skip_special_tokens)
        with patch.object(self.tok,'decode',changed_full),self.assertRaises(ValueError):
            validate_action_utf8_proof(ledger,self.tok)

    def test_rebound_utf8_proof_and_incomplete_cut_refused_by_validator(self):
        self.tok.parts.update({200:b'<answer>Paris</answer>\xe4',201:b'\xb8\xad'})
        _,ledger=self.action([200,201,151645])
        for mutation in ('pending','boundaries','agreement','cut'):
            changed=copy.deepcopy(ledger)
            if mutation=='pending':changed['utf8_boundary_proof']['pending_prefix_events'][0]['pending_utf8_hex']='f0'
            elif mutation=='boundaries':changed['utf8_boundary_proof']['complete_original_token_boundaries']=[1,2,3]
            elif mutation=='agreement':changed['utf8_boundary_proof']['native_prefix_agreement_sha256']='0'*64
            else:changed['cut']=1
            changed['action_binding_sha256']=fingerprint({k:v for k,v in changed.items()
                if not k.startswith('prefix_scan_') and k!='action_binding_sha256'})
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_action_utf8_proof(changed,self.tok)

    def test_split_utf8_trajectory_and_actual_reward_use_legacy_semantic_ids(self):
        self.tok.parts.update({200:b'<answer>\xe4',201:b'\xb8\xad</answer>tail'})
        state=self.state([200,201,151645]);item=self.item(state)
        self.assertEqual(semantic_spans(item,self.tok),['<answer>中</answer>'])
        self.assertEqual(state.response,[200,201])
        self.assertNotEqual(item.non_tensor_batch[FIELD]['semantic_response_token_ids'],state.response)
        item.non_tensor_batch[FIELD]['raw_ledger'][0]['action']['utf8_boundary_proof']['pending_prefix_events'][0]['pending_utf8_hex']='ff'
        with self.assertRaises(ValueError):semantic_spans(item,self.tok)

    def test_pad_equals_eos_genuine_pad_missing_length_empty_and_over500_refuse(self):
        with patch.object(self.tok,'pad_token_id',151645),self.assertRaises(ValueError):self.action([151645])
        with self.assertRaises(ValueError):self.action([1,151643,2,3,151645])
        with self.assertRaises(ValueError):emission([])
        with self.assertRaises(ValueError):emission([10]*501)
        record=emission([1,2,3,151645]);del record['true_emitted_length']
        with self.assertRaises((KeyError,ValueError)):sampled_action(self.tok,record,[1,2,3,151645]+[151643]*496,
            self.tok.encode('<answer>Paris</answer>'),'<answer>Paris</answer>',self.identity)

    def test_exact500_no_tag_preserves_all_emitted_and_identity_duplicate_native_refuses(self):
        action,ledger=self.action([10]*500);self.assertEqual(len(action),500);self.assertEqual(ledger['discarded_sampled_tail_tokens'],0)
        out=NS(token_ids=[151645],logprobs=[{151645:NS(logprob=-.5)}],finish_reason='stop',stop_reason=None)
        request=NS(request_id='same',finished=True,outputs=[out],prompt_token_ids=[8])
        with patch.dict(os.environ,ENV),self.assertRaises(ValueError):capture_native_emissions([request,request],params(),'fixture',0)

    def test_semantic_projection_preserves_old_reward_and_refuses_binding_changes(self):
        from search_r1.qa_protocol import extract_final_answer
        state=self.state();item=self.item(state)
        spans=semantic_spans(item,self.tok)
        self.assertEqual(extract_final_answer(spans),'Paris')
        self.assertIsNone(extract_final_answer([self.tok.decode(state.response)]))
        old=source_functions('verl/trainer/main_ppo.py',None,['decode_model_response_spans'],{})['decode_model_response_spans']
        legacy=state.raw_ledger[0]['action']['canonical_semantic_token_ids']
        self.assertEqual(spans,old(self.tok,torch.tensor(legacy),torch.ones(len(legacy),dtype=torch.long)))
        item.batch['responses'][0]=10
        with self.assertRaises(ValueError):semantic_spans(item,self.tok)

    def test_parent_branch_deepcopies_raw_and_semantics_ids_without_resampling(self):
        parent=self.state();child=parent.branch()
        self.assertEqual(parent.raw_ledger,child.raw_ledger);self.assertNotEqual(parent.trajectory,child.trajectory)
        child.raw_ledger[0]['action']['selected_logprobs'][0]=-99.
        self.assertEqual(parent.raw_ledger[0]['action']['selected_logprobs'][0],-.25)
        with self.assertRaises(ValueError):trajectory_projection(child,4096)

    def test_projection_rejects_scoring_prefix_positions_or_cap_ambiguity(self):
        for kind in ('context','position','policycap','semanticcap'):
            state=self.state()
            if kind=='context':state.raw_ledger[0]['generation_context_token_ids']=[9]
            if kind=='position':state.raw_ledger[0]['generation_context_positions']=[1]
            cap=2 if kind in ('policycap','semanticcap') else 4096
            with self.subTest(kind=kind),self.assertRaises(ValueError):trajectory_projection(state,cap)

    def test_actual_rollout_source_calls_env_once_and_keeps_raw_prefix_in_model_inputs(self):
        engine,calls=self.engine([1,2,3,151645]);state=State(0,[8],[8],observation=[8])
        engine.advance([state],{})
        self.assertEqual(len(calls),1);self.assertEqual(state.response,[1,2,3]);self.assertTrue(state.done)
        output,_=engine.compose([state])
        self.assertEqual(output.batch['responses'][0].tolist(),[1,2,3])
        self.assertEqual(semantic_spans(output[0],self.tok),['<answer>Paris</answer>'])
        self.assertEqual(engine.cost['raw_emitted_tokens'],4);self.assertEqual(engine.cost['raw_retained_tokens'],3)
        self.assertEqual(engine.cost['raw_discarded_tail_tokens'],1)

    def test_actual_rollout_refuses_rolling_leftcrop_and_missing_emission_metadata(self):
        engine,_=self.engine([1,2,3,151645]);self.manager.config.max_prompt_length=3
        with self.assertRaises(ValueError):engine.advance([State(0,[8],[8])],{})
        engine,_=self.engine([1,2,3,151645])
        self.manager.actor_rollout_wg.generate_sequences=lambda batch:TinyProto({'responses':torch.tensor([[151643]*500])})
        with self.assertRaises(ValueError):engine.advance([State(0,[8],[8])],{})

    def test_actual_observation_cap500_keeps_observation_mask0_and_full_next_prefix(self):
        engine,calls=self.engine([7,151645]);state=State(0,[8],[8],observation=[8])
        original=self.tok.__class__.__call__
        def long_observation(tokenizer,texts,**kwargs):
            if all(text.lstrip().startswith('<information>') for text in texts):
                return {'input_ids':torch.tensor([[10]*501]*len(texts))}
            return original(tokenizer,texts,**kwargs)
        with patch.object(self.tok.__class__,'__call__',long_observation):
            engine.advance([state],{})
            self.assertEqual(state.response,[7]+[10]*500)
            self.assertEqual(state.model_mask,[1]+[0]*500)
            engine.advance([state],{})
        self.assertEqual(calls[1].batch['input_ids'][0].tolist(),[8,7]+[10]*500)
        output,_=engine.compose([state])
        self.assertEqual(len(state.raw_ledger[0]['observation_token_ids']),500)
        self.assertEqual(semantic_spans(output[0],self.tok),[
            '<search>capital</search>','<search>capital</search>'])

    def test_actual_tree_round_inherits_sample_ids_and_equal_leaf_weights_unchanged(self):
        engine,calls=self.engine([7,151645])
        engine.rng.choice=lambda nodes:nodes[-1]
        inputs=TinyProto(dict(input_ids=torch.tensor([[8],[8]]),attention_mask=torch.ones((2,1),dtype=torch.long)))
        output,sources=engine.run(inputs,NS(roots=2,rounds=1,expansions=2,leaves=3))
        self.assertEqual(len(output),6);self.assertEqual(sources.tolist(),[0,0,0,1,1,1])
        self.assertEqual(len(calls),5);self.assertEqual(engine.cost['retrieval_calls'],12)
        self.assertEqual(engine.cost['raw_inherited_leaf_tokens'],12)
        function=source_functions('search_r1/baselines/objectives.py',None,['attach_policy_weights'],
            dict(torch=torch,defaultdict=defaultdict))['attach_policy_weights']
        function(output,'tree_grpo')
        self.assertTrue(torch.equal(output.batch['policy_weights'].sum(-1),torch.ones(6)))
        self.assertTrue(all(len(s.raw_ledger)==4 and sum(s.model_mask)==4 for s in engine.last_states))

    def test_actual_rewardmanager_and_evaluation_hooks_preserve_legacy_qa_score(self):
        from search_r1.qa_protocol import extract_final_answer
        from search_r1.baselines.bapo import answer_from_spans,token_f1,legal_action_records
        qa=source_functions('verl/utils/reward_score/qa_em.py',None,
            ['normalize_answer','em_check','compute_score_generated_em'],
            dict(re=re,string=string,extract_final_answer=extract_final_answer))
        namespace=dict(torch=torch,DataProto=TinyProto,qa_em=NS(compute_score_generated_em=qa['compute_score_generated_em']))
        methods=source_functions('verl/trainer/main_ppo.py',None,
            ['decode_model_response_spans','_select_rm_score_fn','RewardManager'],namespace)
        engine,_=self.engine([1,2,3,151645]);state=State(0,[8],[8],observation=[8]);engine.advance([state],{})
        output,_=engine.compose([state])
        output.non_tensor_batch['reward_model']=object_array([{'ground_truth':{'target':'Paris'}}])
        output.non_tensor_batch['data_source']=object_array(['nq'])
        result=methods['RewardManager'](self.tok,0)(output)
        self.assertEqual(float(result.sum()),1.)
        original_spans=methods['decode_model_response_spans'](self.tok,torch.tensor(self.tok.encode('<answer>Paris</answer>')),torch.tensor([1]))
        self.assertEqual(qa['compute_score_generated_em'](original_spans,{'target':'Paris'}),1.)
        fake_main=ModuleType('verl.trainer.main_ppo');fake_main.decode_model_response_spans=methods['decode_model_response_spans']
        evaluation=source_functions('search_r1/baselines/evaluation.py',None,['batch_answer_metrics'],
            dict(__package__='search_r1.baselines',re=re,answer_from_spans=answer_from_spans,
                 token_f1=token_f1,legal_action_records=legal_action_records,em_check=qa['em_check']))['batch_answer_metrics']
        with patch.dict(sys.modules,{'verl.trainer.main_ppo':fake_main}):rows=evaluation(output,self.tok)
        self.assertEqual((rows[0]['f1'],rows[0]['em'],rows[0]['generated_tokens']),(1.,1.,3))

    def test_actual_facade_keeps_lengths_before_padding_and_legacy_return_arity(self):
        from torch.nn.utils.rnn import pad_sequence
        events=[]
        def observe_native(*args):
            events.append('native_capture')
            return capture_native_emissions(*args)
        def observe_pad(*args,**kwargs):
            if facade._tree_emission_call is not None:self.assertEqual(events[0],'native_capture')
            events.append('padding')
            return pad_sequence(*args,**kwargs)
        function=source_functions('verl/third_party/vllm/vllm_v_0_6_3/llm.py','LLM',['_post_process_outputs'],
            dict(torch=torch,pad_sequence=observe_pad,List=List,Tuple=Tuple,RequestOutput=object))['_post_process_outputs']
        ids=[1,2,3,151645]
        completion=NS(token_ids=ids,logprobs=[{token:NS(logprob=-.25)} for token in ids],finish_reason='stop',stop_reason=None)
        request=NS(request_id='actual9',finished=True,outputs=[completion],prompt_token_ids=[8])
        facade=NS(llm_engine=NS(tokenizer=self.tok),_tree_emission_call='fixture',_tree_emission_params=params())
        facade._tree_emission_before_settings=capture_native_sampling_settings(facade._tree_emission_params)
        with patch.dict(os.environ,ENV),patch.object(torch.distributed,'get_rank',return_value=0),patch(
                'search_r1.baselines.tree_raw_tokens.capture_native_emissions',side_effect=observe_native):
            result=function(facade,[request])
        self.assertEqual(len(result),3);self.assertEqual(result[2][0]['true_emitted_length'],4)
        self.assertEqual(result[2][0]['raw_emitted_token_ids'],ids)
        facade._tree_emission_call=None
        self.assertEqual(len(function(facade,[request])),2)

    def test_actual_facade_pre_post_control_binding_and_cleanup(self):
        from torch.nn.utils.rnn import pad_sequence
        methods=source_functions('verl/third_party/vllm/vllm_v_0_6_3/llm.py','LLM',
            ['generate_with_emission_metadata','_post_process_outputs'],
            dict(torch=torch,pad_sequence=pad_sequence,List=List,Tuple=Tuple,RequestOutput=object))
        completion=NS(token_ids=[151645],logprobs=[{151645:NS(logprob=-.25)}],finish_reason='stop',stop_reason=None)
        request=NS(request_id='pre-post-controls',finished=True,outputs=[completion],prompt_token_ids=[8])
        for changed in (None,('temperature',2.),('best_of',2),('guided_decoding',{'json':{}}),
                        ('logit_bias',{151645:1.}),('allowed_token_ids',[151645])):
            calls=[];facade=NS(llm_engine=NS(tokenizer=self.tok))
            def generate(**kwargs):
                calls.append('native-entry')
                if changed:setattr(kwargs['sampling_params'],changed[0],changed[1])
                return methods['_post_process_outputs'](facade,[request])
            facade.generate=generate
            with self.subTest(changed=changed),patch.dict(os.environ,ENV),patch.object(torch.distributed,'get_rank',return_value=0):
                if changed:
                    with self.assertRaises(ValueError):methods['generate_with_emission_metadata'](facade,sampling_params=params())
                else:
                    result=methods['generate_with_emission_metadata'](facade,sampling_params=params())
                    self.assertEqual(len(result),3)
                    self.assertEqual(len(result[2][0]['sampling_params']),24)
                self.assertEqual(calls,['native-entry'])
                self.assertIsNone(facade._tree_emission_call)
                self.assertIsNone(facade._tree_emission_params)
                self.assertIsNone(facade._tree_emission_before_settings)


if __name__=='__main__':unittest.main()
