"""Independent exact frozen source CPU counterexamples; synthetic bytes only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'work/tree-behavior-mu-utf8-v2'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
os.chdir(SOURCE)
import search_r1.baselines.tree_raw_tokens as raw

# Independent reversible byte alphabet, not borrowed from the implementation.
visible=[b for b in range(256) if 33<=b<=126 or 161<=b<=172 or 174<=b<=255]
alphabet={b:chr(b) for b in visible}
alphabet.update({b:chr(256+i) for i,b in enumerate(b for b in range(256) if b not in visible)})
class Tokenizer:
    pad_token_id=151643;eos_token_id=151645;bos_token_id=None
    clean_up_tokenization_spaces=False;all_special_ids=[151643,151645]
    def __init__(self,path,pieces):
        self.name_or_path=str(path);self.pieces=pieces.copy();self.padding=None;self.truncation=None
        self.normalizer=None;self.encoder_calls=0
        self.backend_tokenizer=NS(decoder=NS(__getstate__=lambda:b'{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}'),to_str=self.to_str)
    def to_str(self):
        self.encoder_calls+=1
        return json.dumps(dict(model=dict(type='synthetic_byte_vocab',pieces={str(k):v.hex() for k,v in self.pieces.items()},byte_vocab=list(range(256))),normalizer=self.normalizer,
            pre_tokenizer=None,post_processor=None,decoder=dict(type='ByteLevel',add_prefix_space=False,trim_offsets=False,use_regex=False),
            added_tokens=[dict(id=token,special=True) for token in self.all_special_ids],padding=self.padding,truncation=self.truncation))
    def piece(self,token):return bytes([token-1000]) if 1000<=token<1256 else self.pieces[token]
    def convert_ids_to_tokens(self,token):return ''.join(alphabet[b] for b in self.piece(token))
    def decode(self,ids,skip_special_tokens=False):
        return b''.join((b'' if skip_special_tokens else b'<SPECIAL>') if token in self.all_special_ids else self.piece(token) for token in ids).decode('utf8',errors='replace')
    def encode(self,text,add_special_tokens=False):return [1000+b for b in text.encode('utf8')]

def rehash_action(action):
    action['action_binding_sha256']=raw.fingerprint({k:v for k,v in action.items() if not k.startswith('prefix_scan_') and k!='action_binding_sha256'})
def action(tok,ids,identity):
    emission=dict(protocol=raw.PROTOCOL,schema_version=1,raw_emitted_token_ids=ids,
        raw_selected_logprobs=[-.25*(i+1) for i in range(len(ids))],true_emitted_length=len(ids),
        sampling_identity={'independent_tiny_only':True},prompt_token_ids=[42])
    emission['sample_id']=raw.fingerprint(emission['sampling_identity'])
    text=raw.canonical_text(tok.decode(ids,skip_special_tokens=True))
    return raw.sampled_action(tok,emission,ids+[151643]*(500-len(ids)),tok.encode(text),text,identity)[1]
def state(act):
    return NS(prompt=[42],response=act['action_token_ids'].copy(),model_mask=[1]*act['cut'],
        raw_ledger=[dict(turn=0,action=act,observation_token_ids=[],observation_text='',generation_context_token_ids=[42],
            generation_context_attention=[1],generation_context_positions=[0])])
def item(st,projection):
    return NS(batch=dict(prompts=torch.tensor(st.prompt),responses=torch.tensor(st.response),
        attention_mask=torch.ones(len(st.prompt)+len(st.response),dtype=torch.long),
        info_mask=torch.tensor([1]*len(st.prompt)+st.model_mask)),non_tensor_batch={raw.FIELD:projection},
        meta_info={'tree_raw_token_protocol':raw.PROTOCOL})
def outcome(call):
    try:return dict(accepted=True,result=call())
    except Exception as exc:return dict(accepted=False,error_type=type(exc).__name__,error=str(exc))

started=(time.perf_counter(),time.process_time())
result=dict(actual_gpu_executed=False,actual_remote_executed=False,actual_slurm_executed=False,
    actual_official_assets_loaded=False,source_commit='02b1ea2688ff3fb032ee6699003a7e09d651f166',
    raw_helper_sha256=hashlib.sha256(Path(raw.__file__).read_bytes()).hexdigest(),cases={})
with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary);payload=b'{"independent_synthetic_tokenizer":true}';(path/'tokenizer.json').write_bytes(payload)
    tok=Tokenizer(path,{1:b'<answer>\xe4',2:b'\xb8\xad</answer>tail',42:b'prompt'})
    identity=raw.tokenizer_identity(tok,hashlib.sha256(payload).hexdigest());act=action(tok,[1,2,151645],identity)
    st=state(act);proj=raw.trajectory_projection(st,4096)
    baseline=raw.semantic_spans(item(st,proj),tok);assert baseline==['<answer>中</answer>']
    result['baseline']=dict(spans=baseline,policy_ids=st.response,mu_scalars=act['selected_logprobs'],cut=act['cut'],pending_utf8=act['utf8_boundary_proof']['pending_prefix_events'])
    for fields in ('ids','reward','both'):
        changed=json.loads(json.dumps(act));fake='<answer>WRONG</answer>'
        if fields in ('ids','both'):changed['canonical_semantic_token_ids']=tok.encode(fake)
        if fields in ('reward','both'):changed['legacy_reward_action_text']=fake
        rehash_action(changed);badstate=state(changed)
        badproj=json.loads(json.dumps(proj));badproj['raw_ledger'][0]['action']=changed
        badproj['semantic_response_token_ids']=changed['canonical_semantic_token_ids'].copy();badproj['semantic_model_mask']=[1]*len(badproj['semantic_response_token_ids'])
        badproj['binding_sha256']=raw.fingerprint({k:v for k,v in badproj.items() if k!='binding_sha256'})
        checks={key:outcome(fn) for key,fn in dict(action=lambda:raw.validate_action_utf8_proof(changed,tok),
            trajectory=lambda:raw.trajectory_projection(badstate,4096),reward=lambda:raw.semantic_spans(item(badstate,badproj),tok)).items()}
        assert all(not value['accepted'] and 'Legacy IDs/reward' in value['error'] for value in checks.values())
        assert changed['action_token_ids']==act['action_token_ids'] and changed['selected_logprobs']==act['selected_logprobs'] and changed['utf8_boundary_proof']==act['utf8_boundary_proof']
        result['cases']['resealed_reserialized_'+fields]=checks
    # A second shared ancestor turn remains legitimate and preserves policy bytes.
    chain=copy.deepcopy(st);obs=[1088];chain.raw_ledger[0].update(observation_token_ids=obs,observation_text='X')
    chain.response+=obs;chain.model_mask+=[0]
    second=copy.deepcopy(act);context=chain.prompt+chain.response;second['native_emission']['prompt_token_ids']=context;rehash_action(second)
    chain.raw_ledger.append(dict(turn=1,action=second,observation_token_ids=[],observation_text='',generation_context_token_ids=context,
        generation_context_attention=[1]*len(context),generation_context_positions=list(range(len(context)))))
    chain.response+=second['action_token_ids'];chain.model_mask+=[1]*second['cut']
    before=copy.deepcopy(chain);calls=tok.encoder_calls;t0=time.process_time()
    chainproj=raw.trajectory_projection(chain,4096);cpu=time.process_time()-t0
    assert tok.encoder_calls-calls==1 and chain.response==before.response and chain.model_mask==before.model_mask
    assert raw.semantic_spans(item(chain,chainproj),tok)==['<answer>中</answer>']*2
    result['cases']['legal_shared_ancestor']=dict(policy_ids=chain.response,spans=['<answer>中</answer>']*2,backend_serializations_per_trajectory=1,actual_process_cpu_seconds=cpu)
    # Normal per-call padding/truncation options are irrelevant to semantic encode.
    tok.padding={'length':500};tok.truncation={'max_length':500};assert raw.trajectory_projection(st,4096)==proj
    result['cases']['normal_padding_truncation']=dict(accepted=True)
    def drift(label,mutate,restore):
        mutate();check=outcome(lambda:raw.trajectory_projection(st,4096));restore()
        assert not check['accepted'];result['cases']['live_drift_'+label]=check
    alternate=path/'alternate';alternate.mkdir();(alternate/'tokenizer.json').write_bytes(payload)
    drift('path',lambda:setattr(tok,'name_or_path',str(alternate)),lambda:setattr(tok,'name_or_path',str(path)))
    drift('same_length_file',lambda:(path/'tokenizer.json').write_bytes(payload.replace(b'true',b'null')),lambda:(path/'tokenizer.json').write_bytes(payload))
    drift('backend_vocabulary',lambda:tok.pieces.update({42:b'changed'}),lambda:tok.pieces.update({42:b'prompt'}))
    drift('backend_normalizer',lambda:setattr(tok,'normalizer',{'type':'Lowercase'}),lambda:setattr(tok,'normalizer',None))
    drift('default_decode_flag',lambda:setattr(tok,'clean_up_tokenization_spaces',True),lambda:setattr(tok,'clean_up_tokenization_spaces',False))
    with patch.dict(raw._VALIDATION_TOKENIZERS,{},clear=True):
        missing=outcome(lambda:raw.trajectory_projection(st,4096));assert not missing['accepted'] and 'not registered' in missing['error']
        result['cases']['no_registry_no_ledger_path_load']=missing
    altered=json.loads(json.dumps(act));altered['utf8_boundary_proof']['pending_prefix_events'][0]['pending_utf8_hex']='ff';rehash_action(altered)
    proof=outcome(lambda:raw.trajectory_projection(state(altered),4096));assert not proof['accepted'] and 'UTF8 proof' in proof['error'];result['cases']['resealed_pending_proof']=proof
    # Fresh registration of each actual byte vocabulary ensures these are scanner
    # tests, not stale-backend rejections. Every multibyte cut is covered.
    matrix=[]
    for text in ('¢','é','中','€','한','😀','𐍈','\U0010ffff'):
        encoded=text.encode('utf8')
        for split in range(1,len(encoded)):
            pieces={1:b'<answer>'+encoded[:split],2:encoded[split:]+b'</answer>tail',42:b'prompt'}
            fresh=Tokenizer(path,pieces);freshid=raw.tokenizer_identity(fresh,hashlib.sha256(payload).hexdigest())
            legal=action(fresh,[1,2,151645],freshid);assert legal['action_token_ids']==[1,2] and legal['canonical_text']=='<answer>'+text+'</answer>'
            raw.validate_action_utf8_proof(legal,fresh);matrix.append(dict(text=text,split=split,accepted=True,cut=2))
    result['cases']['fresh_backend_multibyte_split_matrix']=matrix
    invalid=[]
    for label,data in [('invalid_lead',b'\xff'),('orphan_continuation',b'\x80'),('overlong',b'\xc0\xaf'),('surrogate',b'\xed\xa0\x80'),('past_max',b'\xf4\x90\x80\x80'),('terminal_pending',b'\xe4')]:
        fresh=Tokenizer(path,{1:b'<answer>'+data,2:b'',42:b'prompt'});freshid=raw.tokenizer_identity(fresh,hashlib.sha256(payload).hexdigest())
        check=outcome(lambda:action(fresh,[1,151645],freshid));assert not check['accepted'] and 'UTF8' in check['error'] and 'identity drifted' not in check['error']
        invalid.append(dict(label=label,**check))
    result['cases']['fresh_backend_illegal_byte_scan']=invalid
result['process_cpu_seconds']=time.process_time()-started[1];result['wall_seconds']=time.perf_counter()-started[0]
result['all_expected_cases_passed']=True
(OUT/'INDEPENDENT_UTF8_SEMANTIC_PROBE_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
print(json.dumps(dict(status='PASSED',cases=len(result['cases']),process_cpu_seconds=result['process_cpu_seconds'],wall_seconds=result['wall_seconds'],official_tokenizer_compatibility_accepted=False)))
