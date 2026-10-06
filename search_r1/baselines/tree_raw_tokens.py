"""Tree-only sampled-token prefixes plus separately bound legacy semantics.

No token/probability is repaired, no optimizer is called and no probability
tolerance is defined here. Unsupported boundaries reject the current batch.
"""
import codecs
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


PROTOCOL = 'tree_sampled_prefix_legacy_semantics_v1'
FIELD = 'tree_raw_semantics'
EMISSION = '__tree_native_emission'
UTF8_BOUNDARY_PROTOCOL = 'tree_original_bytelevel_utf8_prefix_v1'
BYTELEVEL_DECODER_STATE = dict(type='ByteLevel',add_prefix_space=False,trim_offsets=False,use_regex=False)


def require(value,message):
    if not value:raise ValueError('Tree raw-token protocol: '+message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _bytelevel_alphabet():
    # The reversible ByteLevel alphabet uses visible Latin-1 first, then maps
    # the remaining bytes to consecutive codepoints starting at U+0100.
    visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
    codepoints=visible.copy();extra=0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte);codepoints.append(256+extra);extra+=1
    return {chr(codepoint):byte for byte,codepoint in zip(visible,codepoints)}


BYTELEVEL_INVERSE = _bytelevel_alphabet()
BYTELEVEL_ALPHABET_SHA256 = fingerprint(sorted(BYTELEVEL_INVERSE.items(),key=lambda pair:pair[1]))


def _decoder_binding(tokenizer):
    require(tokenizer.clean_up_tokenization_spaces is False,'UTF8 proof requires cleanup=false')
    backend=getattr(tokenizer,'backend_tokenizer',None)
    decoder=getattr(backend,'decoder',None)
    require(decoder is not None and callable(getattr(decoder,'__getstate__',None)),
            'Explicit native ByteLevel decoder state required')
    try:
        state=decoder.__getstate__()
        if isinstance(state,(str,bytes)):state=json.loads(state)
    except (TypeError,ValueError,UnicodeError) as error:
        raise ValueError('Tree raw-token protocol: Decoder state cannot be proven') from error
    require(type(state) is dict and state==BYTELEVEL_DECODER_STATE and
            all(type(state[key]) is bool for key in ('add_prefix_space','trim_offsets','use_regex')),
            'Unsupported native decoder state; exact frozen ByteLevel required')
    special=getattr(tokenizer,'all_special_ids',None)
    require(isinstance(special,(list,tuple)) and all(type(x) is int for x in special) and
            len(set(special))==len(special),'Explicit native special-token IDs required')
    require(callable(getattr(tokenizer,'convert_ids_to_tokens',None)),
            'Native original token pieces required for UTF8 proof')
    return dict(protocol=UTF8_BOUNDARY_PROTOCOL,decoder_state=copy.deepcopy(state),
        decoder_state_sha256=fingerprint(state),bytelevel_alphabet_sha256=BYTELEVEL_ALPHABET_SHA256,
        skip_special_token_ids=sorted(special))


def _scan_utf8_prefixes(tokenizer,ids,identity,check_native_prefixes=True):
    binding=_decoder_binding(tokenizer)
    require(identity.get('utf8_decoder_binding')==binding and
            identity.get('tokenizer_class')==type(tokenizer).__module__+'.'+type(tokenizer).__qualname__ and
            identity.get('clean_up_tokenization_spaces') is False and
            identity.get('policy_decode_skip_special_tokens') is True and
            identity.get('pad_token_id')==tokenizer.pad_token_id and
            identity.get('eos_token_id')==tokenizer.eos_token_id and
            identity.get('bos_token_id')==tokenizer.bos_token_id,
            'Frozen decoder/tokenizer/special-token identity differs')
    special=set(binding['skip_special_token_ids']);decoder=codecs.getincrementaldecoder('utf-8')('strict')
    raw=bytearray();stable='';decoded=[];complete=[];pending_events=[];prefix_hashes=[]
    for stop,token in enumerate(ids,1):
        piece=b''
        if token not in special:
            spelling=tokenizer.convert_ids_to_tokens(token)
            require(isinstance(spelling,str) and spelling and all(x in BYTELEVEL_INVERSE for x in spelling),
                    'Original token piece is not a provable ByteLevel byte sequence')
            piece=bytes(BYTELEVEL_INVERSE[x] for x in spelling)
        raw.extend(piece)
        try:current=stable+decoder.decode(piece,final=False)
        except UnicodeDecodeError as error:
            raise ValueError('Tree raw-token protocol: Illegal original UTF8 byte sequence') from error
        require('\ufffd' not in current,'Actual UTF8 replacement character unsupported')
        require(current.startswith(stable),'Nonmonotonic completed UTF8 prefix unsupported')
        stable=current;pending=decoder.getstate()[0]
        expected=bytes(raw).decode('utf-8',errors='replace')
        require(expected==stable+pending.decode('utf-8',errors='replace'),
                'Pending UTF8 bytes do not explain native replacement')
        if check_native_prefixes:
            native=tokenizer.decode(ids[:stop],skip_special_tokens=True)
            require(isinstance(native,str) and native==expected,
                    'Native prefix decode differs from proven ByteLevel UTF8/cleanup')
        prefix_hashes.append(fingerprint(expected))
        decoded.append(stable);complete.append(not pending)
        if pending:
            pending_events.append(dict(token_prefix_length=stop,pending_utf8_hex=pending.hex(),
                native_prefix_sha256=fingerprint(expected),completed_prefix_sha256=fingerprint(stable)))
    try:decoder.decode(b'',final=True)
    except UnicodeDecodeError as error:
        raise ValueError('Tree raw-token protocol: Final original UTF8 sequence is incomplete') from error
    require(complete and complete[-1],'Final complete original-token UTF8 boundary required')
    proof=dict(protocol=UTF8_BOUNDARY_PROTOCOL,decoder_binding_sha256=fingerprint(binding),
        original_emitted_byte_sha256=hashlib.sha256(raw).hexdigest(),
        complete_original_token_boundaries=[i+1 for i,valid in enumerate(complete) if valid],
        pending_prefix_events=pending_events,native_prefix_agreement_count=len(ids),
        native_prefix_agreement_sha256=fingerprint(prefix_hashes),
        final_utf8_complete=True,final_native_text_sha256=fingerprint(decoded[-1]))
    return decoded,complete,proof


def tokenizer_identity(tokenizer,expected_sha):
    require(isinstance(expected_sha,str) and re.fullmatch('[0-9a-f]{64}',expected_sha),
            'Explicit frozen tokenizer.json SHA required')
    path=Path(tokenizer.name_or_path)/'tokenizer.json'
    with path.open('rb') as stream:
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    require(digest.hexdigest()==expected_sha,'Tokenizer file identity differs')
    return dict(tokenizer_json_sha256=expected_sha,tokenizer_path=str(path),
        tokenizer_class=type(tokenizer).__module__+'.'+type(tokenizer).__qualname__,
        clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
        policy_decode_skip_special_tokens=True,reward_decode='original_tokenizer.decode_default_flags',
        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,utf8_decoder_binding=_decoder_binding(tokenizer))


def canonical_text(text):
    # Preserve the original SEARCH-first rule, not earliest-tag priority.
    if '</search>' in text:return text.split('</search>')[0]+'</search>'
    if '</answer>' in text:return text.split('</answer>')[0]+'</answer>'
    return text


def capture_native_sampling_settings(sampling_params):
    """Read original native attributes; usable before engine processor construction."""
    fields=('n','best_of','_real_n','temperature','top_p','top_k','min_p','presence_penalty','frequency_penalty',
        'repetition_penalty','min_tokens','max_tokens','ignore_eos','stop','stop_token_ids',
        'include_stop_str_in_output','logprobs','prompt_logprobs','seed','logits_processors',
        'truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids')
    for key in fields:
        require(hasattr(sampling_params,key), 'Explicit actual '+key+' metadata is required')
    settings={key:copy.deepcopy(getattr(sampling_params,key)) for key in fields}
    for key in ('best_of','_real_n'):
        require(type(settings[key]) is int and settings[key]==1,
                'Native raw-token protocol requires actual '+key+'=1')
    for key in ('truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids'):
        require(settings[key] is None, 'Native raw-token protocol requires actual '+key+'=None')
    require(settings['logits_processors'] is None, 'Native raw-token protocol requires actual logits_processors=None')
    require(settings['n']==1 and settings['max_tokens']==500 and settings['ignore_eos'] is False and
            settings['min_tokens']==0 and not settings['stop'], 'Unsupported native stop/distribution protocol')
    return settings


def capture_native_emissions(request_outputs,sampling_params,call_id,rank):
    """Called before facade padding, retaining real request IDs and scalar values."""
    settings=capture_native_sampling_settings(sampling_params)
    provenance={key:os.environ.get(key) for key in ('SEARCH_R1_BASELINE_COMMIT','SEARCH_R1_BASELINE_CONFIG_SHA256',
        'SEARCH_R1_BASELINE_DATA_IDENTITY','SEARCH_R1_MODEL_IDENTITY','SEARCH_R1_BASELINE_RUN_ID','SLURM_JOB_ID')}
    require(all(isinstance(value,str) and value for value in provenance.values()),
            'Own source/config/data/model/run/job provenance is required for opt-in emission capture')
    rows=[];seen=set()
    for row,request in enumerate(request_outputs):
        require(request.request_id not in seen and request.finished and len(request.outputs)==1,
                'Unique final n1 native requests required')
        seen.add(request.request_id);out=request.outputs[0]
        ids=[int(token) for token in out.token_ids]
        require(0<len(ids)<=500 and out.logprobs is not None and len(out.logprobs)==len(ids),
                'Empty/missing/over-cap native emission or scalar metadata')
        logps=[float(values[token].logprob) for values,token in zip(out.logprobs,ids)]
        require(all(math.isfinite(value) for value in logps),'Nonfinite sampled scalar')
        require(out.finish_reason in ('stop','length'), 'Unsupported native abort/finish status')
        identity=dict(call_id=call_id,rank=int(rank),request_id=request.request_id,physical_row=row,
                      runtime_provenance=provenance.copy())
        rows.append(dict(schema_version=1,protocol=PROTOCOL,sampling_identity=identity,
            sample_id=fingerprint(identity),prompt_token_ids=[int(x) for x in request.prompt_token_ids],
            raw_emitted_token_ids=ids,raw_selected_logprobs=logps,true_emitted_length=len(ids),
            finish_reason=out.finish_reason,stop_reason=out.stop_reason,finished=True,
            sampling_params=copy.deepcopy(settings)))
    return rows


def sampled_action(tokenizer,emission,raw_padded,legacy_ids,legacy_text,identity):
    """Scan complete original prefixes; never encode a policy token or split one."""
    started=time.perf_counter(),time.process_time()
    require(emission.get('protocol')==PROTOCOL and emission.get('schema_version')==1,
            'Native emission metadata missing')
    pad,eos=tokenizer.pad_token_id,tokenizer.eos_token_id
    require(type(pad) is int and type(eos) is int and pad!=eos,'PAD==EOS unsupported')
    ids=emission['raw_emitted_token_ids'];n=emission['true_emitted_length'];values=emission['raw_selected_logprobs']
    require(type(n) is int and 0<n<=500 and len(ids)==len(values)==n and
            all(type(x) is int for x in ids) and all(math.isfinite(x) for x in values),
            'Actual emitted length/IDs/scalars invalid')
    require(pad not in ids,'Genuinely emitted PAD requires a separately reviewed explicit-mask protocol')
    require(len(raw_padded)==500 and raw_padded[:n]==ids and all(x==pad for x in raw_padded[n:]),
            'Raw returned row changed or invalid padded suffix')
    require(emission['sample_id']==fingerprint(emission['sampling_identity']), 'Sampling identity mutated')
    utf8_started=time.process_time()
    decoded,complete,utf8_proof=_scan_utf8_prefixes(tokenizer,ids,identity)
    utf8_cpu_seconds=time.process_time()-utf8_started
    full=decoded[-1];text=canonical_text(full)
    require(text==legacy_text and tokenizer.encode(text,add_special_tokens=False)==legacy_ids,
            'Legacy parser text/canonical semantic IDs differ')
    tagged='</search>' in full or '</answer>' in full
    if tagged:
        choices=[i+1 for i,prefix in enumerate(decoded) if complete[i] and prefix.startswith(text)]
        require(choices,'Canonical boundary not covered by a complete original token')
        cut=choices[0]
    else:cut=n
    action=ids[:cut];projection=decoded[cut-1]
    require(projection.startswith(text),'Whole-token boundary does not cover canonical semantics')
    extra=projection[len(text):]
    ledger=dict(protocol=PROTOCOL,tokenizer_identity=copy.deepcopy(identity),utf8_boundary_proof=utf8_proof,
        native_emission=copy.deepcopy(emission),cut=cut,action_token_ids=action,
        selected_logprobs=values[:cut],canonical_text=text,canonical_text_sha256=fingerprint(text),
        canonical_semantic_token_ids=legacy_ids.copy(),legacy_reward_action_text=tokenizer.decode(legacy_ids),
        whole_prefix_decode=projection,whole_prefix_decode_sha256=fingerprint(projection),
        indivisible_extra_text=extra,indivisible_extra_utf8_hex=extra.encode('utf8').hex(),
        boundary_reason='shortest_whole_original_prefix_covering_legacy_tag' if tagged else 'all_true_emitted_tokens_no_tag',
        raw_emitted_tokens=n,retained_sampled_tokens=cut,discarded_sampled_tail_tokens=n-cut,
        emitted_eos_positions=[i for i,x in enumerate(ids) if x==eos],
        eos_retained=any(x==eos for x in action),empty_canonical=text=='',
        prefix_scan_wall_seconds=time.perf_counter()-started[0],
        prefix_scan_cpu_seconds=time.process_time()-started[1],
        prefix_scan_utf8_cpu_seconds=utf8_cpu_seconds,
        probability_accepted=False)
    ledger['action_binding_sha256']=fingerprint({k:v for k,v in ledger.items() if not k.startswith('prefix_scan_')})
    return action,ledger


def validate_action_utf8_proof(action,tokenizer):
    """Reconstruct byte proof and check actual full/cut decode, without resampling."""
    emission=action['native_emission'];ids=emission['raw_emitted_token_ids'];cut=action['cut']
    decoded,complete,proof=_scan_utf8_prefixes(tokenizer,ids,action['tokenizer_identity'],False)
    require(proof==action.get('utf8_boundary_proof'),'Bound original-token UTF8 proof differs')
    require(type(cut) is int and 0<cut<=len(ids) and complete[cut-1],
            'Retained boundary has incomplete UTF8 bytes')
    full=tokenizer.decode(ids,skip_special_tokens=True)
    projection=tokenizer.decode(ids[:cut],skip_special_tokens=True)
    text=canonical_text(full)
    tagged='</search>' in full or '</answer>' in full
    choices=[i+1 for i,prefix in enumerate(decoded) if complete[i] and prefix.startswith(text)]
    require(not tagged or choices,'Canonical tag has no proven complete original-token boundary')
    require(full==decoded[-1] and projection==decoded[cut-1] and action['canonical_text']==text and
            action['whole_prefix_decode']==projection and cut==(choices[0] if tagged else len(ids)) and
            action['action_token_ids']==ids[:cut] and action['selected_logprobs']==emission['raw_selected_logprobs'][:cut] and
            action['indivisible_extra_text']==projection[len(text):] and
            action['indivisible_extra_utf8_hex']==projection[len(text):].encode('utf-8').hex(),
            'Native/canonical complete-token UTF8 boundary differs')


def trajectory_projection(state,cap):
    raw=[];mask=[];semantic=[];semantic_mask=[];contexts=[]
    for turn,ledger in enumerate(state.raw_ledger):
        require(ledger['turn']==turn and ledger['generation_context_token_ids']==state.prompt+raw,
                'Inherited generation context differs from complete scoring prefix')
        require(ledger['generation_context_positions']==list(range(len(state.prompt+raw))) and
                ledger['generation_context_attention']==[1]*len(state.prompt+raw), 'Generation position/mask history differs')
        action=ledger['action'];obs=ledger['observation_token_ids'];n=action['cut']
        require(action['action_token_ids']==action['native_emission']['raw_emitted_token_ids'][:n] and
                action['selected_logprobs']==action['native_emission']['raw_selected_logprobs'][:n],
                'Inherited sampled token/scalar prefix changed')
        bound={k:v for k,v in action.items() if not k.startswith('prefix_scan_') and k!='action_binding_sha256'}
        require(fingerprint(bound)==action['action_binding_sha256'],'Inherited action ledger changed')
        raw+=action['action_token_ids']+obs;mask+=[1]*n+[0]*len(obs)
        legacy=action['canonical_semantic_token_ids'];semantic+=legacy+obs
        semantic_mask+=[1]*len(legacy)+[0]*len(obs);contexts.append(ledger['generation_context_token_ids'])
    require(raw==state.response and mask==state.model_mask and len(raw)<=cap and len(semantic)<=cap,
            'Policy/semantic score cap or retained timeline mismatch')
    value=dict(protocol=PROTOCOL,prompt_token_ids=state.prompt.copy(),policy_response_token_ids=raw,
        policy_model_mask=mask,semantic_response_token_ids=semantic,semantic_model_mask=semantic_mask,
        raw_ledger=copy.deepcopy(state.raw_ledger),response_cap=cap)
    value['binding_sha256']=fingerprint(value)
    return value


def semantic_spans(item,tokenizer):
    """Rebuild legacy contiguous spans; canonical IDs never enter policy tensors."""
    if FIELD not in item.non_tensor_batch:
        require(getattr(item,'meta_info',{}).get('tree_raw_token_protocol') is None,
                'Opted-in Tree semantic projection was dropped')
        return None
    value=item.non_tensor_batch[FIELD]
    bound={k:v for k,v in value.items() if k!='binding_sha256'}
    require(fingerprint(bound)==value['binding_sha256'] and value['protocol']==PROTOCOL,
            'Semantic projection binding changed')
    require(value['response_cap']==4096,'Frozen score cap changed')
    width=item.batch['responses'].shape[-1]
    valid=item.batch['attention_mask'][-width:].bool();ids=item.batch['responses'][valid].tolist()
    mask=item.batch['info_mask'][-width:][valid].tolist()
    prompt=item.batch['prompts'];prompt=prompt[prompt!=tokenizer.pad_token_id].tolist()
    require(ids==value['policy_response_token_ids'] and mask==value['policy_model_mask'] and
            prompt==value['prompt_token_ids'],'Semantic projection attached to different policy inputs/mask')
    state=type('ProjectionState',(),{})()
    state.prompt=prompt;state.response=ids;state.model_mask=mask;state.raw_ledger=value['raw_ledger']
    require(trajectory_projection(state,value['response_cap'])==value, 'Semantic ledger does not reconstruct bound timeline')
    for ledger in value['raw_ledger']:
        action=ledger['action'];identity=action['tokenizer_identity']
        require(identity['pad_token_id']==tokenizer.pad_token_id and identity['eos_token_id']==tokenizer.eos_token_id and
                identity['clean_up_tokenization_spaces']==tokenizer.clean_up_tokenization_spaces and
                tokenizer.decode(action['canonical_semantic_token_ids'])==action['legacy_reward_action_text'],
                'Frozen tokenizer flags or legacy reward projection changed')
        validate_action_utf8_proof(action,tokenizer)
    spans=[];start=None;tokens=value['semantic_response_token_ids'];marks=value['semantic_model_mask']
    for index,is_model in enumerate(marks):
        if is_model and start is None:start=index
        elif not is_model and start is not None:
            spans.append(tokenizer.decode(tokens[start:index]));start=None
    if start is not None:spans.append(tokenizer.decode(tokens[start:]))
    return spans
