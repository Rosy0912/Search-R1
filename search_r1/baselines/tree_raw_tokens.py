"""Tree-only sampled-token prefixes plus separately bound legacy semantics.

No token/probability is repaired, no optimizer is called and no probability
tolerance is defined here. Unsupported boundaries reject the current batch.
"""
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


def require(value,message):
    if not value:raise ValueError('Tree raw-token protocol: '+message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


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
        bos_token_id=tokenizer.bos_token_id)


def canonical_text(text):
    # Preserve the original SEARCH-first rule, not earliest-tag priority.
    if '</search>' in text:return text.split('</search>')[0]+'</search>'
    if '</answer>' in text:return text.split('</answer>')[0]+'</answer>'
    return text


def capture_native_emissions(request_outputs,sampling_params,call_id,rank):
    """Called before facade padding, retaining real request IDs and scalar values."""
    fields=('n','temperature','top_p','top_k','min_p','presence_penalty','frequency_penalty',
        'repetition_penalty','min_tokens','max_tokens','ignore_eos','stop','stop_token_ids',
        'include_stop_str_in_output','logprobs','prompt_logprobs','seed')
    settings={key:copy.deepcopy(getattr(sampling_params,key)) for key in fields}
    require(settings['n']==1 and settings['max_tokens']==500 and settings['ignore_eos'] is False and
            settings['min_tokens']==0 and not settings['stop'], 'Unsupported native stop/distribution protocol')
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
    decoded=[];previous=''
    for stop in range(1,n+1):
        current=tokenizer.decode(ids[:stop],skip_special_tokens=True)
        require(isinstance(current,str) and '\ufffd' not in current,'UTF8 replacement/ambiguous prefix decode')
        current.encode('utf-8',errors='strict')
        require(current.startswith(previous),'Nonmonotonic prefix decode/cleanup unsupported')
        decoded.append(current);previous=current
    full=decoded[-1];text=canonical_text(full)
    require(text==legacy_text and tokenizer.encode(text,add_special_tokens=False)==legacy_ids,
            'Legacy parser text/canonical semantic IDs differ')
    tagged='</search>' in full or '</answer>' in full
    if tagged:
        choices=[i+1 for i,prefix in enumerate(decoded) if prefix.startswith(text)]
        require(choices,'Canonical boundary not covered by a complete original token')
        cut=choices[0]
    else:cut=n
    action=ids[:cut];projection=decoded[cut-1]
    require(projection.startswith(text),'Whole-token boundary does not cover canonical semantics')
    extra=projection[len(text):]
    ledger=dict(protocol=PROTOCOL,tokenizer_identity=copy.deepcopy(identity),
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
        probability_accepted=False)
    ledger['action_binding_sha256']=fingerprint({k:v for k,v in ledger.items() if not k.startswith('prefix_scan_')})
    return action,ledger


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
    spans=[];start=None;tokens=value['semantic_response_token_ids'];marks=value['semantic_model_mask']
    for index,is_model in enumerate(marks):
        if is_model and start is None:start=index
        elif not is_model and start is not None:
            spans.append(tokenizer.decode(tokens[start:index]));start=None
    if start is not None:spans.append(tokenizer.decode(tokens[start:]))
    return spans
