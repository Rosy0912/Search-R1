"""Exact selected-token journals; CPU evidence does not admit native probabilities."""
import hashlib
import json
import math


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError('Fresh200 sampling rejected: ' + reason)


def capture_requests(outputs, requests, row_ids, snapshot, call_id, ordinal, sampling, eos, pad):
    require(len(outputs) == len(requests) == len(row_ids), 'native request count')
    records = []
    for i, request in enumerate(outputs):
        require(list(request.prompt_token_ids) == requests[i], 'native request context')
        require(len(request.outputs) == 1, 'native completion count')
        completion = request.outputs[0]
        tokens = list(completion.token_ids)
        require(tokens and all(type(t) is int and t >= 0 for t in tokens), 'empty or invalid raw completion')
        require(completion.logprobs is not None and len(completion.logprobs) == len(tokens), 'missing selected probabilities')
        probabilities = []
        for token, values in zip(tokens, completion.logprobs):
            require(token in values, 'missing selected token probability')
            value = values[token]
            value = float(value.logprob if hasattr(value, 'logprob') else value)
            require(math.isfinite(value) and value <= 0, 'invalid selected probability')
            probabilities.append(value)
        record = dict(row_id=str(row_ids[i]), policy_snapshot_id=snapshot, call_id=call_id,
            turn_index=ordinal, request_id=str(request.request_id), actual_request_token_ids=requests[i],
            raw_generated_token_ids=tokens, selected_logprobs=probabilities,
            eos_token_id=eos, pad_token_id=pad, finish_reason=completion.finish_reason,
            stop_reason=getattr(completion, 'stop_reason', None), sampling=sampling)
        record['native_sha256'] = sha(record)
        records.append(record)
    return records


def prefix(tokens, decode):
    """Retain complete raw tokens through a closed action; never retokenize."""
    require(bool(tokens), 'empty raw output')
    text = decode(tokens)
    endings = [text.find(tag)+len(tag) for tag in ('</search>', '</answer>') if tag in text]
    if not endings:
        return len(tokens), text
    end = min(endings)
    for count in range(1,len(tokens)+1):
        retained = decode(tokens[:count])
        if len(retained) >= end and retained[:end] == text[:end]:
            require(not retained[end:].strip(), 'closed-tag token straddles further text')
            return count,text[:end]
    raise ValueError('Fresh200 sampling rejected: unstable prefix projection')


def audit_row(prompt, response, model_mask, selected, turns, row_id, snapshot):
    require(bool(prompt) and bool(turns), 'empty context or journal')
    rebuilt,mask,probabilities = [],[],[]
    for ordinal,turn in enumerate(turns):
        native = {k:v for k,v in turn.items() if k not in ('native_sha256','kept_token_count','environment_token_ids','turn_sha256')}
        require(sha(native)==turn['native_sha256'], 'native SHA')
        if 'effective_stop_provenance' in turn:
            from .sampling_capture import validate_stop_proof, KEYS
            if 'actual_loaded_generation_config_fields' in turn['sampling']:
                require(turn['sampling']['actual_loaded_generation_config_fields']==turn['effective_stop_provenance']['actual_loaded_generation_config_fields'], 'actual generation configuration drift')
            validate_stop_proof(turn['effective_stop_provenance'],turn['request_id'],
                turn['actual_request_token_ids'],{k:turn['sampling'].get(k) for k in KEYS},
                turn['eos_token_id'],turn['pad_token_id'])
        require(turn['row_id']==row_id and turn['policy_snapshot_id']==snapshot and turn['turn_index']==ordinal, 'row/snapshot/turn identity')
        require(turn['actual_request_token_ids']==prompt+rebuilt, 'sampled context differs from actor context')
        n=turn['kept_token_count']
        require(type(n) is int and 0<n<=len(turn['raw_generated_token_ids']), 'retained count')
        require(len(turn['selected_logprobs'])==len(turn['raw_generated_token_ids']), 'probability count')
        env=turn['environment_token_ids']
        rebuilt.extend(turn['raw_generated_token_ids'][:n]+env)
        mask.extend([1]*n+[0]*len(env))
        probabilities.extend(turn['selected_logprobs'][:n]+[0.]*len(env))
        require(sha({k:v for k,v in turn.items() if k!='turn_sha256'})==turn['turn_sha256'], 'turn SHA')
    require(rebuilt==response and mask==model_mask, 'actor token/mask reconstruction')
    require(len(selected)==len(probabilities) and all(math.isfinite(a) and abs(a-b)<=1e-6*max(1.,abs(b)) for a,b in zip(selected,probabilities)), 'selected probability transport')
    return dict(row_id=row_id,context_sha256=sha(dict(prompt=prompt,response=response,model_mask=model_mask)),turns=len(turns))
