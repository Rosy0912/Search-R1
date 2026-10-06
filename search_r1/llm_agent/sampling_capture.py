"""Read actual enqueued request groups without changing their sampler policy."""
from contextlib import contextmanager
import copy
from .sampling_trace import require, sha

KEYS=('n','best_of','temperature','top_p','top_k','min_p','presence_penalty',
      'frequency_penalty','repetition_penalty','ignore_eos','min_tokens','max_tokens',
      'stop','stop_token_ids','logprobs','truncate_prompt_tokens','logits_processors',
      'logit_bias','allowed_token_ids','guided_decoding')


def sampler_values(params):
    return {key:copy.deepcopy(getattr(params,key,None)) for key in KEYS}


def neutral_sampler(values, evaluation=False):
    expected=dict(n=1,temperature=0. if evaluation else 1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,
                  frequency_penalty=0.,repetition_penalty=1.,ignore_eos=False,min_tokens=0)
    require(all(values.get(k)==v for k,v in expected.items()),'sampling transformation changes old policy')
    require(values.get('best_of') in (None,1) and values.get('truncate_prompt_tokens') in (None,-1), 'best_of or prompt truncation')
    require(not any(values.get(k) for k in ('stop','logits_processors','logit_bias','allowed_token_ids','guided_decoding')),'custom sampler transform or textual stop')
    require(type(values.get('max_tokens')) is int and values['max_tokens']>0 and values.get('logprobs') is not None,'missing max tokens/selected probabilities')


def stop_ids(value):
    if value is None:return []
    values=[value] if type(value) is int else list(value)
    require(all(type(v) is int and v>=0 for v in values),'invalid effective stop IDs')
    return sorted(set(values))


def validate_stop_proof(proof,request_id,request,caller,eos,pad):
    require(proof['request_id']==request_id and proof['actual_request_sha256']==sha(request), 'effective group context identity')
    require(proof['tokenizer_eos_token_id']==eos and proof['tokenizer_pad_token_id']==pad,'special token identity')
    require(proof['caller_sampling_sha256']==sha(caller),'effective caller identity')
    effective=proof['effective_sampling'];neutral_sampler(effective, proof.get('probability_role')=='greedy_model_logprob_not_sampling_policy')
    require({k:v for k,v in effective.items() if k!='stop_token_ids'}=={k:v for k,v in caller.items() if k!='stop_token_ids'}, 'engine sampler transformation')
    seq_eos=proof['sequence_eos_token_id'];generation=stop_ids(proof['actual_loaded_generation_config_fields'].get('eos_token_id'))
    all_expected=set(stop_ids(caller['stop_token_ids']))|set(generation)|set(stop_ids(seq_eos))
    expected=set(stop_ids(caller['stop_token_ids']))|{v for v in generation if v!=seq_eos}
    require(stop_ids(effective['stop_token_ids'])==sorted(expected) and proof['effective_all_stop_token_ids']==sorted(all_expected),'effective stop set differs')
    require(sha({k:v for k,v in proof.items() if k!='sha256'})==proof['sha256'],'effective group SHA')


@contextmanager
def capture_groups(engine,caller,requests,tokenizer=None,evaluation=False):
    """Capture immediately after native add_request, before scheduler execution.

    Fail closed if the installed engine no longer exposes enqueued groups.
    """
    require(hasattr(engine,'add_request') and hasattr(engine,'scheduler'),'unsupported native group capture API')
    if tokenizer is None:
        # CPU fixtures and supported engine APIs may expose the real tokenizer.
        if hasattr(engine, 'get_tokenizer'):
            tokenizer=engine.get_tokenizer()
        elif hasattr(getattr(engine, 'tokenizer', None), 'get_lora_tokenizer'):
            tokenizer=engine.tokenizer.get_lora_tokenizer(None)
        else:
            tokenizer=getattr(engine, 'tokenizer', None)
    require(tokenizer is not None and hasattr(tokenizer,'eos_token_id') and hasattr(tokenizer,'pad_token_id'),
            'actual tokenizer special identity unavailable')
    original=engine.add_request
    had_method='add_request' in vars(engine);records={}
    caller_values=sampler_values(caller);neutral_sampler(caller_values,evaluation)
    generation=copy.deepcopy(getattr(engine,'generation_config_fields',{}))
    def add_request(*args,**kwargs):
        require(sampler_values(caller)==caller_values, "caller sampler mutated before native request")
        require(getattr(engine, "generation_config_fields", {})==generation, "loaded generation configuration drift")
        result=original(*args,**kwargs)
        require(sampler_values(caller)==caller_values, "engine mutated actual caller sampler")
        request_id=kwargs.get('request_id',args[0] if args else None)
        require(isinstance(request_id,str) and request_id not in records,'native group request ID')
        schedulers=engine.scheduler if isinstance(engine.scheduler,(list,tuple)) else [engine.scheduler]
        groups=[g for scheduler in schedulers for g in getattr(scheduler,'waiting',[]) if g.request_id==request_id]
        require(len(groups)==1,'missing/duplicate actual enqueued group')
        group=groups[0];seqs=group.get_seqs()
        require(len(seqs)==1,'actual sequence group count')
        index=len(records);request=list(group.prompt_token_ids)
        require(index<len(requests) and request==requests[index],'enqueued request differs from caller')
        values=sampler_values(group.sampling_params)
        proof=dict(probability_role="greedy_model_logprob_not_sampling_policy" if evaluation else "stochastic_sampling_policy",request_id=request_id,sequence_id=int(seqs[0].seq_id),actual_request_sha256=sha(request),
            caller_sampling_sha256=sha(caller_values),effective_sampling=values,
            effective_all_stop_token_ids=stop_ids(group.sampling_params.all_stop_token_ids),
            sequence_eos_token_id=seqs[0].eos_token_id,tokenizer_eos_token_id=tokenizer.eos_token_id,
            tokenizer_pad_token_id=tokenizer.pad_token_id,actual_loaded_generation_config_fields=generation)
        proof['sha256']=sha(proof)
        require(seqs[0].eos_token_id == tokenizer.eos_token_id, 'actual sequence/tokenizer EOS differs')
        validate_stop_proof(proof,request_id,request,caller_values,tokenizer.eos_token_id,tokenizer.pad_token_id)
        records[request_id]=proof
        return result
    engine.add_request=add_request
    try:yield records,caller_values
    finally:
        if had_method:engine.add_request=original
        else:del engine.add_request
