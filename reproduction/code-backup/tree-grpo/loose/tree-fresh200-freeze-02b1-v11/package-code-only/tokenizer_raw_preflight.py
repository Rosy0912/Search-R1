"""DRAFT: h100-9 CPU-only official-tokenizer check of bound historical JSON.

No model/PT load, no generation, no sample filtering, no probability acceptance.
Historical RequestOutput JSON is adapted for the existing parser helper only;
this does not fabricate a new native-emission or fresh training provenance.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import struct
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
MODEL = ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
TOKENIZER_SHA = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
FIELDS = ('n','best_of','_real_n','temperature','top_p','top_k','min_p','presence_penalty','frequency_penalty',
          'repetition_penalty','min_tokens','max_tokens','ignore_eos','stop','stop_token_ids',
          'include_stop_str_in_output','logprobs','prompt_logprobs','seed','logits_processors',
          'truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids')


def require(condition, message):
    if not condition: raise ValueError(message)


def bound_json(entry):
    path = Path(entry['path'])
    require(path.resolve().is_relative_to(ROOT), 'JSON source must remain in this task on h100-9')
    with path.open('rb') as stream: raw = stream.read()
    require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'Historical JSON source SHA changed')
    return json.loads(raw)


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def inspect_row(helper, tokenizer, identity, ready_entry, record, params):
    """Only call5 action-local and no-tool bounds; not a Tree capacity proof."""
    require(type(params) is dict, 'Actual complete captured parameters object required')
    # Historical capture may lack a field now required by fresh native capture.
    # Preserve the gap without inventing None/default and continue only the
    # independently checkable token-boundary analysis.
    missing = [k for k in FIELDS if k not in params]
    settings = {k: params[k] for k in FIELDS if k in params}
    expected = dict(n=1,best_of=1,_real_n=1,temperature=1.0,top_p=1.0,top_k=-1,min_p=0.0,presence_penalty=0.0,
                    frequency_penalty=0.0,repetition_penalty=1.0,min_tokens=0,max_tokens=500,
                    ignore_eos=False,logprobs=1,prompt_logprobs=None,seed=None,logits_processors=None,
                    truncate_prompt_tokens=None,guided_decoding=None,logit_bias=None,allowed_token_ids=None)
    supported = (not missing and all(settings[k] == v for k,v in expected.items())
                 and type(settings['best_of']) is int
                 and type(settings['_real_n']) is int
                 and settings['stop'] in (None,[]) and settings['stop_token_ids'] in (None,[]))
    ids = record['token_ids']; prompt = record['prompt_token_ids']
    require(record['finished'] is True and record['completion_index'] == 0
            and record['true_response_length'] == len(ids) == len(record['logprobs']),
            'Historical true request length/finality differs')
    values = [entry[str(token)]['logprob'] for entry,token in zip(record['logprobs'],ids)]
    require(all(type(value) in (int,float) and math.isfinite(value)
                and struct.unpack('<f',struct.pack('<f',value))[0] == value for value in values),
            'Original sampler scalar is missing/nonfinite/not exactFP32')
    derived = dict(scope='HISTORICAL_JSON_ADAPTATION_FOR_CPU_PARSER_ONLY',
                   ready_sha256=ready_entry['sha256'],request_id=record['request_id'],physical_row=record['input_row'])
    emission = dict(schema_version=1,protocol=helper.PROTOCOL,sampling_identity=derived,
        sample_id=helper.fingerprint(derived),prompt_token_ids=prompt,
        raw_emitted_token_ids=ids,raw_selected_logprobs=values,true_emitted_length=len(ids),
        finish_reason=record['finish_reason'],stop_reason=record['stop_reason'],finished=True,
        sampling_params=settings)
    # Exactly the old manager decode -> SEARCH-first truncation -> encode semantics.
    full = tokenizer.decode(ids,skip_special_tokens=True)
    canonical = helper.canonical_text(full)
    legacy = tokenizer.encode(canonical,add_special_tokens=False)
    padded = ids + [tokenizer.pad_token_id]*(500-len(ids))
    action, ledger = helper.sampled_action(tokenizer,emission,padded,legacy,canonical,identity)
    return dict(retained_tokens=len(action),emitted_tokens=len(ids),prompt_tokens=len(prompt),
        raw_finish_reason=record['finish_reason'],action_binding_sha256=ledger['action_binding_sha256'],
        actual_native24_complete=not missing,missing_actual_sampling_fields=missing,
        actual_best_of_status='UNKNOWN_NOT_RECORDED' if 'best_of' in missing else 'RECORDED',
        actual_control_fields_status={key:'UNKNOWN_NOT_RECORDED' if key in missing else 'RECORDED'
            for key in ('_real_n','truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids')},
        actual_sampling_support_validated=supported,
        prompt_plus_action_exceeds4096=len(prompt)+len(action)>4096,
        prompt_exceeds2048=len(prompt)>2048,
        context_sha256=json_sha(prompt),actual_sampling_params_sha256=json_sha(settings))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--output',required=True); args=parser.parse_args()
    require(socket.gethostname().split('.')[0]=='h100-9' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'CPU draft only runs on h100-9 with CUDA_VISIBLE_DEVICES explicitly empty')
    plan=bound_json(dict(path=args.plan,sha256=args.plan_sha)); output=Path(args.output)
    require(output.resolve().is_relative_to(ROOT) and not output.exists(), 'Own no-clobber metadata output required')
    entry=plan['raw_helper']; source=Path(entry['path'])
    require(source.resolve().is_relative_to(ROOT) and hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256'],
            'New raw-token helper source binding changed')
    spec=importlib.util.spec_from_file_location('frozen_raw_token_cpu_preflight',source)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from transformers import AutoTokenizer
    import torch
    require(not torch.cuda.is_initialized(), 'CPU tokenizer import initialized CUDA')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    identity=helper.tokenizer_identity(tokenizer,TOKENIZER_SHA)
    require(tokenizer.pad_token_id==151643 and tokenizer.eos_token_id==151645, 'Official PAD/EOS changed')
    records=[]; starts=time.perf_counter(); count=0
    for group in plan['runs']:
        require(group['job_id'] in ('3295','3296') and len(group['rank_ready'])==8, 'Only frozen historical3295/3296 eight-rank JSON supported')
        for rank,ready_entry in enumerate(group['rank_ready']):
            ready=bound_json(ready_entry)
            require(ready['rank']==rank and ready['world_size']==8 and ready['optimizer_committed']==0
                    and ready['original_generation_call']==5 and ready['original_prompt_rows']==320
                    and ready['observer']['observer_evidence_complete'] is True, 'Historical window receipt incomplete')
            params=ready['sampling_params_before']; after=ready['sampling_params_after']
            present=[k for k in FIELDS if k in params or k in after]
            require(all(k in params and k in after and params[k]==after[k] for k in present),
                    'Captured native sampling field changed during original window')
            generated=sorted(ready['observer']['generated'],key=lambda item:item['input_row'])
            require(len(generated)==320 and [r['input_row'] for r in generated]==list(range(320)),
                    'Original physical request rows incomplete')
            for record in generated:
                count+=1
                result=dict(job_id=group['job_id'],rank=rank,row=record['input_row'],
                            request_id=record['request_id'],ready_sha256=ready_entry['sha256'])
                try:
                    result.update(status='HISTORICAL_ACTION_LOCAL_PARSER_ACCEPTED',
                                  **inspect_row(helper,tokenizer,identity,ready_entry,record,params))
                except Exception as exc:
                    result.update(status='HISTORICAL_ACTION_LOCAL_PARSER_REJECTED',reason=repr(exc),
                                  context_sha256=json_sha(record['prompt_token_ids']),
                                  raw_ids_sha256=json_sha(record['token_ids']))
                records.append(result)
    require(not torch.cuda.is_initialized(), 'Historical CPU preflight initialized CUDA')
    failures=[r for r in records if r['status'].endswith('REJECTED')]
    result=dict(status='HISTORICAL_RAW_CPU_PREFLIGHT_OBSERVED_NOT_FRESH_TRAINING_ACCEPTED',
        tokenizer_identity=identity,plan_sha256=args.plan_sha,rows=count,rejected_rows=len(failures),
        rows_with_sampling_metadata_gap=sum(bool(r.get('missing_actual_sampling_fields')) for r in records),
        raw_helper_sha256=entry['sha256'],records=records,wall_seconds=time.perf_counter()-starts,
        model_weight_loads=0,tensor_asset_loads=0,generation_calls=0,optimizer_writes=0,cuda_initialized=False,
        original_sampler_actor_gate_accepted=False,fresh_tree_rollout_accepted=False,
        limitations='Historical call5 only. Tool observations and complete fresh Tree timelines unavailable; accepted action-local rows do not prove score-cap/rolling compatibility. No filtering or production update follows this report.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','tokenizer_identity')},allow_nan=False))


if __name__=='__main__': main()
