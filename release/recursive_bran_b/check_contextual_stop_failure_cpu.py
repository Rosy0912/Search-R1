"""Replay the retained failed sample on CPU; raw data never leave the node."""
import argparse,copy,json,os
from pathlib import Path

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser();p.add_argument('--failure',type=Path,required=True)
    p.add_argument('--model',required=True);a=p.parse_args()
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from search_r1.baselines import tree_raw_tokens as raw
    f=json.loads(a.failure.read_text());tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    c=f['failure_context'];e=copy.deepcopy(c['native_emissions'][194])
    # This historical record predates explicit decoder flag capture. Bind the
    # installed native defaults for this diagnostic only, never in production.
    defaults=SamplingParams()
    for key in ('detokenize','skip_special_tokens','spaces_between_special_tokens'):
        assert key not in e['sampling_params']
        e['sampling_params'][key]=getattr(defaults,key)
    ids=e['raw_emitted_token_ids'];text=c['legacy_text'][194]
    action,ledger=raw.sampled_action(tok,e,ids+[tok.pad_token_id]*(500-len(ids)),
        tok.encode(text,add_special_tokens=False),text,f['tokenizer_identity'])
    raw.validate_action_utf8_proof(ledger,tok)
    assert action==ids and ledger['selected_logprobs']==e['raw_selected_logprobs']
    assert ledger['canonical_text']==text and ledger['contextual_native_stop_proof']['stop_token_boundary']==141
    decoded=[tok.decode(ids[:n],skip_special_tokens=True) for n in range(1,len(ids)+1)]
    for edit in ('forged_text','extra_token'):
        bad=copy.deepcopy(e)
        if edit=='forged_text':bad['native_returned_text']='forged'+bad['native_returned_text']
        else:
            bad['raw_emitted_token_ids']=ids+[ids[-1]]
        try:raw._replay_contextual_string_stop(tok,bad)
        except ValueError:pass
        else:raise AssertionError(edit+' was accepted')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASSED',actual_retained_tokens=len(action),
        actual_logprobs_unchanged=True,legacy_text_unchanged=True,
        forged_text_rejected=True,post_stop_token_rejected=True,cuda_initialized=False,
        contextual_stop_proof=ledger['contextual_native_stop_proof'])))

if __name__=='__main__':main()
