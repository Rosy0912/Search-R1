"""CPU-only comparison of recorded R/PR initial-outer generation and HTTP traces.

No Torch/tokenizer/model/HTTP/Node writes. Emits compact counts and private paths.
"""
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

R_CONFIG = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-R/receipts/authorized-launch-20261009/config.json')
PR_CONFIG = Path('/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/receipts/training-config.json')
EXPECTED_CONFIG = {'R':'8cc81ae28882f0fb32a924d7a4d96fc6f7f0702b210ef62847e409c404d36ab9',
                   'PR':'651dfb40b1bf22fb5f604b9b17c45ffe8619f20baf521f399bc8059b11d6aa91'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canon(x):
    return sha(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())


def read(path, proofs, expected=None, jsonl=False):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), str(path)
    digest = sha(raw)
    if expected is not None:
        assert digest == expected, 'SHA mismatch: ' + str(path)
    if jsonl:
        assert not raw or raw.endswith(b'\n'), 'Partial JSONL: ' + str(path)
    proofs.append(dict(path=str(path), bytes=len(raw), sha256=digest, metadata_stable=True))
    return [json.loads(line) for line in raw.splitlines()] if jsonl else json.loads(raw)


def key(row):
    return (row['question_id'], row['sibling'])


def query(segment):
    if '</answer>' in segment:
        return None
    match = re.search(r'<search>(.*?)</search>', segment, re.S)
    return match.group(1).strip() if match else None


def first_difference(a, b):
    for i, (x,y) in enumerate(zip(a,b)):
        if x != y:
            return i
    return min(len(a),len(b)) if len(a)!=len(b) else None


def region(text, position):
    if position is None:
        return 'no_native_text_difference'
    for tag in ['search','answer']:
        for m in re.finditer(r'<'+tag+r'>(.*?)</'+tag+r'>', text, re.S):
            if m.start(1)<=position<m.end(1):
                return 'search_query' if tag=='search' else 'answer'
    for m in re.finditer(r'</?(?:think|search|answer|information)>', text):
        if m.start() <= position < m.end():
            return 'protocol_tag'
    opening = text.rfind('<think>',0,position+1)
    closing = text.rfind('</think>',0,position+1)
    if opening>=0 and opening>=closing:
        return 'think'
    return 'unmarked_or_end'


def load_group(name, cfgpath):
    proofs = []
    cfg = read(cfgpath,proofs,EXPECTED_CONFIG[name])
    root = Path(cfg['paths']['output_root'])/'outers/outer-00000'
    complete = read(root/'sampling-complete.json',proofs)
    assert complete['accepted'] is True and complete['run_id']==cfg['run_id']
    model = complete['sampling_model_identity']
    assert model['optimizer_updates']==0 and model['checkpoint'] is None
    assert model['source_model_identity']==cfg['source_model_manifest']['sha256']
    assert model['adapter_tensor_sha256']==cfg['common_initial_adapter_tensor_sha256']
    assert len(complete['workers'])==8
    workers = [read(Path(x['path']),proofs,x['sha256']) for x in complete['workers']]
    assert sorted(w['rank'] for w in workers)==list(range(8))
    trajectories = {}; validations=Counter(); actual_load=[]
    for w in sorted(workers,key=lambda x:x['rank']):
        rank = w['rank']
        assert w['accepted'] and w['run_id']==cfg['run_id'] and w['rows']==320
        actual_load.append(dict(rank=rank,actual_dtype=w['actual_llm_load']['actual_dtype'],seed=w['actual_llm_load']['kwargs']['seed']))
        rr=w['rollouts'];gg=w['generation_trace'];hh=w['http_trace']
        rows = read(Path(rr['path']),proofs,rr['sha256'],True)
        traces = read(Path(gg['path']),proofs,gg['sha256'],True)
        https = read(Path(hh['path']),proofs,hh['sha256'],True)
        assert len(rows)==320 and len(https)==w['http_requests']
        percall=defaultdict(list)
        for lineno,t in enumerate(traces,1):
            assert t['rank']==rank and t['sampling_model_identity']==model['sha256']
            percall[t['call']].append((lineno,t))
        perhttp={}
        for lineno,h in enumerate(https,1):
            assert h['rank']==rank and h['status_code']==200 and h['generation_call'] not in perhttp
            perhttp[h['generation_call']]=(lineno,h)
        local={key(row):dict(row=row,rank=rank,row_line=i+1,row_path=rr['path'],generation_path=gg['path'],http_path=hh['path'],turns=[]) for i,row in enumerate(rows)}
        assert len(local)==320
        for call in sorted(percall):
            ts=sorted(percall[call],key=lambda x:x[1]['index'])
            active=[row for row in rows if len(row['policy_spans'])>call]
            assert len(ts)==len(active) and [t['index'] for _,t in ts]==list(range(len(active))), 'Native active-state mapping failed'
            searches=[]
            for (line,t),row in zip(ts,active):
                start,end=row['policy_spans'][call]
                assert row['response'][start:end]==t['native_output_text'], 'Native output/state span differs'
                item=dict(trace=t,line=line,query=query(t['native_output_text']),http=None)
                local[key(row)]['turns'].append(item)
                validations['generation_rows_exact_original_policy_span']+=1
                if item['query'] is not None and sum(x['http'] is not None for x in local[key(row)]['turns'])<4:
                    searches.append(item)
            if searches:
                assert call in perhttp
                hline,h=perhttp[call];payload=h['keyword_args']['json'];docs=h['response_json']['result']
                assert payload['topk']==3 and payload['return_scores'] is True
                assert payload['queries']==[x['query'] for x in searches] and len(docs)==len(searches), 'HTTP query/state ordering failed'
                for i,(item,passages) in enumerate(zip(searches,docs)):
                    item['http']=dict(line=hline,item_index=i,query=payload['queries'][i],docs=passages)
                    validations['http_queries_aligned_to_original_generated_action']+=1
            else:
                assert call not in perhttp
        assert set(perhttp)=={c for c in percall if any(x['http'] is not None for y in local.values() for x in y['turns'] if x['trace']['call']==c)}
        assert all(len(item['turns'])==len(item['row']['policy_spans']) for item in local.values())
        trajectories.update(local)
    assert len(trajectories)==2560
    return dict(cfg=cfg,proofs=proofs,model=model,trajectories=trajectories,validations=dict(validations),actual_load=actual_load)


def signature(docs, scores=False, contents=False):
    values=[]
    for p in docs:
        d=p.get('document',p)
        v={'id':d.get('id')}
        if scores:v['score']=p.get('score')
        if contents:v['contents_sha256']=sha(d.get('contents','').encode())
        values.append(v)
    return values


def sample_pointer(a,b,first,relation):
    qid=sha(a['row']['question_id'].encode())
    result=dict(question_id_sha256=qid,sibling=a['row']['sibling'],rank=a['rank'],reward_R=a['row']['reward'],reward_PR=b['row']['reward'],category=relation)
    for name,item in [('R',a),('PR',b)]:
        result[name]=dict(rollout_path=item['row_path'],rollout_line=item['row_line'],response_sha256=sha(item['row']['response'].encode()),generation_path=item['generation_path'],http_path=item['http_path'])
        h=next((t['http'] for t in item['turns'] if t['http'] is not None),None)
        if h:result[name].update(first_query_sha256=sha(h['query'].encode()),first_query_http_line=h['line'],first_query_http_item=h['item_index'],first_docs_sha256=canon(signature(h['docs'],True,True)))
    if first:
        result['earliest_native_generated_id_difference']=first
    return result


def main():
    groups={n:load_group(n,p) for n,p in [('R',R_CONFIG),('PR',PR_CONFIG)]}
    r,p=groups['R'],groups['PR'];a,b=r['trajectories'],p['trajectories']
    assert set(a)==set(b)
    assert r['cfg']['source_model_manifest']['sha256']==p['cfg']['source_model_manifest']['sha256']
    assert r['cfg']['common_initial_adapter_tensor_sha256']==p['cfg']['common_initial_adapter_tensor_sha256']
    assert r['cfg']['runtime_manifest']['sha256']==p['cfg']['runtime_manifest']['sha256']
    c=Counter();firstcall=Counter();regions=Counter();transitions=Counter();query_transitions=Counter();example_candidates=defaultdict(list)
    for k in sorted(a):
        left,right=a[k],b[k];x,y=left['row'],right['row']
        assert left['rank']==right['rank'] and canon({z:x[z] for z in ['question_id','question','answers','data_source']})==canon({z:y[z] for z in ['question_id','question','answers','data_source']})
        c['aligned_trajectories']+=1
        transitions[str(x['reward'])+'->'+str(y['reward'])]+=1
        if x['response']==y['response']:c['complete_response_equal']+=1
        lt,rt=left['turns'],right['turns']
        assert lt[0]['trace']['prompt_token_ids']==rt[0]['trace']['prompt_token_ids'], 'Initial native prompt IDs differ'
        c['initial_prompt_ids_equal']+=1
        first=None
        for i in range(min(len(lt),len(rt))):
            l,z=lt[i]['trace'],rt[i]['trace']
            idx=first_difference(l['generated_token_ids'],z['generated_token_ids'])
            if i==0:
                c['first_call_ids_equal' if idx is None else 'first_call_ids_different']+=1
                c['first_call_text_equal' if l['native_output_text']==z['native_output_text'] else 'first_call_text_different']+=1
            if idx is not None and first is None:
                pos=first_difference(l['native_output_text'],z['native_output_text'])
                rg=region(l['native_output_text'],pos);pg=region(z['native_output_text'],pos)
                first=dict(turn_index=i,generated_token_index=idx,prompt_ids_equal=l['prompt_token_ids']==z['prompt_token_ids'],native_text_difference_char_index=pos,text_region_R=rg,text_region_PR=pg,
                           R_trace_line=lt[i]['line'],PR_trace_line=rt[i]['line'],R_generated_ids_sha256=canon(l['generated_token_ids']),PR_generated_ids_sha256=canon(z['generated_token_ids']))
                regions[rg+'|'+pg]+=1;firstcall[str(i)]+=1
                c['first_divergence_on_equal_prompt_ids' if first['prompt_ids_equal'] else 'first_divergence_after_prompt_changed']+=1
        if first is None:
            if len(lt)!=len(rt):c['different_turn_count_without_shared_turn_id_difference']+=1;firstcall['turn_presence_only']+=1
            else:c['all_recorded_generated_ids_equal']+=1;firstcall['none']+=1
        lh=next((t['http'] for t in lt if t['http'] is not None),None);rh=next((t['http'] for t in rt if t['http'] is not None),None)
        relation='other'
        if lh and rh:
            c['both_executed_first_query']+=1
            if lh['query']==rh['query']:
                c['first_query_equal']+=1
                ids_equal=signature(lh['docs'])==signature(rh['docs'])
                full_equal=signature(lh['docs'],True,True)==signature(rh['docs'],True,True)
                c['equal_first_query_same_doc_ids' if ids_equal else 'equal_first_query_different_doc_ids']+=1
                c['equal_first_query_same_full_doc_score_content' if full_equal else 'equal_first_query_different_full_doc_score_content']+=1
                query_transitions['same_first_query:'+str(x['reward'])+'->'+str(y['reward'])]+=1
                if x['reward']!=y['reward']:relation='same_first_query_answer_flip'
            else:
                c['first_query_different']+=1
                c['different_first_query_same_doc_ids' if signature(lh['docs'])==signature(rh['docs']) else 'different_first_query_different_doc_ids']+=1
                query_transitions['different_first_query:'+str(x['reward'])+'->'+str(y['reward'])]+=1
                relation='first_query_divergence'
        elif lh or rh:
            c['first_search_presence_differs']+=1;query_transitions['search_presence_differs:'+str(x['reward'])+'->'+str(y['reward'])]+=1
            relation='first_search_presence_divergence'
        else:c['neither_executed_search']+=1
        lqueries=[t['http'] for t in lt if t['http'] is not None];rqueries=[t['http'] for t in rt if t['http'] is not None]
        if [h['query'] for h in lqueries]==[h['query'] for h in rqueries]:c['all_executed_queries_equal']+=1
        else:c['executed_query_sequence_different']+=1
        for i,(lhh,rhh) in enumerate(zip(lqueries,rqueries)):
            if i==0:continue
            c['subsequent_search_ordinal_comparisons']+=1
            if lhh['query']==rhh['query']:
                c['subsequent_same_query']+=1
                c['subsequent_same_query_same_full_docs' if signature(lhh['docs'],True,True)==signature(rhh['docs'],True,True) else 'subsequent_same_query_different_full_docs']+=1
            else:
                c['subsequent_different_query']+=1
                c['subsequent_different_query_different_doc_ids' if signature(lhh['docs'])!=signature(rhh['docs']) else 'subsequent_different_query_same_doc_ids']+=1
        example_candidates[relation].append(sample_pointer(left,right,first,relation))
    examples={name:sorted(vals,key=lambda x:(x['question_id_sha256'],x['sibling']))[:2] for name,vals in example_candidates.items() if name!='other'}
    result=dict(schema='opsd-recorded-precision-first-divergence-v1',accepted=True,completed_remote_unix=time.time(),scope='R BF16 vs PR FP16 outer0 before any Adam;512 identical TRAIN questions x5 siblings',
        comparison=dict(source100_sha256=r['cfg']['source_model_manifest']['sha256'],initial_adapter_tensor_sha256=r['cfg']['common_initial_adapter_tensor_sha256'],all_initial_prompt_native_ids_equal=True,generated_ids_are_original=True,no_retokenization=True,no_new_generation=True,trajectory_statistical_unit='sibling;correlated siblings are not independent questions',generation_seed='42+physical_rank'),
        counts=dict(c),first_generated_id_divergence_turn=dict(firstcall),first_divergence_native_text_region=dict(regions),strict_reward_paired_sibling_transitions=dict(transitions),first_query_conditioned_strict_reward_transitions=dict(query_transitions),private_case_pointers=examples,
        groups={name:dict(config_sha256=EXPECTED_CONFIG[name],mode=g['cfg']['precision']['mode'],sampling=g['cfg']['sampling'],source_model_identity=g['model']['source_model_identity'],optimizer_updates_before=g['model']['optimizer_updates'],sampling_kind=g['model']['kind'],sampling_model_identity=g['model']['sha256'],actual_worker_load=g['actual_load'],validations=g['validations'],input_files=g['proofs']) for name,g in groups.items()},
        limitations=['Recorded sampled trajectories, not argmax/logit comparison or trained heldout512 effects.','BF16/FP16 changes plus model casting/merge execution are bundled;same seed does not guarantee identical RNG consumption as token lengths and active batches diverge.','Earliest native generated ID index is exact;think/search/answer location uses longest common native output text prefix, not a reconstructed per-token character offset.','Search comparisons after initial use search ordinal;different histories/batches make them descriptive, not single-variable causal effects.','No repeat same-dtype source100 rollout control exists in this output;do not exclude vLLM stochastic scheduling variability.','Private case pointers keep originals on HKUST;no question/answer/generated text/doc content/token arrays copied to Mac.'])
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    main()
