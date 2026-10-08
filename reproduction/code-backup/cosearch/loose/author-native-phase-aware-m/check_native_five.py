"""Strict offline native joint/speed/full-load check; never creates successes."""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

def records(root):
    result=[]
    for path in sorted(root.rglob('*.jsonl')):
        raw=path.read_bytes()
        assert not raw or raw.endswith(b'\n'),f'Incomplete evidence line: {path}'
        result.extend(json.loads(line) for line in raw.splitlines())
    return result

def all_finite(value):
    if isinstance(value,dict):return all(all_finite(item) for item in value.values())
    if isinstance(value,list):return all(all_finite(item) for item in value)
    if isinstance(value,(int,float)):return math.isfinite(value)
    return True

def one(rows,kind,**values):
    matching=[row for row in rows if row.get('kind')==kind and all(row.get(key)==value for key,value in values.items())]
    assert len(matching)==1,(kind,values,len(matching))
    return matching[0]

def semantic_groups(row,tokenizer,inferences):
    # Author validator, same 50->5 grammar; identity passage IDs only assess validity/order.
    from verl.tools.utils.validates import validate_rerank_output
    meta=row['reward_metadata']
    scores=row['score_sums']['token_level_scores']
    size=len(row['uids'])
    assert size>0 and all(len(row[key])==size for key in ['rewards','response_ids','response_masks','advantages'])
    assert len(scores)==size and all_finite(row)
    assert len(meta['tool_score'])==len(meta['sub_query'])==size
    assert all(len(ids)==len(mask)==len(adv) for ids,mask,adv in zip(row['response_ids'],row['response_masks'],row['advantages']))
    assert all(value in [0,1] for mask in row['response_masks'] for value in mask)
    groups=defaultdict(list)
    for index,(uid,ids,mask) in enumerate(zip(row['uids'],row['response_ids'],row['response_masks'])):
        text=tokenizer.decode([token for token,enabled in zip(ids,mask) if enabled],skip_special_tokens=True)
        result=validate_rerank_output(text,50,5,{i:str(i) for i in range(1,51)})
        if result['error_type'] is not None or float(meta['tool_score'][index])<0:continue
        response=[token for token,enabled in zip(ids,mask) if enabled]
        matching=[entry for entry in inferences if entry['query']==meta['sub_query'][index] and entry['response_ids']==response]
        assert matching,'Selected ranker response has no original inference/candidate evidence'
        inventories={tuple(str(value) for value in entry['actual_candidate_ids']) for entry in matching}
        rankings={tuple(str(value) for value in entry['actual_reranked_docs']) for entry in matching}
        assert len(inventories)==len(rankings)==1,'Ambiguous candidate/response binding'
        assert all(entry['candidate_count']==50 and entry['top_m']==5 and entry['original_error_type'] is None for entry in matching)
        ranking=next(iter(rankings))
        assert len(ranking)==5 and len(set(ranking))==5
        nonzero=any(abs(adv)>1e-5 and enabled for adv,enabled in zip(row['advantages'][index],mask))
        groups[uid].append((ranking,scores[index],nonzero))
    informative=0
    for values in groups.values():
        if any(a[0]!=b[0] and a[1]!=b[1] and (a[2] or b[2]) for a in values for b in values):
            informative+=1
    return informative

def check(run,reload_run):
    from transformers import AutoTokenizer
    assert json.loads((run/'driver-exit.json').read_text())['returncode']==0
    assert json.loads((reload_run/'driver-exit.json').read_text())['returncode']==0
    launch=json.loads((run/'launch.json').read_text())
    reload_launch=json.loads((reload_run/'launch.json').read_text())
    assert launch['source_commit']==reload_launch['source_commit']=='763bf8c4d7caa363ad845d39ddfd53b81ae377bd'
    rows=records(run/'evidence');loaded=records(reload_run/'evidence')
    assert not list((run/'evidence/context-refusals').glob('*'))
    reload_contract=one(loaded,'launch')
    assert reload_contract['reload_only'] and not reload_contract['fresh_five']
    assert reload_contract['source_run']==run.name
    reload_acks=one(loaded,'all_rank_ack_gate')['records']
    assert len(reload_acks)==8 and len({row['uuid'] for row in reload_acks})==8
    assert all(str(row['job'])==str(reload_launch['job']) for row in reload_acks)
    acks=one(rows,'all_rank_ack_gate')['records']
    assert len(acks)==8 and len({row['uuid'] for row in acks})==8
    expected={(role,rank) for role in ['main','reranker'] for rank in range(4)}
    assert {(row['role'],row['rank']) for row in acks}==expected
    assert {(row['role'],row['rank']) for row in reload_acks}==expected
    bound={row['pid']:(row['role'],row['rank']) for row in acks}
    native=one(rows,'launch')['config']
    tokenizer=AutoTokenizer.from_pretrained(native['reranker_actor_rollout_ref']['model']['path'],local_files_only=True)
    outers=[row for row in rows if row.get('kind')=='outer']
    assert [row['step'] for row in outers]==[1,2,3,4,5]
    contracts=[row for row in rows if row.get('kind')=='ranker_inference_contract']
    assert contracts and all(row['candidate_count']==50 and row['top_m']==5 and row['full_prompt_tokens']<=16384 for row in contracts)
    inferences=[row for row in rows if row.get('kind')=='ranker_ranking_result']
    costs=[];milestones=[]
    for step in range(1,6):
        outer=one(rows,'outer',step=step)
        assert all_finite(outer['metrics'])
        start=one(rows,'generation_start',step=step,validation=False)
        ends=[one(rows,'global_update_completed',step=step,role=role)['monotonic_ns'] for role in ['main','reranker']]
        cost=(max(ends)-start['monotonic_ns'])/1e9
        assert 0<cost<600,('Training iteration exceeds speed goal',step,cost)
        costs.append(cost)
        updates=[row for row in rows if row.get('kind')=='update' and row.get('outer_step')==step]
        assert len(updates)==8 and {(row['role'],row['rank']) for row in updates}==expected
        for row in updates:
            assert bound[row['pid']]==(row['role'],row['rank'])
            assert row['actual_adam_delta']>0 and all_finite(row['original_metrics'])
            assert row['parameter_dtype']=='torch.float32'
        for role in ['main','reranker']:
            global_row=one(rows,'global_pre_update',step=step,role=role)
            observed_uids={uid for row in updates if row['role']==role for uid in row['local_uids']}
            assert observed_uids==set(global_row['uids']),'Worker UID coverage differs from original global batch'
        global_ranker=one(rows,'global_pre_update',step=step,role='reranker')
        informative=semantic_groups(global_ranker,tokenizer,inferences)
        genuine=informative>0 and all(row['lr_before']>0 and row['parameter_probe_delta_l1']>0
            and any(value>1e-5 for value in row['original_gradient_norms']) for row in updates)
        if genuine:milestones.append(step)
    assert len(milestones)>=2,('Fewer than two genuine ranker learning steps',milestones)
    validations=[row for row in rows if row.get('kind')=='generation_start' and row.get('validation')]
    assert validations and sum(row['input_rows'] for row in validations)==64
    saved=[row for row in rows if row.get('kind')=='checkpoint_saved_full_digest']
    restored=[row for row in loaded if row.get('kind')=='checkpoint_loaded_full_digest']
    assert len(saved)==len(restored)==8
    assert {(row['role'],row['rank']) for row in saved}==expected
    assert {(row['role'],row['rank']) for row in restored}==expected
    for row in restored:
        previous=one(rows,'checkpoint_saved_full_digest',role=row['role'],rank=row['rank'])
        assert row['state']==previous['state'] and row['source_job']==launch['job']
        assert row['path']==previous['path']
        assert Path(row['path']).parent==run.resolve()/'checkpoints/global_step_5'
        assert row['state']['master_dtypes']==['torch.float32']
        assert row['state']['model_master']['tensor_bytes']>0
        assert row['state']['optimizer']['tensor_bytes']>0
    assert not [row for row in loaded if row.get('kind')=='update']
    one(loaded,'pure_reload_returned',step=5,no_training_fit_called=True)
    return dict(accepted=True,real_joint_outer_updates=5,training_seconds=costs,
        genuine_ranker_steps=milestones,validation_questions=64,full_reload_ranks=8)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--reload-run',required=True)
    args=parser.parse_args()
    print(json.dumps(check(Path(args.run),Path(args.reload_run)),allow_nan=False))
