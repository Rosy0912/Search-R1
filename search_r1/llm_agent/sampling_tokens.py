"""Opt-in raw-token generation with exact per-turn sampled context audits."""
import uuid
import numpy as np
import torch
from verl import DataProto
from .sampling_trace import require, prefix, sha, audit_row


def left_pad(rows,pad):
    require(bool(rows) and all(rows),'empty request')
    width=max(map(len,rows))
    ids=torch.full((len(rows),width),pad,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        ids[i,-len(row):]=torch.tensor(row); mask[i,-len(row):]=1
    return dict(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1)*mask)


def objects(rows):
    result=np.empty(len(rows),dtype=object); result[:]=rows
    return result


def audit_batch(data):
    require(data.meta_info.get('preserve_sampling_tokens') is True,'missing opt-in identity')
    snapshot=data.meta_info.get('policy_snapshot_id')
    require(isinstance(snapshot,str) and bool(snapshot),'missing policy snapshot')
    require('fresh200_turns' in data.non_tensor_batch and 'fresh200_row_id' in data.non_tensor_batch,'missing row journals')
    row_ids=data.non_tensor_batch['fresh200_row_id']
    require(len(row_ids)==len(data) and len(set(map(str,row_ids)))==len(data),'row identity duplication')
    width=data.batch['prompts'].shape[1]; reports=[]
    for i,turns in enumerate(data.non_tensor_batch['fresh200_turns']):
        require(all('effective_stop_provenance' in turn for turn in turns), 'missing actual effective group sampling proof')
        role='greedy_model_logprob_not_sampling_policy' if data.meta_info.get('fresh200_evaluation') is True else 'stochastic_sampling_policy'
        require(all(turn['effective_stop_provenance'].get('probability_role', 'stochastic_sampling_policy')==role for turn in turns), 'training/evaluation probability role mismatch')
        row=data.batch['input_ids'][i].cpu().tolist(); attention=data.batch['attention_mask'][i].cpu().tolist()
        require(all(v in (0,1) for v in attention),'invalid attention')
        require(attention[:width]==sorted(attention[:width]) and
                attention[width:]==sorted(attention[width:],reverse=True), 'padding topology')
        prompt=[t for t,v in zip(row[:width],attention[:width]) if v]
        response=[t for t,v in zip(row[width:],attention[width:]) if v]
        require(data.batch['responses'][i].cpu().tolist()==row[width:] and data.batch['prompts'][i].cpu().tolist()==row[:width],'input tensor divergence')
        model=data.batch['sampling_mask'][i].cpu().tolist(); info=data.batch['info_mask'][i,width:].cpu().tolist()
        require(model==[bool(v) for v in info] and not any(m and not a for m,a in zip(model,attention[width:])),'sampling/info mask divergence')
        cumulative=0; expected=[]
        for valid in attention:
            cumulative+=valid; expected.append((cumulative-1)*valid)
        require(data.batch['position_ids'][i].cpu().tolist()==expected,'position reconstruction')
        selected=data.batch['sampling_log_probs'][i].cpu().tolist()
        reports.append(audit_row(prompt,response,[int(m) for m,a in zip(model,attention[width:]) if a],
            [p for p,a in zip(selected,attention[width:]) if a],turns,str(data.non_tensor_batch['fresh200_row_id'][i]),snapshot))
    return dict(schema_version=1,context_verified=True,native_probability_verified=False,
                row_count=len(reports),rows_sha256=sha(reports),policy_snapshot_id=snapshot,
                probability_role='greedy_model_logprob_not_sampling_policy' if data.meta_info.get('fresh200_evaluation') is True else 'stochastic_sampling_policy')


def run(manager,gen_batch,initial_input_ids):
    config,tokenizer=manager.config,manager.tokenizer
    require(not config.no_think_rl,'no_think retokenization unsupported')
    require(type(config.num_gpus) is int and config.num_gpus>0,'invalid generation rank count')
    require(gen_batch.meta_info.get('preserve_sampling_tokens') is True,'missing worker opt-in')
    snapshot=gen_batch.meta_info.get('policy_snapshot_id')
    require(isinstance(snapshot,str) and bool(snapshot),'missing policy snapshot')
    row_ids=gen_batch.non_tensor_batch.get('fresh200_row_id')
    require(row_ids is not None and len(row_ids)==len(gen_batch) and len(set(map(str,row_ids)))==len(gen_batch),'missing unique row identities')
    row_ids=list(map(str,row_ids)); ids=initial_input_ids.cpu().long(); masks=gen_batch.batch['attention_mask'].cpu().long()
    require(ids.shape==masks.shape,'initial prompt mismatch')
    prompts=[[int(t) for t,m in zip(row.tolist(),mask.tolist()) if m] for row,mask in zip(ids,masks)]
    require(all(0<len(p)<=config.max_start_length for p in prompts),'initial context would be cropped')
    require(all(m in (0,1) for mask in masks.tolist() for m in mask),'initial mask')
    pad=tokenizer.pad_token_id
    responses,probabilities,model_masks,journals=([[] for _ in prompts] for _ in range(4))
    active=[True]*len(prompts); turns_stats,actions,searches=([0]*len(prompts) for _ in range(3))
    for ordinal in range(config.max_turns+int(config.force_final_answer)):
        indices=[i for i,a in enumerate(active) if a]
        if not indices: break
        final=ordinal==config.max_turns
        requests=[prompts[i]+responses[i] for i in indices]
        require(all(len(r)<=config.max_prompt_length for r in requests),'rolling context would be cropped')
        count=(-len(indices))%config.num_gpus
        identities=[row_ids[i] for i in indices]+['padding-'+uuid.uuid4().hex for _ in range(count)]
        meta=gen_batch.meta_info.copy(); meta.update(fresh200_call_id=uuid.uuid4().hex,fresh200_turn_index=ordinal,recompute_log_prob=False)
        call=DataProto.from_dict(left_pad(requests+[requests[0]]*count,pad),non_tensors={'fresh200_row_id':identities},meta_info=meta)
        output=manager.actor_rollout_wg.generate_sequences(call)
        require(len(output)==len(identities) and 'fresh200_sampling' in output.non_tensor_batch,'missing native capture')
        records=output.non_tensor_batch['fresh200_sampling'].tolist()
        require([r['row_id'] for r in records]==identities,'padded row identity')
        texts=['']*len(prompts); cuts=[]
        for local,i in enumerate(indices):
            record=records[local]
            require(record['actual_request_token_ids']==requests[local] and record['policy_snapshot_id']==snapshot,'actual request changed')
            raw=record['raw_generated_token_ids']
            require(output.batch['responses'][local,:len(raw)].cpu().tolist()==raw,'raw token transport')
            native=torch.tensor(record['selected_logprobs'],dtype=torch.float32)
            require(torch.equal(output.batch['sampling_log_probs'][local,:len(raw)].cpu(),native),'raw probability transport')
            n,text=prefix(raw,lambda ts: tokenizer.decode(ts,skip_special_tokens=False,clean_up_tokenization_spaces=False))
            cuts.append(n); texts[i]=text
        obs,dones,valid,is_search=manager.execute_predictions(texts,tokenizer.pad_token,torch.tensor(active),do_search=not final)
        for local,i in enumerate(indices):
            record=records[local]
            env=[] if final else tokenizer.encode(obs[i],add_special_tokens=False)[:config.max_obs_length]
            n=cuts[local]
            require(len(responses[i])+n+len(env)<=config.max_prompt_length,'response would be truncated')
            record.update(kept_token_count=n,environment_token_ids=env); record['turn_sha256']=sha(record)
            journals[i].append(record); responses[i].extend(record['raw_generated_token_ids'][:n]+env)
            probabilities[i].extend(record['selected_logprobs'][:n]+[0.]*len(env)); model_masks[i].extend([1]*n+[0]*len(env))
            turns_stats[i]+=1; actions[i]+=int(valid[i]); searches[i]+=int(is_search[i]); active[i]=not bool(dones[i])
    require(all(journals),'missing trajectory completion')
    left=left_pad(prompts,pad); width=max(map(len,responses))
    values=torch.full((len(prompts),width),pad,dtype=torch.long); attention=torch.zeros_like(values); model=torch.zeros_like(values); selected=torch.zeros_like(values,dtype=torch.float32)
    for i,row in enumerate(responses):
        values[i,:len(row)]=torch.tensor(row); attention[i,:len(row)]=1
        model[i,:len(row)]=torch.tensor(model_masks[i]); selected[i,:len(row)]=torch.tensor(probabilities[i])
    full_attention=torch.cat((left['attention_mask'],attention),-1)
    result=DataProto.from_dict(dict(prompts=left['input_ids'],responses=values,responses_with_info_mask=torch.where(model.bool(),values,pad),
        input_ids=torch.cat((left['input_ids'],values),-1),attention_mask=full_attention,info_mask=torch.cat((left['attention_mask'],model),-1),
        position_ids=(full_attention.cumsum(-1)-1)*full_attention,sampling_log_probs=selected,sampling_mask=model.bool()),
        non_tensors={'fresh200_row_id':row_ids,'fresh200_turns':objects(journals)},meta_info=gen_batch.meta_info.copy())
    result.meta_info.update(turns_stats=turns_stats,active_mask=active,valid_action_stats=actions,valid_search_stats=searches)
    result.meta_info['sampling_context_audit']=audit_batch(result)
    return result
