"""Owned CUDA full-context/static-RoPE 3B resource diagnostic; never an RL milestone."""
import os, json
# Establish the full leased physical mask before importing HF/Torch.
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(json.loads(os.environ['COSEARCH_RANKER_LEASE_UUIDS']))
import copy, time, threading
from pathlib import Path
from types import SimpleNamespace
import requests, torch
from omegaconf import OmegaConf
from search_r1.baselines.cosearch import GenerativeRanker, RankerSequenceLoss, eval_decoder_checkpoint, sequence_ppo_loss
from transformers import Qwen2Config, Qwen2ForCausalLM
import torch.nn.functional as F

def _model(kernel):
    torch.manual_seed(902)
    torch.set_num_threads(1)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=43, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=256,
        eos_token_id=0, pad_token_id=0, attention_dropout=0.,
        attn_implementation=kernel)).eval()


def _single_logp(model, ids, response_length, temperature):
    logits = model(ids[None], use_cache=False).logits[:, -response_length-1:-1]
    return logits[0].float().div(temperature).log_softmax(-1).gather(
        -1, ids[-response_length:, None]).squeeze(-1)


def _examples(model, rows, temperature):
    events = []
    for i in range(rows):
        prompt = torch.tensor([2 + (i*3+j) % 39 for j in range(1 + (i*7) % 8)])
        response_length = 1 + (i*3) % 6
        # PAD=EOS appears as the actual final response action and stays scored.
        response = torch.tensor([3 + (i+j*5) % 37 for j in range(response_length-1)] + [0])
        ids = torch.cat((prompt, response))
        with torch.no_grad():
            logp = _single_logp(model, ids, response_length, temperature)
        # Exercise both clipped and unclipped ratios, not only ratio=1.
        offset = (0.0, .5, -.4, .03)[i % 4]
        events.append(dict(input_ids=ids, response_length=response_length,
                           old_logp=logp + offset))
    advantages = torch.tensor([(.8, -.7, 0., 1.1, -.2, .4, -.6, .3)[i % 8]
                               for i in range(rows)])
    return events, advantages


def _pack(events):
    width = max(e['input_ids'].numel() for e in events)
    response_width = max(e['response_length'] for e in events)
    ids = torch.stack([F.pad(e['input_ids'], (width-e['input_ids'].numel(), 0), value=0)
                       for e in events])
    masks = torch.stack([F.pad(torch.ones_like(e['input_ids']),
                         (width-e['input_ids'].numel(), 0)) for e in events])
    old = torch.stack([F.pad(e['old_logp'], (response_width-e['response_length'], 0))
                       for e in events])
    lengths = torch.tensor([e['response_length'] for e in events])
    return ids, masks, lengths, old, response_width



torch.backends.cuda.matmul.allow_tf32=False
assert torch.cuda.device_count()==8
master=7; device=torch.device('cuda',master)
all_devices=[7,0,1,2,3,4,5,6]
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
run=Path(os.environ['COSEARCH_PROBE_RUN'])
record=dict(probe_only=True,acceptance_updates=0,native_parallel_cases=[])
# Native FP32 SDPA scatter, per-row expectation, summed replica gradients,
# actual clipping and two Adam steps compared to original unpadded oracle.
for count in (2,8):
    for rows in (8,3,1):
        for temperature in (.7,1.3):
            initial=_model('sdpa')
            events,advantages=_examples(initial,rows,temperature)
            packed=_pack(events)
            ids,mask,lengths,old,width=[x.to(device) if torch.is_tensor(x) else x for x in packed]
            advantages=advantages.to(device)
            reference=copy.deepcopy(initial).to(device); actual=copy.deepcopy(initial).to(device)
            optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.01) for m in (reference,actual)]
            maximum_error=0.
            for step in range(2):
                for opt in optimizers:opt.zero_grad(set_to_none=True)
                expected=[]
                for e,a in zip(events,advantages):
                    loss=sequence_ppo_loss(_single_logp(reference,e['input_ids'].to(device),e['response_length'],temperature),e['old_logp'].to(device),a,.2)
                    expected.append(loss.detach());(loss/rows).backward()
                with eval_decoder_checkpoint(actual,True):
                    parallel=torch.nn.DataParallel(RankerSequenceLoss(actual,temperature,.2),device_ids=all_devices[:min(count,rows)],output_device=master)
                    losses=parallel(ids,mask,lengths,old,advantages,width)
                    torch.testing.assert_close(losses.detach(),torch.stack(expected),atol=3e-6,rtol=3e-5)
                    (losses.sum()/rows).backward()
                    del parallel,losses
                for p,q in zip(reference.parameters(),actual.parameters()):
                    assert p.grad is not None and q.grad is not None
                    maximum_error=max(maximum_error,float((p.grad-q.grad).abs().max()))
                    torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5)
                norms=[torch.nn.utils.clip_grad_norm_(m.parameters(),.05) for m in (reference,actual)]
                assert float(norms[0])>.05
                torch.testing.assert_close(norms[0],norms[1],atol=3e-6,rtol=3e-5)
                for opt in optimizers:opt.step()
                for p,q in zip(reference.parameters(),actual.parameters()):torch.testing.assert_close(p,q,atol=2e-7,rtol=2e-6)
                for left,right in zip(optimizers[0].state.values(),optimizers[1].state.values()):
                    for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(left[key],right[key],atol=2e-7,rtol=3e-5)
            record['native_parallel_cases'].append(dict(requested_devices=count,effective_devices=min(count,rows),rows=rows,temperature=temperature,max_gradient_abs_error=maximum_error))
            del initial,reference,actual,optimizers,ids,mask,lengths,old,advantages,events,expected,loss,norms,p,q,left,right
            for i in all_devices:
                with torch.cuda.device(i):torch.cuda.empty_cache()
print('COSEARCH_NATIVE_PARALLEL_MATH_JSON='+json.dumps(record),flush=True)
c=OmegaConf.load(root/'repo/verl/trainer/config/ppo_baselines.yaml').baselines.ranker
if (run/'pre-warm-resolved.yaml').exists():
    c.keep=OmegaConf.load(run/'pre-warm-resolved.yaml').baselines.ranker.keep
c.model_path=os.environ.get('COSEARCH_RANKER', str(root/'assets-root/assets/qwen2.5-3b'))
c.device='cpu';c.attn_implementation='sdpa';c.cpu_threads=32;c.response_only_logits=True
c.phase_gpu=True;c.phase_gpu_devices=8;c.phase_seed=42;c.inference_parallel=True;c.training_parallel=True
c.full_context_extrapolation=True
c.reuse_generation_logits=True;c.inference_batch_size=1;c.inference_token_budget=131072;c.gradient_checkpointing=True
# Repeat four full real search prompts to exercise all eight generation GPUs;
# no candidate truncation, model replacement, or artificial acceptance reward.
queries=['Who wrote The Old Man and the Sea?', 'Where was Marie Curie born?', 'What is the capital of France?', 'Who discovered penicillin?']*2
docs=requests.post('http://127.0.0.1:18086/retrieve',json=dict(queries=queries,topk=50,return_scores=True),timeout=60).json()['result']
assert len(docs)==8 and all(len(d)==50 for d in docs)
# Controlled context stress, not a replay of the missing K oversized row.
# Retain each real retrieved document, repeating its entire contents to create
# a transparent full50 synthetic long fixture. Never truncate any passage.
base_docs=copy.deepcopy(docs)

r=GenerativeRanker(c);r.question_uids=list(range(len(queries)))
config_before=copy.deepcopy(r.model.config.to_dict())
rope_before=r.model.model.rotary_emb.inv_freq.detach().clone()
encoded=[]
for i,(question,base) in enumerate(zip(queries,base_docs)):
    repeat=1
    while True:
        row=copy.deepcopy(base)
        for d in row:d['document']['contents']=('\n'.join([d['document']['contents']]*repeat))
        passages='\n'.join(f'Doc {j+1}: {d["document"]["contents"]}' for j,d in enumerate(row))
        prompt=(f'Question context:\n{question}\nSearch query: {question}\nCandidates:\n{passages}\n'
                f'Select exactly {c.keep} distinct document IDs ordered by relevance. '
                'Think briefly, then output <ranking>[1, 2, ...]</ranking>.\n')
        ids=r.tokenizer(prompt,return_token_type_ids=False)['input_ids']
        if len(ids)>=33792:break
        repeat+=1
    docs[i]=row;encoded.append(ids)
fixture=dict(synthetic_stress=True,not_replay_of_K=True,queries=queries,documents=docs,
             input_ids=encoded,prompt_tokens=[len(x) for x in encoded],model_config_limit=32768)
(run/'long-context-fixture.json').write_text(json.dumps(fixture)+'\n')
# Reserve the approximately3GiB/card idle actor footprint for the pre-Ray gate.
# Production still needs its actual actor/ranker coexistence evidence.
headroom=[torch.empty(3*1024**3,dtype=torch.uint8,device=torch.device('cuda',i)) for i in range(8)]
record['actor_headroom_reservation_bytes_per_card']=3*1024**3
record['long_prompt_tokens']=fixture['prompt_tokens']

original_generate=r._generate_batch
def observed_generate(model,device,inputs,sampling):
    events,wall=original_generate(model,device,inputs,sampling)
    for event in events:event['diagnostic_generation_device']=int(device)
    return events,wall
r._generate_batch=observed_generate
states=[SimpleNamespace(source=i,trajectory=i) for i in range(len(queries))]
started=time.perf_counter();r.rank_batch(queries,queries,docs,states)
record.update(wall_seconds=time.perf_counter()-started,events=len(r.events),cost=r.cost.copy())
assert len(r.events)==8 and all(e['question_uid']==i for i,e in enumerate(r.events))
assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in r.model.parameters())
# Exercise the actual parquet alias type through reward preparation before any
# production rollout. All unique question UIDs prevent synthetic RL selection.
import pandas as pd, numpy as np
frame=pd.read_parquet(root/'assets-root/data/paper-grpo/train.parquet',columns=['reward_model'])
aliases=[item['ground_truth']['target'] for item in frame['reward_model']
         if len(item['ground_truth']['target'])>1][:8]
assert len(aliases)==8 and all(isinstance(a,np.ndarray) for a in aliases)
saved_events=list(r.events);r.events=[dict(e) for e in saved_events]
contract=r._update({e['trajectory_uid']:(.8,a,True) for e,a in zip(r.events,aliases)})
assert contract['ranker/training_events']==0 and r.cost['optimizer_steps']==0
r.events=saved_events
record['real_parquet_alias_contract']=dict(rows=8,alias_counts=[len(a) for a in aliases],
                                         types=[type(a).__name__ for a in aliases],metrics=contract)
with r.execution_phase(training=True):
    # Observe actual generation devices; input sorting and token-budget chunks
    # make fixed original row offsets insufficient to prove replica coverage.
    by_device={}
    for event in r.events:by_device.setdefault(event['diagnostic_generation_device'],event)
    assert set(by_device)==set(range(8))
    events=[by_device[i] for i in all_devices]
    checked_events=events+r.events[:4]
    deviations=[]
    with torch.no_grad():
        for e in checked_events:
            logp=r.response_logp(e['input_ids'][None].to(r.device),e['response_length'])[0].cpu()
            torch.testing.assert_close(logp,e['old_logp'],atol=3e-4,rtol=3e-5)
            deviations.append(float((logp-e['old_logp']).abs().max()))
    # Exercise the full256 response budget in long-context training, regardless
    # of early EOS in the actual sampled response used for cache parity above.
    events=[dict(e) for e in events]
    with torch.no_grad():
        for e in events:
            prompt=e['input_ids'][:-e['response_length']]
            response=torch.tensor([42]*255+[r.tokenizer.eos_token_id])
            e['input_ids']=torch.cat((prompt,response));e['response_length']=256
            e['old_logp']=r.response_logp(e['input_ids'][None].to(r.device),256)[0].cpu()
    packed=_pack(events)
    ids,mask,lengths,old,width=[x.to(r.device) if torch.is_tensor(x) else x for x in packed]
    advantages=torch.tensor([1.,-.7,.3,-1.1,.8,-.2,.5,-.4],device=r.device)
    checkpoint_calls={};lock=threading.Lock()
    original=r.model.model._gradient_checkpointing_func
    def counted_checkpoint(*args,**kwargs):
        with lock:
            index=torch.cuda.current_device();checkpoint_calls[index]=checkpoint_calls.get(index,0)+1
        return original(*args,**kwargs)
    r.model.model._gradient_checkpointing_func=counted_checkpoint
    before=next(r.model.parameters()).detach().cpu().clone()
    started=time.perf_counter()
    parallel=torch.nn.DataParallel(RankerSequenceLoss(r.model,c.temperature,c.clip_ratio),device_ids=r.phase_devices,output_device=r.device.index)
    losses=parallel(ids,mask,lengths,old,advantages,width)
    assert losses.shape==(8,) and torch.isfinite(losses).all()
    (losses.sum()/8).backward()
    assert set(checkpoint_calls)==set(range(8)) and all(n>=r.model.config.num_hidden_layers for n in checkpoint_calls.values())
    norm=torch.nn.utils.clip_grad_norm_(r.model.parameters(),c.grad_clip)
    assert torch.isfinite(norm) and float(norm)>1e-5
    r.optimizer.step()
    assert all(torch.isfinite(p).all() for p in r.model.parameters())
    record.update(stress_training_wall_seconds=time.perf_counter()-started,checkpoint_calls=checkpoint_calls,
        native_old_logp_max_abs_delta=max(deviations),checked_generation_devices=sorted(by_device),checked_queries=queries[:4],stress_gradient_norm=float(norm),
        diagnostic_parameter_abs_change=float((next(r.model.parameters()).detach().cpu()-before).abs().sum()))
    assert record['diagnostic_parameter_abs_change']>0
    r.model.model._gradient_checkpointing_func=original
    del parallel,losses,ids,mask,lengths,old,advantages,logp
record['cost_after_stress']=r.cost.copy()
assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in r.model.parameters())
r.events.clear();r.validation=True
started=time.perf_counter();r.rank_batch(queries[:8],queries[:8],docs[:8],states[:8])
assert not r.events and r.cost.get('validation_score_forward_tokens',0)==0
record['validation_probe_wall_seconds']=time.perf_counter()-started
assert r.model.config.to_dict()==config_before
assert torch.equal(r.model.model.rotary_emb.inv_freq.cpu(),rope_before.cpu())
record['original_model_config_unchanged']=True
record['original_static_rope_unchanged']=True
record['synthetic_full_response_budget']=256
record['passed']=True
(run/'native-long-context-probe.json').write_text(json.dumps(record,indent=2)+'\n')
print('COSEARCH_NATIVE_GPU_PROBE_JSON='+json.dumps(record),flush=True)
