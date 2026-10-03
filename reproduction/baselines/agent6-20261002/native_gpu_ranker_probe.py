"""Owned native 3B GPU throughput/math/memory probe; NEVER acceptance updates."""
import json, os, time
from pathlib import Path
from types import SimpleNamespace
import requests, torch
from omegaconf import OmegaConf
from search_r1.baselines.cosearch import GenerativeRanker, sequence_ppo_loss

root=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
run=Path(os.environ['COSEARCH_PROBE_RUN'])
c=OmegaConf.load(root/'repo/verl/trainer/config/ppo_baselines.yaml').baselines.ranker
c.model_path=str(root/'assets-root/assets/qwen2.5-3b')
c.device='cpu';c.attn_implementation='sdpa';c.cpu_threads=32;c.response_only_logits=True
c.phase_gpu=True;c.reuse_generation_logits=True;c.inference_batch_size=4;c.inference_token_budget=32768
c.gradient_checkpointing=True
queries=['Who wrote The Old Man and the Sea?', 'Where was Marie Curie born?',
         'What is the capital of France?', 'Who discovered penicillin?']
docs=requests.post('http://127.0.0.1:18086/retrieve',json=dict(queries=queries,topk=50,return_scores=True),timeout=60).json()['result']
assert all(len(d)==50 for d in docs)
r=GenerativeRanker(c);r.question_uids=list(range(4))
states=[SimpleNamespace(source=i,trajectory=i) for i in range(4)]
torch.manual_seed(42)
started=time.perf_counter();r.rank_batch(queries,queries,docs,states)
record=dict(probe_only=True,acceptance_updates=0,wall_seconds=time.perf_counter()-started,
            parameter_dtype=str(next(r.model.parameters()).dtype),cost=r.cost.copy(), events=len(r.events))
assert all(p.dtype==torch.float32 for p in r.model.parameters())
# Actual native full-prefix teacher forcing vs cached raw scores.
with r.execution_phase(training=True):
    r.optimizer.zero_grad()
    deviations=[]
    checkpoints=[0]
    original_checkpoint=r.model.model._gradient_checkpointing_func
    def counted_checkpoint(*args,**kwargs):
        checkpoints[0]+=1
        return original_checkpoint(*args,**kwargs)
    r.model.model._gradient_checkpointing_func=counted_checkpoint
    for i,e in enumerate(r.events):
        ids=e['input_ids'][None].to(r.device)
        logp=r.response_logp(ids,e['response_length'])[0]
        delta=float((logp.detach().cpu()-e['old_logp']).abs().max())
        deviations.append(delta)
        torch.testing.assert_close(logp.detach().cpu(),e['old_logp'],atol=3e-4,rtol=3e-5)
        # Synthetic stress gradients are diagnostic, NOT real RL or acceptance.
        loss=sequence_ppo_loss(logp,e['old_logp'].to(r.device),torch.tensor(1. if i%2 else -1.,device=r.device),c.clip_ratio)/len(r.events)
        loss.backward()
    assert checkpoints[0] >= len(r.events)*r.model.config.num_hidden_layers
    record['checkpoint_calls']=checkpoints[0]
    r.model.model._gradient_checkpointing_func=original_checkpoint
    norm=torch.nn.utils.clip_grad_norm_(r.model.parameters(),c.grad_clip)
    assert torch.isfinite(norm) and float(norm)>0
    r.optimizer.step()
    record.update(native_old_logp_max_abs_delta=max(deviations),stress_gradient_norm=float(norm),
                  stress_optimizer_step=True,checkpointing_actual_required=True)
record['cost_after_stress']=r.cost
assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in r.model.parameters())
# Greedy 64-question validation will not retain old logprobs or training events.
r.events.clear();r.validation=True
started=time.perf_counter();r.rank_batch(queries[:2],queries[:2],docs[:2],states[:2])
assert not r.events and r.cost.get('validation_score_forward_tokens',0)==0
record['validation_probe_wall_seconds']=time.perf_counter()-started
record['passed']=True
(run/'native-ranker-probe.json').write_text(json.dumps(record,indent=2)+'\n')
print('COSEARCH_NATIVE_GPU_PROBE_JSON='+json.dumps(record),flush=True)
