"""Owned h100-9 CPU-only read of hash-bound3295 tensors; no model/GPU calls."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import socket
import time

assert socket.gethostname().split('.')[0] == 'h100-9'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
start = time.monotonic()
run = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-probability-probe-3295')
bindings = json.loads(BINDINGS_JSON)  # injected literal identity from terminal metadata
records = []
all_delta = []


def digest_tensor(value):
    value = value.detach().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()+b'\0'+str(tuple(value.shape)).encode()+b'\0')
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


for rank, binding in enumerate(bindings):
    path = run/('gpu-records/rank-%02d/window.pt' % rank)
    assert path == Path(binding['path']) and path.resolve() == path
    data = path.read_bytes()
    assert len(data) == binding['bytes'] and hashlib.sha256(data).hexdigest() == binding['sha256']
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    inputs = payload['new_raw_inputs']
    mask = payload['actual_emitted_mask']
    sampled = payload['actual_sampled_logprobs']
    actor = payload['actual_actor_logprobs']
    assert mask.dtype == torch.bool and tuple(mask.shape) == (320, 500)
    assert actor.shape == sampled.shape == mask.shape
    assert all(t.device.type == 'cpu' for t in [mask, sampled, actor, *inputs.values()])
    ids, attention, positions, responses = (inputs[k] for k in ['input_ids','attention_mask','position_ids','responses'])
    assert ids.shape == attention.shape == positions.shape
    assert torch.equal(ids[:,-500:], responses)
    assert torch.equal(attention[:,-500:].bool(), mask)
    assert torch.equal(inputs['__sampling_log_probs'][mask].double(), sampled[mask].double())
    assert torch.isfinite(actor[mask]).all() and torch.isfinite(sampled[mask]).all()
    delta = actor.double()-sampled.double()
    current = delta[mask]
    all_delta.append(current)
    maximum = int(delta.abs().masked_fill(~mask, -1.).argmax())
    row, offset = divmod(maximum, 500)
    target_col = ids.shape[1]-500+offset
    assert mask[row,offset] and attention[row,target_col-1]
    prefix_mask = attention[row,:target_col].bool()
    prefix = ids[row,:target_col][prefix_mask].contiguous()
    prefix_positions = positions[row,:target_col][prefix_mask].contiguous()
    extreme = dict(row=row,response_offset=offset,prediction_physical_column=target_col-1,
                   target_token_id=int(responses[row,offset]),sampled_mu_logprob=float(sampled[row,offset]),
                   actor_logprob=float(actor[row,offset]),signed_delta=float(delta[row,offset]),
                   absolute_delta=float(delta[row,offset].abs()),context_length=len(prefix),
                   context_ids_shape_dtype_bytes_sha256=digest_tensor(prefix),
                   context_positions_shape_dtype_bytes_sha256=digest_tensor(prefix_positions),
                   context_token_ids=prefix.tolist(),context_positions=prefix_positions.tolist())
    records.append(dict(rank=rank,tensor_file_identity=binding,emitted_count=len(current),
                        mean_abs=float(current.abs().mean()),rms=float(current.square().mean().sqrt()),
                        min_delta=float(current.min()),max_delta=float(current.max()),
                        above_diagnostic001_count=int((current.abs()>.01).sum()),extreme=extreme))

values = torch.cat(all_delta)
assert len(values) == 808994 and abs(float(values.abs().max())-1.3392277956008911)<1e-12
assert int((values.abs()>.01).sum()) == 84390
assert bool(torch.isfinite(values).all()) and float(values.abs().max())<30
global_extreme = max(records, key=lambda r:r['extreme']['absolute_delta'])
usage = resource.getrusage(resource.RUSAGE_SELF)
result = dict(status='ACTUAL3295_HASH_BOUND_TENSOR_CPU_EXTREMES_OBSERVED_NO_TRAINING_OR_PROBABILITY_ACCEPTANCE',
              job_id=3295,host=socket.gethostname(),mode='Read-only CPU analysis of owned saved tensors; management process, not Slurm allocation',
              tensor_assets_remained_on_h1009=True,tensor_assets_transferred=False,new_generation_invocations=0,
              model_forwards=0,optimizer_writes=0,backward_calls=0,cuda_initialized=torch.cuda.is_initialized(),
              raw_emitted_count=len(values),raw_mean_abs=float(values.abs().mean()),raw_rms=float(values.square().mean().sqrt()),
              raw_max_abs=float(values.abs().max()),raw_min_delta=float(values.min()),raw_max_delta=float(values.max()),
              raw_abs_quantiles={str(q):float(torch.quantile(values.abs(),q)) for q in [.5,.9,.95,.99,.999,1.]},
              above_diagnostic001_count=int((values.abs()>.01).sum()),
              raw_k3_mean=float((torch.expm1(values)-values).mean()),
              k3_scope='Raw unweighted sampler-mu emitted tokens and actor execution; not Tree loss mask/weights or production threshold calibration',
              global_extreme_rank=global_extreme['rank'],ranks=records,
              analysis_wall_seconds=time.monotonic()-start,self_cpu_seconds=usage.ru_utime+usage.ru_stime,
              self_peak_rss_kib=usage.ru_maxrss,probability_accepted=False,training_accepted=False)
assert not result['cuda_initialized']
output = run/'cpu-extremes-analysis.json'
data = json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()
with output.open('xb') as stream: stream.write(data)
summary = dict(result)
summary['ranks'] = [dict(r,extreme={k:v for k,v in r['extreme'].items()
                                  if k not in ['context_token_ids','context_positions']}) for r in records]
summary['saved_full_context_report'] = dict(path=str(output),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(summary,ensure_ascii=False,sort_keys=True,allow_nan=False))
