import json,os,time
from pathlib import Path
import ray
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
out=root/'runs/opsd-acceptance-2875/evidence/teacher-precision-cpu-probe.json'
assert not out.exists()
ray.init(address='127.0.0.1:16384')
@ray.remote(num_cpus=8,num_gpus=0)
class Probe:
 def run(self):
  import gc,resource,torch
  import torch.nn.functional as F
  from transformers import AutoModelForCausalLM,AutoTokenizer
  assert os.environ['SLURM_JOB_ID']=='2875'
  assert '/job_2875/' in Path('/proc/self/cgroup').read_text()
  assert not torch.cuda.is_available(), 'CPU-only diagnostic'
  torch.set_num_threads(8);torch.set_num_interop_threads(1)
  base=root/'assets-root/assets/qwen2.5-3b'
  tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
  fragment=tokenizer.encode('Question: What is the capital of France? [Teacher-only training context: accepted answers: Paris] Information: Paris is the capital of France. ',add_special_tokens=False)
  ids=torch.tensor((fragment*((3712+len(fragment)-1)//len(fragment)))[:3712])[None,:]
  mask=torch.ones_like(ids);positions=torch.linspace(0,3711,16).long()
  logs={}; reference=None
  for dtype in (torch.float32,torch.bfloat16):
   start=time.monotonic();model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,torch_dtype=dtype,attn_implementation='eager').eval().requires_grad_(False)
   assert next(model.parameters()).device.type=='cpu' and next(model.parameters()).dtype==dtype
   assert all(not p.requires_grad and p.device.type=='cpu' for p in model.parameters())
   assert not torch.cuda.is_initialized()
   load=time.monotonic()-start
   with torch.no_grad():
    model.base_model(input_ids=ids[:,:128],attention_mask=mask[:,:128],use_cache=False)
    cpu=time.process_time();wall=time.monotonic()
    hidden=model.base_model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    hidden_wall=time.monotonic()-wall;hidden_cpu=time.process_time()-cpu
    features=hidden.index_select(1,positions);del hidden
    cpu=time.process_time();wall=time.monotonic()
    for _ in range(16):logits=model.get_output_embeddings()(features)
    head_wall=(time.monotonic()-wall)/16;head_cpu=(time.process_time()-cpu)/16
    logp=F.log_softmax(logits.float(),-1);assert torch.isfinite(logp).all()
    data=dict(dtype=str(dtype),load_wall_seconds=load,hidden_3712_wall_seconds=hidden_wall,hidden_3712_cpu_seconds=hidden_cpu,head_chunk16_wall_seconds=head_wall,head_chunk16_cpu_seconds=head_cpu,trainable_parameters=0,process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if reference is None:reference=logp.detach().clone()
    else:
     p=reference.exp();q=logp.exp();kl=(p*(reference-logp)).sum(-1)
     data['vs_fp32']=dict(probability_max_abs_delta=float((p-q).abs().max()),probability_mean_abs_delta=float((p-q).abs().mean()),forward_kl_fp32_to_bf16_mean=float(kl.mean()),forward_kl_fp32_to_bf16_max=float(kl.max()),top1_match_fraction=float((p.argmax(-1)==q.argmax(-1)).float().mean()))
    assert all(not p.requires_grad and p.device.type=='cpu' for p in model.parameters())
    assert not torch.cuda.is_initialized()
    logs[str(dtype)]=data
    del model,logits,logp,features;gc.collect()
  return dict(job_id='2875',pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),cgroup=Path('/proc/self/cgroup').read_text(),synthetic_context=True,input_tokens=3712,sampled_token_positions=16,timing_repetitions={'hidden':1,'head':16},rss_scope='process lifetime HWM including earlier dtype stages; not separate per-dtype peaks',scope='read-only frozen CPU precision diagnostic under concurrent production load; not a protocol change, throughput benchmark or OPSD loss-error measurement',results=logs)
probe=None
try:
 probe=Probe.options(name='opsd-cpu-precision-probe-'+str(time.time_ns())).remote()
 result=ray.get(probe.run.remote(),timeout=300)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
finally:
 try:
  if probe is not None:ray.kill(probe,no_restart=True)
 finally:
  ray.shutdown()
