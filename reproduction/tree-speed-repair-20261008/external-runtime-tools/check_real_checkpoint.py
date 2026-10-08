"""CPU-only full HF reload and immutable-base parameter delta acceptance."""
import argparse, hashlib, json, os, pathlib, resource, time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--checkpoint',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
assert not a.output.exists(), 'Refuse to overwrite acceptance evidence'
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM,AutoTokenizer
torch.set_num_threads(8)
start=time.perf_counter();cpu=time.process_time()
base_tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
assert base_tok.get_vocab()==tok.get_vocab(), 'Tokenizer vocab differs'
assert (base_tok.bos_token_id,base_tok.eos_token_id)==(tok.bos_token_id,tok.eos_token_id), 'BOS/EOS differs'
model,loading=AutoModelForCausalLM.from_pretrained(a.checkpoint,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True,output_loading_info=True)
assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']), 'Incomplete/incompatible HF weights: '+str(loading)
model.eval()
assert all(x.device.type=='cpu' for x in model.parameters())
index=a.base/'model.safetensors.index.json'
if index.exists():mapping=json.loads(index.read_text())['weight_map']
else:
 with safe_open(str(a.base/'model.safetensors'),framework='pt',device='cpu') as f:mapping={k:'model.safetensors' for k in f.keys()}
state=model.state_dict()
# HF may omit a tied alias in safetensors. Require every saved base key in
# the loaded model; omitted state keys must alias a saved tensor, not be new.
assert set(mapping)<=set(state), 'Checkpoint/base keys differ'
for key in set(state)-set(mapping):
 assert any(state[key].data_ptr()==state[other].data_ptr() for other in mapping), 'Unmatched non-tied key: '+key
changed=0;elements=0;max_delta=0.;changed_tensors=0
for file in sorted(set(mapping.values())):
 with safe_open(str(a.base/file),framework='pt',device='cpu') as f:
  for key in f.keys():
   before=f.get_tensor(key).float();after=state[key].detach()
   assert before.shape==after.shape and torch.isfinite(after).all(), 'Shape/nonfinite checkpoint: '+key
   delta=(after-before).abs();n=int(torch.count_nonzero(delta));changed+=n;elements+=after.numel();changed_tensors+=bool(n);max_delta=max(max_delta,float(delta.max()))
assert changed>0 and max_delta>0, 'No actual parameter change from immutable base'
inputs=tok('The capital of France is',return_tensors='pt')
with torch.no_grad():logits=model(**inputs).logits
assert logits.shape[-1]==model.config.vocab_size and torch.isfinite(logits).all(), 'Reloaded CPU forward invalid'
files={}
for f in sorted(a.checkpoint.rglob('*')):
 if not f.is_file():continue
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 files[str(f.relative_to(a.checkpoint))]=h.hexdigest()
report=dict(status='HF_RELOAD_PARAMETER_DELTA_ACCEPTED',base=str(a.base.resolve()),checkpoint=str(a.checkpoint.resolve()),hf_loading_info=loading,tokenizer_vocab_equal=True,bos_eos_equal=True,all_parameters_finite=True,changed_parameters=changed,total_parameters=elements,changed_tensors=changed_tensors,max_abs_parameter_delta=max_delta,cpu_forward_logits_finite=True,checkpoint_file_sha256=files,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,peak_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,exact_optimizer_rng_dataloader_resume=False)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as out:json.dump(report,out,indent=2,allow_nan=False)
print(json.dumps(report))
