"""Fresh complete Qwen2 HF checkpoint inventory; no GPU/load-inference claim."""
import hashlib,json,struct
from pathlib import Path
from contracts import require,sha

def shapes(cfg):
 h=cfg['hidden_size'];v=cfg['vocab_size'];ff=cfg['intermediate_size'];heads=cfg['num_attention_heads'];kv=cfg['num_key_value_heads'];d=h//heads
 require(h%heads==0,'Qwen2 head dimension')
 result={'model.embed_tokens.weight':[v,h],'model.norm.weight':[h]}
 if not cfg.get('tie_word_embeddings',False):result['lm_head.weight']=[v,h]
 for layer in range(cfg['num_hidden_layers']):
  p='model.layers.'+str(layer)+'.'
  for name,width in [('q_proj',h),('k_proj',kv*d),('v_proj',kv*d)]:
   result[p+'self_attn.'+name+'.weight']=[width,h];result[p+'self_attn.'+name+'.bias']=[width]
  result[p+'self_attn.o_proj.weight']=[h,h]
  for name,shape in [('gate_proj',[ff,h]),('up_proj',[ff,h]),('down_proj',[h,ff])]:result[p+'mlp.'+name+'.weight']=shape
  for name in ('input_layernorm','post_attention_layernorm'):result[p+name+'.weight']=[h]
 return result

def complete_hf(directory,student_directory):
 directory=Path(directory);student_directory=Path(student_directory)
 cfg=json.loads((directory/'config.json').read_text());expected_cfg=json.loads((student_directory/'config.json').read_text())
 require(cfg['model_type']=='qwen2','Checkpoint must remain Qwen2 student')
 for key in ('vocab_size','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','num_hidden_layers','tie_word_embeddings'):
  require(cfg.get(key)==expected_cfg.get(key),'Checkpoint student semantic config mismatch '+key)
 expected=shapes(cfg);headers={};files={}
 for path in sorted(directory.iterdir()):
  require(path.is_file() and not path.is_symlink(),'Checkpoint nonregular/aliased member')
  files[path.name]=dict(size=path.stat().st_size,sha256=sha(path))
  if path.suffix!='.safetensors':continue
  with path.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];require(0<n<16*1024**2,'Invalid checkpoint header size');header=json.loads(f.read(n));start=8+n
  spans=[]
  for name,t in header.items():
   if name=='__metadata__':continue
   require(name not in headers and name in expected and t['shape']==expected[name],'Checkpoint duplicate/unknown/missing shape '+name)
   width={'F32':4,'BF16':2,'F16':2}.get(t['dtype']);require(width is not None,'Checkpoint unexpected parameter dtype')
   count=1
   for dim in t['shape']:count*=dim
   a,b=t['data_offsets'];require(0<=a<b and b-a==count*width and start+b<=path.stat().st_size,'Checkpoint tensor data bounds')
   spans.append((a,b));headers[name]=dict(t,file=path.name)
  spans.sort();require(spans and spans[0][0]==0 and all(a[1]==b[0] for a,b in zip(spans,spans[1:])) and start+spans[-1][1]==path.stat().st_size,'Checkpoint tensor offsets gap/overlap/trailing bytes')
 require(set(headers)==set(expected),'Incomplete HF checkpoint weights')
 index=directory/'model.safetensors.index.json'
 if index.exists():require(json.loads(index.read_text())['weight_map']=={k:v['file'] for k,v in headers.items()},'HF arbitrary shard index incomplete/mismatch')
 elif len({v['file'] for v in headers.values()})>1:raise RuntimeError('Multi-shard HF index required')
 for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):require(name in files and isinstance(json.loads((directory/name).read_text()),dict),'Complete tokenizer JSON metadata required')
 return dict(status='FULL_HF_BYTES_SHAPES_COMPLETE_UNREVIEWED_INFERENCE_PENDING',files=files,parameter_tensors=len(headers),shapes=expected,loaded_inference_verified=False)
