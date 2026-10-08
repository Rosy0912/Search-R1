"""HF complete safe-serialization checkpoints: no fixed shard count or tokenizer omission."""
import hashlib,json,pathlib,struct

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def expected_shapes(config):
 architecture=(config['num_hidden_layers'],config['hidden_size'],config['vocab_size'],config['num_attention_heads'],config['num_key_value_heads'],config['intermediate_size'],config.get('tie_word_embeddings',False))
 if architecture==(64,5120,152064,40,8,27648,False):layers,hidden,vocab,kv,mlp=64,5120,152064,1024,27648
 elif architecture==(36,2048,151936,16,2,11008,True):layers,hidden,vocab,kv,mlp=36,2048,151936,256,11008
 else:raise ValueError('Only exact official32B Base or3B-Instruct architecture')
 shapes={'model.embed_tokens.weight':[vocab,hidden],'model.norm.weight':[hidden]}
 if not config.get('tie_word_embeddings',False):shapes['lm_head.weight']=[vocab,hidden]
 for layer in range(layers):
  prefix=f'model.layers.{layer}.'
  for key,shape in {'self_attn.q_proj.weight':[hidden,hidden],'self_attn.q_proj.bias':[hidden],'self_attn.k_proj.weight':[kv,hidden],'self_attn.k_proj.bias':[kv],'self_attn.v_proj.weight':[kv,hidden],'self_attn.v_proj.bias':[kv],'self_attn.o_proj.weight':[hidden,hidden],'mlp.gate_proj.weight':[mlp,hidden],'mlp.up_proj.weight':[mlp,hidden],'mlp.down_proj.weight':[hidden,mlp],'input_layernorm.weight':[hidden],'post_attention_layernorm.weight':[hidden]}.items():shapes[prefix+key]=shape
 return shapes

def inventory(directory,expected_role="teacher32b"):
 root=pathlib.Path(directory).resolve();files={}
 for p in root.rglob('*'):
  if p.is_symlink():raise ValueError('HF alias/symlink rejected')
  if p.is_file():files[str(p.relative_to(root))]={'size':p.stat().st_size,'sha256':sha(p)}
 config=json.loads((root/'config.json').read_text())
 if config['model_type']!='qwen2':raise ValueError('Only official Qwen2 architecture')
 if expected_role=='teacher32b' and (config['num_hidden_layers'],config['hidden_size'],config['vocab_size'])!=(64,5120,152064):raise ValueError('Teacher must remain exact32B')
 if expected_role=='student3b' and (config['num_hidden_layers'],config['hidden_size'],config['vocab_size'])!=(36,2048,151936):raise ValueError('Student must remain exact3B-Instruct')
 if expected_role not in ('teacher32b','student3b'):raise ValueError('Unknown checkpoint role')
 exact_shapes=expected_shapes(config)
 index=root/'model.safetensors.index.json'
 if index.exists():expected=json.loads(index.read_text())['weight_map'];shards=sorted(set(expected.values()))
 else:expected=None;shards=['model.safetensors']
 keys={};schema={}
 for name in shards:
  if pathlib.PurePath(name).name!=name or name not in files:raise ValueError('missing/unsafe HF shard')
  with open(root/name,'rb') as f:
   count=struct.unpack('<Q',f.read(8))[0]
   if count>128*1024**2:raise ValueError('invalid safetensor header')
   header=json.loads(f.read(count))
  data_size=files[name]['size']-8-count
  for key,record in header.items():
   if key=='__metadata__':continue
   if key in keys or record['data_offsets'][0]<0 or record['data_offsets'][1]>data_size:raise ValueError('duplicate/incomplete tensor shard')
   keys[key]=name
   import math
   if record['dtype'] not in ('BF16','F16','F32') or record['data_offsets'][1]-record['data_offsets'][0]!=math.prod(record['shape'])*({'BF16':2,'F16':2,'F32':4}[record['dtype']]):raise ValueError('tensor dtype/size differs from header')
   schema[key]=record['shape']
 if schema!=exact_shapes:raise ValueError('incomplete exact32B tensor keys/shapes')
 if expected is not None and keys!=expected:raise ValueError('HF index does not match complete actual tensor headers')
 if 'model.embed_tokens.weight' not in keys or (not config.get('tie_word_embeddings',False) and 'lm_head.weight' not in keys) or any(not any(k.startswith(f'model.layers.{layer}.') for k in keys) for layer in range(config['num_hidden_layers'])):raise ValueError('incomplete exact official model layer/head set')
 required=('tokenizer.json','tokenizer_config.json')
 if any(name not in files for name in required):raise ValueError('missing tokenizer metadata')
 return {'directory':str(root),'files':files,'shards':shards,'tensor_count':len(keys),'tensor_schema_sha256':hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest(),'all_json':sorted(name for name in files if name.endswith('.json')),'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}

def tokenizer_compatible(model,base):
 from transformers import AutoTokenizer
 a=AutoTokenizer.from_pretrained(model,local_files_only=True);b=AutoTokenizer.from_pretrained(base,local_files_only=True)
 if a.get_vocab()!=b.get_vocab():raise ValueError('native complete tokenizer vocabulary changed')
 if (a.bos_token_id,a.eos_token_id,a.pad_token_id)!=(b.bos_token_id,b.eos_token_id,b.pad_token_id):raise ValueError('BOS/EOS/PAD identity drift')
 aj=json.loads(a.backend_tokenizer.to_str());bj=json.loads(b.backend_tokenizer.to_str())
 if aj['model']!=bj['model'] or aj.get('added_tokens')!=bj.get('added_tokens'):raise ValueError('tokenizer merges/added token behavior drift')
 texts=['Question: A?\n<search>q</search>','中文 空格\n\t<answer>x</answer>','<|endoftext|>','a  b\n\n','é １２３🙂']
 ids=[a.encode(t,add_special_tokens=False) for t in texts]
 if ids!=[b.encode(t,add_special_tokens=False) for t in texts]:raise ValueError('native tokenizer encoding drift')
 return {'vocab_size':len(a.get_vocab()),'bos':a.bos_token_id,'eos':a.eos_token_id,'pad':a.pad_token_id,'native_encode_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
