import ast,json,os,tempfile,torch
from pathlib import Path
from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer,models,pre_tokenizers
from safetensors.torch import load_file,save_file
SOURCE=Path(__file__).resolve().parents[5]/'runtime-20261003/checkpoint_reload.py'
tree=ast.parse(SOURCE.read_text());nodes=[]
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='weights' for t in node.targets):break
 nodes.append(node)
code=compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec')
results={}
with tempfile.TemporaryDirectory() as td:
 root=Path(td);base=root/'base';saved=root/'saved';out=root/'out';out.mkdir()
 model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,tie_word_embeddings=True))
 backend=Tokenizer(models.WordLevel({'[UNK]':0,'The':1,'capital':2,'of':3,'France':4,'is':5},unk_token='[UNK]'));backend.pre_tokenizer=pre_tokenizers.Whitespace()
 tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',pad_token='[UNK]',eos_token='[UNK]')
 model.save_pretrained(base);tokenizer.save_pretrained(base)
 for field in ('bos_token_id','eos_token_id','pad_token_id'):setattr(model.config,field,getattr(tokenizer,field))
 with torch.no_grad():model.model.layers[0].self_attn.q_proj.weight.add_(.125)
 model.save_pretrained(saved);tokenizer.save_pretrained(saved)
 os.environ.update(OPSD_TEACHER=str(base),OPSD_CHECKPOINT=str(saved),OPSD_EVIDENCE=str(out))
 def run(label,expected):
  try:
   ns={};exec(code,ns);ok=True;detail={'changes_keys':list(ns['changes']),'tensor_count':len(ns['saved_map'])}
  except Exception as e:ok=False;detail={'error':type(e).__name__,'message':str(e)}
  assert ok==expected,(label,detail);results[label]={'accepted':ok,**detail}
 run('valid_tied_changed',True)
 weight=saved/'model.safetensors';original=load_file(str(weight));state=dict(original);state.pop('model.layers.0.self_attn.q_proj.weight');save_file(state,str(weight),metadata={'format':'pt'});run('missing_tensor',False)
 state=dict(original);state['model.layers.0.self_attn.q_proj.weight']=state['model.layers.0.self_attn.q_proj.weight'][:1];save_file(state,str(weight),metadata={'format':'pt'});run('wrong_shape',False)
 state=dict(original);state['lm_head.weight']=state['model.embed_tokens.weight']+1;save_file(state,str(weight),metadata={'format':'pt'});run('unequal_tied_alias',False)
 state=dict(original);state['lm_head.weight']=state['model.embed_tokens.weight'].clone();save_file(state,str(weight),metadata={'format':'pt'});run('equal_tied_alias',True)
 save_file(original,str(weight),metadata={'format':'pt'})
 tok=saved/'tokenizer.json';initial=tok.read_text();j=json.loads(initial);j['normalizer']={'type':'Lowercase'};tok.write_text(json.dumps(j));run('same_vocab_changed_normalizer',False);tok.write_text(initial)
 cfg=saved/'config.json';initial=cfg.read_text();j=json.loads(initial);j['rms_norm_eps']=1.;cfg.write_text(json.dumps(j));run('changed_rms_epsilon',False);cfg.write_text(initial)
 j=json.loads(initial);j['use_sliding_window']=True;cfg.write_text(json.dumps(j));run('changed_sliding_window',False);cfg.write_text(initial)
 j=json.loads(initial);j['eos_token_id']=31;cfg.write_text(json.dumps(j));run('saved_token_id_mismatch',False);cfg.write_text(initial)
print(json.dumps(results,indent=2))
