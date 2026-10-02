import json,tempfile,torch
from pathlib import Path
from transformers import Qwen2Config,Qwen2ForCausalLM
from safetensors.torch import load_file,save_file
from tokenizers import Tokenizer,models,normalizers,pre_tokenizers
m=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,tie_word_embeddings=True))
with tempfile.TemporaryDirectory() as d:
 m.save_pretrained(d);p=Path(d)/'model.safetensors';state=load_file(str(p))
 good,info=Qwen2ForCausalLM.from_pretrained(d,output_loading_info=True)
 initial={'saved_head': 'lm_head.weight' in state,'saved_embedding':'model.embed_tokens.weight' in state,'loaded_tied':good.lm_head.weight is good.model.embed_tokens.weight,'loading_info':info}
 state.pop('model.layers.0.self_attn.q_proj.weight');save_file(state,str(p),metadata={'format':'pt'})
 bad,badinfo=Qwen2ForCausalLM.from_pretrained(d,output_loading_info=True)
 finite=bool(torch.isfinite(bad(torch.tensor([[1,2,3]])).logits).all())
 vocab={'[UNK]':0,'Hello':1,'hello':2}
 a=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));a.pre_tokenizer=pre_tokenizers.Whitespace()
 b=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));b.pre_tokenizer=pre_tokenizers.Whitespace();b.normalizer=normalizers.Lowercase()
 print(json.dumps({'legitimate_tied_export':initial,'missing_real_parameter':{'loading_info':badinfo,'finite_forward':finite},'same_vocab_different_tokenization':{'same_vocab':a.get_vocab()==b.get_vocab(),'a_ids':a.encode('Hello').ids,'b_ids':b.encode('Hello').ids}},indent=2))
