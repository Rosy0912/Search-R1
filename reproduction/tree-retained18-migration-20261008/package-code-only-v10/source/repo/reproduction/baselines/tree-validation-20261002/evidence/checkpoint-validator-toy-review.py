from pathlib import Path
import tempfile,subprocess,sys,json,shutil
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from safetensors.torch import load_file,save_file
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2');validator=root/'work/check_real_checkpoint.py'
results=[]
with tempfile.TemporaryDirectory() as temp:
 task=Path(temp);base=task/'base';base.mkdir()
 validator=task/'checker.py';validator.write_text((root/'work/check_real_checkpoint.py').read_text())
 vocab={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'The':4,'capital':5,'of':6,'France':7,'is':8}
 backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',bos_token='<bos>',eos_token='<eos>',unk_token='<unk>')
 cfg=Qwen2Config(vocab_size=len(vocab),hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,tie_word_embeddings=False,bos_token_id=1,eos_token_id=2)
 torch.manual_seed(42);model=Qwen2ForCausalLM(cfg);model.save_pretrained(base,safe_serialization=True);tok.save_pretrained(base)
 for case in ('unchanged','updated','missing_checkpoint_weight'):
  cp=task/case;shutil.copytree(base,cp)
  if case!='unchanged':
   state=load_file(str(cp/'model.safetensors'));state['model.embed_tokens.weight']=state['model.embed_tokens.weight']+.01
   if case=='missing_checkpoint_weight':state.pop('model.layers.0.self_attn.q_proj.weight')
   save_file(state,str(cp/'model.safetensors'),metadata={'format':'pt'})
  output=task/(case+'.json')
  call=subprocess.run([sys.executable,str(validator),'--base',str(base),'--checkpoint',str(cp),'--output',str(output)],capture_output=True,text=True)
  report=json.loads(output.read_text()) if output.exists() else None
  results.append({'case':case,'returncode':call.returncode,'output_exists':output.exists(),'status':report['status'] if report else None,'stderr_tail':call.stderr[-1500:]})
 # Additional legitimate tied-embedding HF safe serialization.
 cfg.tie_word_embeddings=True
 tied_base=task/'tied-base';tied_cp=task/'tied-updated'
 tied_model=Qwen2ForCausalLM(cfg);tied_model.save_pretrained(tied_base,safe_serialization=True);tok.save_pretrained(tied_base)
 with torch.no_grad():tied_model.get_input_embeddings().weight.add_(.01)
 tied_model.save_pretrained(tied_cp,safe_serialization=True);tok.save_pretrained(tied_cp)
 output=task/'tied.json'
 call=subprocess.run([sys.executable,str(validator),'--base',str(tied_base),'--checkpoint',str(tied_cp),'--output',str(output)],capture_output=True,text=True)
 results.append({'case':'tied_updated','returncode':call.returncode,'output_exists':output.exists(),'stderr_tail':call.stderr[-1500:]})
print(json.dumps({'toy_only_not_production_checkpoint':True,'production_checker_source_copy_unmodified':True,'results':results},indent=2))

assert [v["returncode"]==0 for v in results]==[False,True,False,True]
