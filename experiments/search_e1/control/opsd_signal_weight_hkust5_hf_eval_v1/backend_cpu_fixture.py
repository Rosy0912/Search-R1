"""Actual HF4.46/Qwen generation, tiny random CPU model; no downloads or GPU."""
import json,os,sys,unittest
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from dependencies import verify_core,import_core

class Tokenizer:
    eos_token_id=151645
    def encode(self,text,**kwargs):return {'short':[10,11],'long':[10,12,13,14]}[text]
    def decode(self,ids,**kwargs):return ''.join(str(int(x))+',' for x in ids if int(x) not in (151645,151643))

class CPUGenerate(unittest.TestCase):
    def test_true_hf_greedy_batch_left_padding_matches_single_and_restores_processors(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'')
        torch.manual_seed(42);torch.set_num_threads(2)
        cfg=Qwen2Config(vocab_size=151646,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32,eos_token_id=[151645,151643],pad_token_id=151645)
        model=Qwen2ForCausalLM(cfg).eval();model.generation_config.repetition_penalty=1.05
        _,batch=import_core(verify_core());tok=Tokenizer();texts=['short','long'];events=[]
        before=model._get_logits_processor
        result=batch(model,tok,texts,3,events.append)
        expected=[]
        with torch.no_grad():
            for text in texts:
                prompt=tok.encode(text);ids=torch.tensor([prompt]);rr=model.generate(ids,attention_mask=torch.ones_like(ids),
                    do_sample=False,max_new_tokens=3,pad_token_id=151645)
                expected.append(rr[0,len(prompt):].tolist())
        self.assertEqual([r['token_ids'] for r in result],expected)
        self.assertEqual(model._get_logits_processor,before)
        self.assertTrue(events[0]['padding_tokens_not_counted_as_repetition'])
        self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
