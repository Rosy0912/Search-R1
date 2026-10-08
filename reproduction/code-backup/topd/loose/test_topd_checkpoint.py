"""Real tiny Qwen HF checkpoint load/forward, with no training or GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture
def tiny(tmp_path):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    config=Qwen2Config(vocab_size=3,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,bos_token_id=0,eos_token_id=1,tie_word_embeddings=True)
    model=Qwen2ForCausalLM(config)
    raw=Tokenizer(WordLevel({'<bos>':0,'<eos>':1,'<unk>':2},unk_token='<unk>'));raw.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,bos_token='<bos>',eos_token='<eos>',unk_token='<unk>')
    base,checkpoint=tmp_path/'base',tmp_path/'checkpoint'
    for root in (base,checkpoint):
        model.save_pretrained(root);tokenizer.save_pretrained(root)
    return model,tokenizer,base,checkpoint


def verify(base,checkpoint,tmp_path):
    output=tmp_path/'verification.json'
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('verify-topd-checkpoint.py')),'--base',str(base),'--checkpoint',str(checkpoint),'--output',str(output)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1'),capture_output=True,text=True)
    return result,output


def test_real_updated_checkpoint_cpu_hf_load_finite_forward(tiny,tmp_path):
    import torch
    model,tokenizer,base,checkpoint=tiny
    with torch.no_grad():
        model.model.embed_tokens.weight[2,0].add_(.1)
    model.save_pretrained(checkpoint);tokenizer.save_pretrained(checkpoint)
    result,output=verify(base,checkpoint,tmp_path)
    assert result.returncode==0,result.stderr
    record=json.loads(output.read_text())
    assert record['status']=='PASS' and record['semantic_weights']['all_finite']
    assert 'model.embed_tokens.weight' in record['semantic_weights']['changed_tensor_names']
    assert record['cpu_parameter_bytes']>0 and record['hf_files']
    assert 'independent search inference still required' in record['scope']


def test_packaging_only_no_update_rejected(tiny,tmp_path):
    _,_,base,checkpoint=tiny
    result,output=verify(base,checkpoint,tmp_path)
    assert result.returncode!=0 and 'numerically identical' in result.stderr and not output.exists()


def test_nonfinite_checkpoint_rejected(tiny,tmp_path):
    import torch
    model,tokenizer,base,checkpoint=tiny
    with torch.no_grad():
        model.model.embed_tokens.weight[2,0]=float('nan')
    model.save_pretrained(checkpoint);tokenizer.save_pretrained(checkpoint)
    result,output=verify(base,checkpoint,tmp_path)
    assert result.returncode!=0 and 'Nonfinite semantic tensor' in result.stderr and not output.exists()
