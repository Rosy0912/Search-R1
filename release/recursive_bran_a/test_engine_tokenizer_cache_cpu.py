"""Actual tokenizer-group check, no model/optimizer/GPU initialization."""
import argparse
from importlib.metadata import version
from transformers import AutoTokenizer
from verl.third_party.vllm.vllm_v_0_6_3.tokenizer import TokenizerGroup
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
original=AutoTokenizer.from_pretrained(a.model,local_files_only=True);oldtype=type(original)
group=TokenizerGroup(original,enable_lora=False,max_num_seqs=32,max_input_length=4096)
cached=group.tokenizer
assert original is not cached and type(original) is oldtype
assert cached.get_vocab()==original.get_vocab() and len(cached)==len(original)
assert set(cached.all_special_ids)==set(original.all_special_ids)
texts=['<search>capital of France</search>','<answer>Paris</answer>','<|im_end|>','中文搜索🙂',' whitespace\n and tabs\t']
for text in texts:
 assert original.encode(text,add_special_tokens=False)==cached.encode(text,add_special_tokens=False)
 ids=original.encode(text,add_special_tokens=False)
 for skip in (True,False):assert original.decode(ids,skip_special_tokens=skip)==cached.decode(ids,skip_special_tokens=skip)
print({'vllm_package_version':version('vllm'),'caller_original_type_unchanged':True,'engine_copy':True,'complete_vocabulary_equal':True,'encode_decode_equal':True})
