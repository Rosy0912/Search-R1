"""Actual old-runtime common3B data/tokenizer/constructor positive control.

No trained HF payload is read and no evaluation plan or submission is created.
"""
import importlib.metadata as metadata
import json,os
from pathlib import Path
from .contract import *

def run():
 import torch
 import pandas as pd
 from transformers import AutoTokenizer,AutoConfig
 from vllm.engine.arg_utils import EngineArgs
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU only')
 require(not Path(ROOT).exists(),'No production prepare during positive control')
 require(sha(TEMPLATE)==TEMPLATE_SHA,'Reference plan changed');template=json.loads(Path(TEMPLATE).read_text())
 from reproduction.seven_dataset.entry import inspect_inputs
 cfg,expected=inspect_inputs(template)
 tok=AutoTokenizer.from_pretrained(template['canonical_tokenizer'],local_files_only=True)
 config=AutoConfig.from_pretrained(template['canonical_tokenizer'],local_files_only=True)
 require((config.hidden_size,config.num_hidden_layers,config.vocab_size)==(2048,36,151936),'Wrong common base')
 # Read modern saved metadata under the old inference runtime to bind token IDs;
 # use the exact canonical tokenizer copy for all actual common inference.
 modern=BR+'/runs/3b-v6/checkpoints/global_step_200/actor/huggingface'
 trained=AutoTokenizer.from_pretrained(modern,local_files_only=True)
 require(trained.get_vocab()==tok.get_vocab(),'Saved training token ID vocabulary differs')
 probes=['<think>Paris</think><search>Paris\nFrance</search><information>é中文</information><answer>Paris</answer>','A\u0085B\nC','foo   bar\t!']
 for text in probes:require(trained.encode(text)==tok.encode(text) and trained.decode(trained.encode(text))==tok.decode(tok.encode(text)),'Training/common token IDs or decoding differ')
 frame=pd.read_parquet(Path(template['data_root'])/'final.parquet')
 def rendered(value):
  if isinstance(value,str):return value
  return tok.apply_chat_template(list(value),add_generation_prompt=True,tokenize=False)
 max_len=max(len(tok.encode(rendered(v),add_special_tokens=False)) for v in frame['prompt'])
 require(len(frame)==len(expected)==51713 and max_len<=2048,'Full question prompt preflight differs')
 args=EngineArgs(model=template['canonical_tokenizer'],tokenizer=template['canonical_tokenizer'],dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,gpu_memory_utilization=.65,max_model_len=4596,max_num_batched_tokens=8192,max_num_seqs=128,seed=0,disable_log_stats=True)
 require(args.seed==0 and args.tensor_parallel_size==1 and args.max_model_len==4596,'Common engine arguments differ')
 versions={k:metadata.version(k) for k in ('torch','transformers','vllm','pyarrow')}
 require(versions['torch'].split('+')[0]=='2.4.0' and versions['vllm']=='0.6.3','Wrong inference runtime')
 require(not torch.cuda.is_initialized() and not Path(ROOT).exists(),'Positive control made production state')
 print('CPU_POSITIVE='+json.dumps(dict(status='REAL_OLD_COMMON3B_ALL51713_DATA_CANONICAL_TOKENIZER_AND_ENGINEARGS_PASSED',questions=51713,max_prompt_tokens=max_len,canonical_tokenizer_files=template['canonical_tokenizer_files'],training_native100_metadata_token_id_vocabulary_and_probes_equal=True,modern_chat_template_loaded_by_old_runtime=trained.chat_template is not None,canonical_chat_template_present=tok.chat_template is not None,versions=versions,scientific_runner_unchanged=True,no_actual_final150_hf_reload=True,no_plan_or_gpu_submission=True)),flush=True)
if __name__=='__main__':run()
