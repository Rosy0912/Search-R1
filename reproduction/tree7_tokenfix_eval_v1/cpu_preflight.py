"""Real common-data/tokenizer/runtime positive control, using the fixed base metadata.

No trained-model weights are loaded here. This cannot stand in for final HF
reload, native acceptance, prepared.json, or a scientific evaluation.
"""
import ast
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
from reproduction.official_eval_v2.protocol import prompt,question_from_old_prompt,validate_rows,digest
from .contract import BASE,COMMON,DATA_MANIFEST,FINAL_JSONL,ID_GOLD,sha,PROFILE,SCIENCE,INITIAL_MODEL


def run():
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoConfig,AutoTokenizer
    from vllm.engine.arg_utils import EngineArgs
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    versions={n:metadata.version(n) for n in ['torch','vllm','transformers','pyarrow']}
    assert versions['vllm']=='0.6.3' and versions['torch'].split('+')[0]=='2.4.0'
    path=Path(importlib.util.find_spec('vllm').origin).parent/'platforms/cuda.py'
    definition=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='device_id_to_physical_device_id')
    scope={'os':os};exec(compile(ast.Module(body=[definition],type_ignores=[]),str(path),'exec'),scope)
    try:
        for index in range(8):
            os.environ['CUDA_VISIBLE_DEVICES']=str(index)
            assert scope['device_id_to_physical_device_id'](0)==index
    finally:os.environ['CUDA_VISIBLE_DEVICES']=''
    manifest=COMMON/'common-data/manifest.json';assert sha(manifest)==DATA_MANIFEST
    dm=json.loads(manifest.read_text());original=COMMON/'common-data/final.parquet'
    assert sha(original)==dm['splits']['final']['sha256']
    rows=[];raw_digest=__import__('hashlib').sha256()
    for i,r in enumerate(pq.read_table(original).to_pylist()):
        question=question_from_old_prompt(r['prompt']);gold=r['reward_model']['ground_truth']['target']
        row=dict(final_index=i,question_id=r['extra_info']['index'],source=r['data_source'],
                 gold_answers=[gold] if isinstance(gold,str) else list(gold),question=question,prompt=prompt(question))
        rows.append(row);raw_digest.update((json.dumps(row,ensure_ascii=False)+'\n').encode())
    validate_rows(rows)
    identity=digest([{k:r[k] for k in ('final_index','question_id','source','gold_answers')} for r in rows])
    assert raw_digest.hexdigest()==FINAL_JSONL and identity==ID_GOLD
    model=INITIAL_MODEL
    cfg=AutoConfig.from_pretrained(model,local_files_only=True)
    assert (cfg.hidden_size,cfg.num_hidden_layers,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.tie_word_embeddings)==(3584,28,28,4,False)
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    assert tok.eos_token_id==151645 and tok.pad_token_id==151643
    max_len=max(len(tok.apply_chat_template([dict(role='user',content=r['prompt'])],add_generation_prompt=True,tokenize=True)) for r in rows)
    assert max_len<=2048
    args=EngineArgs(model=str(model),tokenizer=str(model),dtype='bfloat16',tensor_parallel_size=1,
                   enforce_eager=True,gpu_memory_utilization=.65,max_model_len=4596,max_num_batched_tokens=8192,
                   max_num_seqs=128,seed=0,disable_log_stats=True)
    assert args.seed==0 and args.max_model_len==4596 and args.max_num_seqs==128
    assert not torch.cuda.is_initialized()
    print('CPU_POSITIVE_CONTROL='+json.dumps(dict(status='REAL_BASE_METADATA_AND_ALL51713_COMMONV2_PROMPTS_CPU_PASSED',
        rows=len(rows),max_prompt_tokens=max_len,profile=PROFILE,science=SCIENCE,versions=versions,
        data_sha256=raw_digest.hexdigest(),id_gold_identity=identity,cuda_initialized=False,
        no_model_or_optimizer_payload_read=True,not_final_trained_hf_acceptance=True,no_gpu_action=True)),flush=True)


if __name__=='__main__':run()
