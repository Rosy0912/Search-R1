"""Real full input/runtime/constructor controls, without final weights or CUDA."""
from pathlib import Path
import ast,hashlib,importlib.metadata,json,os,subprocess
from .scope import BASE,all_slots,validate_slot,DATA_SHA,ID_GOLD,digest
from .contract import CANONICAL,CANONICAL_FILES,COMMON,DATA_MANIFEST
from reproduction.official_eval_v2.protocol import prompt,question_from_old_prompt,validate_rows

def run():
 import torch
 import pyarrow.parquet as pq
 from transformers import AutoTokenizer,AutoConfig
 from vllm.engine.arg_utils import EngineArgs
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 versions={n:importlib.metadata.version(n) for n in ['torch','transformers','vllm']}
 assert versions['torch'].split('+')[0]=='2.4.0' and versions['transformers']=='4.46.3' and versions['vllm']=='0.6.3'
 plan=json.loads((BASE/'plans/fresh-budget-5.json').read_text())
 for n,h in plan['source_files'].items():assert hashlib.sha256((Path(plan['source'])/n).read_bytes()).hexdigest()==h,n
 common=Path(COMMON)/'common-data';manifest=common/'manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==DATA_MANIFEST
 dm=json.loads(manifest.read_text());final=common/'final.parquet';assert hashlib.sha256(final.read_bytes()).hexdigest()==dm['splits']['final']['sha256']
 rows=[];raw=hashlib.sha256()
 for i,r in enumerate(pq.read_table(final).to_pylist()):
  question=question_from_old_prompt(r['prompt']);gold=r['reward_model']['ground_truth']['target'];row=dict(final_index=i,question_id=r['extra_info']['index'],source=r['data_source'],gold_answers=[gold] if isinstance(gold,str) else list(gold),question=question,prompt=prompt(question));rows.append(row);raw.update((json.dumps(row,ensure_ascii=False)+'\n').encode())
 validate_rows(rows);assert raw.hexdigest()==DATA_SHA and digest([{k:r[k] for k in ('final_index','question_id','source','gold_answers')} for r in rows])==ID_GOLD
 for n,h in CANONICAL_FILES.items():assert hashlib.sha256((Path(CANONICAL)/n).read_bytes()).hexdigest()==h,n
 tok=AutoTokenizer.from_pretrained(CANONICAL,local_files_only=True);cfg=AutoConfig.from_pretrained(CANONICAL,local_files_only=True)
 assert tok.eos_token_id==151645 and tok.pad_token_id==151643 and cfg.tie_word_embeddings is False
 maximum=max(len(tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True)) for r in rows);assert maximum==311
 args=EngineArgs(model=CANONICAL,tokenizer=CANONICAL,dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,gpu_memory_utilization=.65,max_model_len=4596,max_num_batched_tokens=8192,max_num_seqs=128,seed=0,disable_log_stats=True)
 assert args.seed==0 and args.tensor_parallel_size==1 and args.max_model_len==4596 and args.max_num_seqs==128
 slots=all_slots();assert len(slots)==23
 for row in slots:validate_slot(row)
 names=[v['job_name'] for v in slots]
 actual=subprocess.check_output(['sacct','-X','-n','-P','-S','2026-10-20T00:00:00','--name='+','.join(names),'--format=JobIDRaw,JobName%100,State'],text=True);assert not actual.strip()
 assert not subprocess.check_output(['squeue','-h','--name='+','.join(names)],text=True).strip()
 assert subprocess.check_output(['sacct','-X','-n','-P','-j','139584','--format=JobIDRaw,State'],text=True).strip()
 assert not(BASE/'evaluations/ablation-commonv2-v1').exists()
 assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
 print('PREFLIGHT='+json.dumps({'rows':51713,'max_prompt_tokens':maximum,'data_sha256':DATA_SHA,'id_gold_identity':ID_GOLD,'versions':versions,'old_scientific_training_files':len(plan['source_files']),'canonical_tokenizer_verified':True,'actual_EngineArgs_constructed_not_LLM':True,'new23_exact_scheduler_names_zero':True,'new_evaluation_ROOT_absent':True,'no_final_model_weights_or_native_payload_read':True,'no_GPU_action':True}),flush=True)
if __name__=='__main__':run()
