"""Explicit CPU prepare and unique held Slurm submit; no implicit retry."""
from pathlib import Path
import json,os,shlex,subprocess,time,uuid
from .contract import *
RUNTIME='/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/env/bin/python3'

def once(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def zero():
 rows=[r.split('|') for r in subprocess.check_output(['sacct','-n','-P','-X','-S','2026-10-18','-u','jmzhang','--format=JobIDRaw,JobName%100,State'],text=True).splitlines() if r.strip()]
 require(any(v[0]=='139135' for v in rows),'Missing scheduler positive control')
 live=[r.split('|') for r in subprocess.check_output(['squeue','-h','-u','jmzhang','-o','%i|%j|%T'],text=True).splitlines() if r.strip()]
 names=[job(i) for i in (1,2,3)]
 require(not any(v[1] in names for v in rows+live),'An exact slot is already registered; inspect, never replay')
 return dict(at=time.time(),names=names,sacct_exact=0,squeue_exact=0,positive_control='139135')


def require_training_terminal(native):
 rows=[v.split('|') for v in subprocess.check_output(['sacct','-n','-P','-X','-j',native['job_id'],'--format=JobIDRaw,User,JobName%100,State,ExitCode,AllocTRES%250'],text=True).splitlines() if v.strip()]
 require(len(rows)==1 and rows[0][:5]==[native['job_id'],'jmzhang',native['job'],'COMPLETED','0:0'],'Actual final training terminal differs')
 parent=Path(BR)/'runs/7b-150-v1';sub=json.loads((parent/'submission.json').read_text())
 require(sub['job_id']==native['job_id'] and sub['plan_sha256']==native['plan_sha256'] and sub['source']==TRAIN_SOURCE,'Final training submission differs')
 require((parent/'exit.code').read_text().strip()=='0' and json.loads((parent/'training-returned.json').read_text())['actual_updates']==150,'Training did not return150')
 require(sha(sub['plan'])==native['plan_sha256'],'Final training plan changed')
 return rows[0]

def verify_source(deployment):
 source=Path(deployment['source']);require(source==Path(__file__).resolve().parents[2],'Deployed source path differs')
 for name,h in deployment['source_files'].items():require(sha(source/name)==h,'Source differs: '+name)
 for name,h in SCIENCE_FILES.items():require(deployment['source_files'][name]==h,'Scientific source differs')
 return source

def reload_model(model,shapes):
 import torch
 from transformers import AutoModelForCausalLM
 from safetensors import safe_open
 loaded=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map='cpu')
 require(set(loaded.state_dict())==set(shapes),'Full HF reload coverage differs')
 mapping={}
 for p in Path(model).glob('*.safetensors'):
  with safe_open(p,framework='pt',device='cpu') as f:
   for key in f.keys():require(key not in mapping,'Duplicate HF parameter');mapping[key]=p
 require(set(mapping)<=set(shapes),'Unexpected HF parameter')
 for name,value in loaded.state_dict().items():
  physical=name
  require(name in mapping,'Missing HF parameter')
  with safe_open(mapping[physical],framework='pt',device='cpu') as f:expected=f.get_tensor(physical)
  require(value.device.type=='cpu' and value.dtype==torch.bfloat16 and list(value.shape)==shapes[name] and torch.isfinite(value).all().item() and torch.equal(value,expected),'Actual HF reload tensor differs: '+name)
 require(not loaded.config.tie_word_embeddings and loaded.lm_head.weight.data_ptr()!=loaded.model.embed_tokens.weight.data_ptr(),'Untied storage differs')
 del loaded;require(not torch.cuda.is_initialized(),'CPU reload initialized CUDA')
 return dict(logical_parameters=len(shapes),all_tensors_finite_bf16_and_exact_safetensors=True,tied_pointer_shared=False,no_cuda_initialized=True)

def prepare(deployment,bundle_path):
 import torch,shutil
 import pyarrow.parquet as pq
 from transformers import AutoConfig,AutoTokenizer,GenerationConfig
 from vllm.engine.arg_utils import EngineArgs
 from reproduction.official_eval_v2.protocol import prompt,question_from_old_prompt
 from reproduction.official_eval_v2.driver import search
 from reproduction.official_eval_v2.prepare import retriever_endpoint
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU preparation only')
 source=verify_source(deployment);bundle=json.loads(Path(bundle_path).read_text())
 require(set(bundle['evidence'])=={'cold.json','native.json','hf.json','export-intent.json'},'Incomplete accepted evidence')
 values={}
 for name,row in bundle['evidence'].items():
  require(sha(row['path'])==row['sha256'],'Staged evidence SHA changed');values[name]=json.loads(Path(row['path']).read_text())
 native,hf=values['native.json'],values['hf.json'];validate_lineage(values['cold.json'],native,hf,values['export-intent.json']);require_training_terminal(native)
 require(bundle['local_all_raw_sha_verified'] is True,'Missing independent local original evidence gate')
 require(not Path(ROOT).exists(),'Existing preparation must be inspected')
 before=zero();Path(ROOT).mkdir(exist_ok=False)
 once(Path(ROOT)/'prepare-once.json',dict(at=time.time(),zero=before,bundle_sha256=sha(bundle_path),source=deployment['commit'],no_auto_retry=True))
 inputs=Path(ROOT)/'inputs';inputs.mkdir();evidence={}
 for name,row in bundle['evidence'].items():
  path=inputs/name;path.open('xb').write(Path(row['path']).read_bytes());require(sha(path)==row['sha256'],'Evidence copy differs');evidence[name]=dict(path=str(path),sha256=sha(path))
 for name,row in hf['files'].items():require(sha(Path(MODEL)/name)==row['sha256'],'Accepted HF changed')
 cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True)
 require((cfg.hidden_size,cfg.num_hidden_layers,cfg.vocab_size,cfg.tie_word_embeddings)==(3584,28,152064,False),'Not common7B architecture')
 reload=reload_model(MODEL,native['shapes'])
 # Immutable trained weights retain exact bytes. Old inference loads a
 # separate canonical tokenizer copy instead of modern jinja metadata.
 view=Path(VIEW);view.mkdir(exist_ok=False)
 for p in Path(MODEL).iterdir():
  if p.name.endswith('.safetensors') or p.name in ('model.safetensors.index.json','config.json','generation_config.json'):os.link(p,view/p.name)
 for name,h in CANONICAL_FILES.items():
  p=Path(CANONICAL)/name;require(sha(p)==h,'Canonical tokenizer changed');shutil.copyfile(p,view/name);require(sha(view/name)==h,'Canonical copy changed')
 files={p.name:sha(p) for p in view.iterdir()};tok=AutoTokenizer.from_pretrained(view,local_files_only=True);base=AutoTokenizer.from_pretrained(CANONICAL,local_files_only=True)
 modern=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 require(modern.get_vocab()==tok.get_vocab()==base.get_vocab() and tok.chat_template==base.chat_template and tok.special_tokens_map==base.special_tokens_map,'Vocabulary/common template differs')
 from search_r1.baselines.tree_raw_tokens import _encoder_binding
 require(_encoder_binding(modern)==_encoder_binding(tok)==_encoder_binding(base),'Encoder/decoder differs')
 require(tok.eos_token_id==151645 and tok.pad_token_id==151643,'EOS/pad differs')
 gen=GenerationConfig.from_pretrained(view,local_files_only=True);require(gen.eos_token_id==[151645,151643] and gen.pad_token_id==151643,'Generation semantics changed')
 original=Path(BASE)/'common-data/final.parquet';manifest=Path(BASE)/'common-data/manifest.json';require(sha(manifest)==DATA_MANIFEST,'Data manifest changed')
 dm=json.loads(manifest.read_text());require(sha(original)==dm['splits']['final']['sha256'],'Final parquet changed')
 rows=[]
 for i,row in enumerate(pq.read_table(original).to_pylist()):
  q=question_from_old_prompt(row['prompt']);gold=row['reward_model']['ground_truth']['target']
  rows.append(dict(final_index=i,question_id=row['extra_info']['index'],source=row['data_source'],gold_answers=[gold] if isinstance(gold,str) else list(gold),question=q,prompt=prompt(q)))
 validate_rows(rows);data=Path(ROOT)/'final.jsonl'
 with data.open('x') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 data_id=digest([{k:r[k] for k in ('final_index','question_id','source','gold_answers')} for r in rows])
 require(sha(data)==FINAL_JSONL and data_id==ID_GOLD,'Full51713 identity differs')
 max_len=max(len(tok.apply_chat_template([dict(role='user',content=r['prompt'])],add_generation_prompt=True,tokenize=True)) for r in rows);require(max_len<=2048,'Prompt capacity differs')
 args=EngineArgs(model=VIEW,tokenizer=VIEW,dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,gpu_memory_utilization=.65,max_model_len=4596,max_num_batched_tokens=8192,max_num_seqs=128,seed=0,disable_log_stats=True)
 require(args.seed==0 and args.tensor_parallel_size==1 and args.max_model_len==4596,'Engine configuration differs')
 receipt=Path(BASE)/'receipts/gpu-retriever-ready.json';endpoint=retriever_endpoint(json.loads(receipt.read_text())['url'])
 docs,body_sha=search(endpoint,['Beijing capital']);require(len(docs)==1 and len(docs[0])==3,'Protected retriever not ready')
 once(Path(ROOT)/'retriever-preflight.json',dict(at=time.time(),endpoint=endpoint,response_sha256=body_sha,documents=3))
 paths=[];(Path(ROOT)/'plans').mkdir()
 for i in (1,2,3):
  plan=dict(profile=PROFILE,science=SCIENCE,kind='trained-branpo',repeat=i,repeat_count=3,root=str(root(i)),job=job(i),source=str(source),source_commit=deployment['commit'],source_files=deployment['source_files'],accepted_model=MODEL,hf_files={n:r['sha256'] for n,r in hf['files'].items()},model=VIEW,model_files=files,model_revision=None,model_identity=identity(native,hf),training_job_id=native['job_id'],training_source=TRAIN_SOURCE,actual_training_iterations=150,final_directory_label=250,evidence=evidence,data=str(data),data_sha256=sha(data),id_gold_identity=data_id,original_data_sha256=sha(original),data_manifest_sha256=DATA_MANIFEST,retriever_receipt=str(receipt),retriever_receipt_sha256=sha(receipt),retriever_url=endpoint,gpus=8,cpus=64,memory_gib=600,no_auto_retry=True)
  root(i).mkdir();path=Path(ROOT)/'plans'/('r'+str(i)+'.json');once(path,plan);verify(plan);paths.append(str(path))
 require(not torch.cuda.is_initialized(),'CPU prepare initialized CUDA')
 proof=dict(at=time.time(),status='BRANPO7B_FINAL150_ACTUAL_HF_COMMONV2_51713_CPU_PREPARED',source=deployment['commit'],plans=paths,questions=51713,max_prompt_tokens=max_len,reload=reload,model_identity=identity(native,hf),model_files=files,evidence=evidence,no_gpu_action=True)
 once(Path(ROOT)/'prepared.json',proof);print('PREPARED='+json.dumps(proof),flush=True);return proof
def submit(index,deployment):
 source=verify_source(deployment);require(index in (1,2,3),'Only fixed three')
 path=Path(ROOT)/'plans'/('r'+str(index)+'.json');plan=json.loads(path.read_text());r=root(index)
 require(str(path) in json.loads((Path(ROOT)/'prepared.json').read_text())['plans'],'No prepared plan')
 # Exact name only: other fixed repeats may already have been uniquely submitted.
 rows=subprocess.check_output(['sacct','-n','-P','-X','-S','2026-10-18','--name='+job(index),'--format=JobIDRaw,JobName%100,State'],text=True)
 require(not rows.strip() and not subprocess.check_output(['squeue','-h','--name='+job(index)],text=True).strip(),'Slot already registered')
 require(not (r/'submission-intent.json').exists() and not (r/'submission.json').exists(),'Existing intent must be inspected')
 verify(plan)
 bound_native=json.loads(Path(plan['evidence']['native.json']['path']).read_text());require_training_terminal(bound_native)
 q=shlex.quote
 lines=['#!/bin/bash','set -euo pipefail','export PATH='+q(str(Path(RUNTIME).parent))+':"$PATH"','export PYTHONPATH='+q(str(source))+' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1','export SEARCH_R1_EXECUTION_BACKEND=slurm_lumia RECURSIVE_SOURCE_COMMIT='+q(deployment['commit']),'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false','export VLLM_ATTENTION_BACKEND=XFORMERS VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn','export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled','unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS','cd '+q(str(source)),'ulimit -n 65535','set +e',q(RUNTIME)+' -B -m reproduction.branpo7b_final_v2.launch --plan '+q(str(path)),'rc=$?','echo "$rc" > '+q(str(r/'exit.code')),'exit "$rc"']
 script='\n'.join(lines)+'\n'
 cmd=['sbatch','--parsable','--hold','--no-requeue','--account=lumia','--qos=normal','--job-name='+job(index),'--nodes=1','--ntasks=1','--gres=gpu:8','--cpus-per-task=64','--mem=600G','--time=2-00:00:00','--partition=ADA6000,L40S','--output='+str(r/'slurm-%j.log')]
 once(r/'submission-intent.json',dict(at=time.time(),job=plan['job'],plan_sha256=sha(path),command=cmd,script=script,no_auto_retry=True))
 result=subprocess.run(cmd,input=script,text=True,capture_output=True);once(r/'submission-response.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));result.check_returncode()
 jid=result.stdout.strip().split(';')[0];require(jid.isdigit(),'Invalid scheduler ID')
 sub=dict(at=time.time(),job_id=jid,job=plan['job'],root=str(r),output=str(r/'output'),plan=str(path),plan_sha256=sha(path),source=deployment['commit'],group='BranPO',size='7b',outer=150,repeat_index=index,repeat_count=3,checkpoint_identity=plan['model_identity'],gpus=8,no_auto_retry=True)
 once(r/'submission.json',sub)
 row=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True);require('JobName='+plan['job']+' ' in row and 'JobState=PENDING' in row and 'Reason=JobHeldUser' in row,'Unexpected held allocation')
 subprocess.check_call(['scontrol','release',jid]);print('SUBMISSION='+json.dumps(sub),flush=True);return sub
