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

def verify_source(deployment):
 source=Path(deployment['source']);require(source==Path(__file__).resolve().parents[2],'Deployed source path differs')
 for name,h in deployment['source_files'].items():require(sha(source/name)==h,'Source differs: '+name)
 for name,h in SCIENCE.items():require(deployment['source_files'][name]==h,'Scientific source differs')
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
  physical=name if name in mapping else 'model.embed_tokens.weight'
  require(name in mapping or name=='lm_head.weight','Missing HF parameter')
  with safe_open(mapping[physical],framework='pt',device='cpu') as f:expected=f.get_tensor(physical)
  require(value.device.type=='cpu' and value.dtype==torch.bfloat16 and list(value.shape)==shapes[name] and torch.isfinite(value).all().item() and torch.equal(value,expected),'Actual HF reload tensor differs: '+name)
 require(loaded.config.tie_word_embeddings and loaded.lm_head.weight.data_ptr()==loaded.model.embed_tokens.weight.data_ptr(),'Tied storage differs')
 del loaded;require(not torch.cuda.is_initialized(),'CPU reload initialized CUDA')
 return dict(logical_parameters=len(shapes),all_tensors_finite_bf16_and_exact_safetensors=True,tied_pointer_shared=True,no_cuda_initialized=True)

def prepare(deployment,bundle_path):
 import torch
 from transformers import AutoConfig,AutoTokenizer,GenerationConfig
 from vllm.engine.arg_utils import EngineArgs
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU preparation only')
 source=verify_source(deployment);bundle=json.loads(Path(bundle_path).read_text())
 require(set(bundle['evidence'])=={'cold.json','native.json','hf.json','export-intent.json'},'Incomplete accepted evidence')
 values={}
 for name,row in bundle['evidence'].items():
  require(sha(row['path'])==row['sha256'],'Staged evidence SHA changed');values[name]=json.loads(Path(row['path']).read_text())
 native,hf=values['native.json'],values['hf.json'];validate_lineage(values['cold.json'],native,hf,values['export-intent.json'])
 require(bundle['local_all_raw_sha_verified'] is True,'Missing independent local original evidence gate')
 require(not Path(ROOT).exists(),'Existing preparation must be inspected')
 before=zero();Path(ROOT).mkdir(exist_ok=False)
 once(Path(ROOT)/'prepare-once.json',dict(at=time.time(),zero=before,bundle_sha256=sha(bundle_path),source=deployment['commit'],no_auto_retry=True))
 inputs=Path(ROOT)/'inputs';inputs.mkdir();evidence={}
 for name,row in bundle['evidence'].items():
  path=inputs/name;path.open('xb').write(Path(row['path']).read_bytes());require(sha(path)==row['sha256'],'Evidence copy differs');evidence[name]=dict(path=str(path),sha256=sha(path))
 require(sha(TEMPLATE)==TEMPLATE_SHA,'Template plan changed');template=json.loads(Path(TEMPLATE).read_text())
 for name,row in hf['files'].items():require(sha(Path(MODEL)/name)==row['sha256'],'HF identity changed')
 cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True)
 require((cfg.hidden_size,cfg.num_hidden_layers,cfg.vocab_size,cfg.tie_word_embeddings)==(2048,36,151936,True),'Not common3B architecture')
 reload=reload_model(MODEL,native['shapes'])
 inherited={k:template[k] for k in INHERITED};plan=dict(inherited,protocol='branpo3b_final150_common_v1',group='BranPO',outer=150,model=MODEL,model_files={n:v['sha256'] for n,v in hf['files'].items()},training_job_id=native['job_id'],training_source=TRAIN_SOURCE,checkpoint_identity=identity(native,hf),template_plan=TEMPLATE,template_plan_sha256=TEMPLATE_SHA,protocol_template_config_sha256=CONFIG_SHA,evidence=evidence,inference_model=ROOT+'/model-view',evaluation_code_commit=deployment['commit'],evaluation_run_id=uuid.uuid4().hex,job=job(1),output=str(root(1)/'output'),repeat_index=1,repeat_count=3,no_auto_retry=True,allocation=dict(gpus=8,cpus=64,memory_gib=600))
 from reproduction.seven_dataset.model_view import create_view
 from reproduction.seven_dataset.base_contracts import FILES
 create_view(plan,FILES['tokenizer.json'])
 # The modern training tokenizer's complete vocabulary/template was checked in
 # its native runtime. The old inference runtime consumes the exact canonical
 # common tokenizer copies, preserving the original scoring/token protocol.
 from reproduction.seven_dataset.entry import inspect_inputs
 reference,expected=inspect_inputs(template)
 tok=AutoTokenizer.from_pretrained(plan['inference_model'],local_files_only=True)
 common=AutoTokenizer.from_pretrained(plan['canonical_tokenizer'],local_files_only=True)
 require(tok.get_vocab()==common.get_vocab() and tok.chat_template==common.chat_template,'Canonical tokenizer changed')
 from search_r1.baselines.tree_raw_tokens import _encoder_binding
 require(_encoder_binding(tok)==_encoder_binding(common),'Old runtime encoder semantics changed')
 import pandas as pd
 frame=pd.read_parquet(Path(plan['data_root'])/'final.parquet')
 def text(prompt):
  if isinstance(prompt,str):return prompt
  return tok.apply_chat_template(list(prompt),add_generation_prompt=True,tokenize=False)
 max_tokens=max(len(tok.encode(text(v),add_special_tokens=False)) for v in frame['prompt'])
 require(max_tokens<=2048 and len(frame)==len(expected)==51713,'Full prompt preflight differs')
 gen=GenerationConfig.from_pretrained(plan['inference_model'],local_files_only=True)
 require(gen.eos_token_id==[151645,151643] and gen.pad_token_id==151643,'Generation semantics changed')
 args=EngineArgs(model=plan['inference_model'],tokenizer=plan['inference_model'],dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,gpu_memory_utilization=.65,seed=0)
 require(args.seed==0 and args.tensor_parallel_size==1,'Engine defaults differ')
 # Reuse the unchanged entry validation and runtime; each inference plan gets
 # its own execution ID and root, while sharing one accepted frozen HF model.
 from .entry import inspect
 paths=[];(Path(ROOT)/'plans').mkdir()
 for i in (1,2,3):
  v=dict(plan,repeat_index=i,job=job(i),output=str(root(i)/'output'),evaluation_run_id=uuid.uuid4().hex)
  root(i).mkdir();path=Path(ROOT)/'plans'/('r'+str(i)+'.json');once(path,v)
  inspect(v);paths.append(str(path))
 proof=dict(at=time.time(),status='BRANPO3B_FINAL150_ACTUAL_HF_COMMON51713_CPU_PREPARED',source=deployment['commit'],plans=paths,questions=51713,max_prompt_tokens=max_tokens,reload=reload,checkpoint_identity=plan['checkpoint_identity'],model_files=plan['model_files'],evidence=evidence,no_gpu_action=True)
 once(Path(ROOT)/'prepared.json',proof);print('PREPARED='+json.dumps(proof),flush=True)
 return proof

def submit(index,deployment):
 source=verify_source(deployment);require(index in (1,2,3),'Only fixed three')
 path=Path(ROOT)/'plans'/('r'+str(index)+'.json');plan=json.loads(path.read_text());r=root(index)
 require(str(path) in json.loads((Path(ROOT)/'prepared.json').read_text())['plans'],'No prepared plan')
 # Exact name only: other fixed repeats may already have been uniquely submitted.
 rows=subprocess.check_output(['sacct','-n','-P','-X','-S','2026-10-18','--name='+job(index),'--format=JobIDRaw,JobName%100,State'],text=True)
 require(not rows.strip() and not subprocess.check_output(['squeue','-h','--name='+job(index)],text=True).strip(),'Slot already registered')
 require(not (r/'submission-intent.json').exists() and not (r/'submission.json').exists(),'Existing intent must be inspected')
 from .entry import inspect
 inspect(plan)
 q=shlex.quote
 lines=['#!/bin/bash','set -euo pipefail','export PATH='+q(str(Path(RUNTIME).parent))+':"$PATH"','export PYTHONPATH='+q(str(source))+' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1','export SEARCH_R1_EXECUTION_BACKEND=slurm_lumia RECURSIVE_SOURCE_COMMIT='+q(deployment['commit']),'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false','export VLLM_ATTENTION_BACKEND=XFORMERS VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn','export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled','unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS','cd '+q(str(source)),'ulimit -n 65535','set +e',q(RUNTIME)+' -B -m reproduction.branpo3b_final.launch --plan '+q(str(path)),'rc=$?','echo "$rc" > '+q(str(r/'exit.code')),'exit "$rc"']
 script='\n'.join(lines)+'\n'
 cmd=['sbatch','--parsable','--hold','--no-requeue','--account=lumia','--qos=normal','--job-name='+job(index),'--nodes=1','--ntasks=1','--gres=gpu:8','--cpus-per-task=64','--mem=600G','--time=2-00:00:00','--partition=ADA6000,L40S,RTX4090,RTX3090','--output='+str(r/'slurm-%j.log')]
 once(r/'submission-intent.json',dict(at=time.time(),job=plan['job'],plan_sha256=sha(path),command=cmd,script=script,no_auto_retry=True))
 result=subprocess.run(cmd,input=script,text=True,capture_output=True);once(r/'submission-response.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));result.check_returncode()
 jid=result.stdout.strip().split(';')[0];require(jid.isdigit(),'Invalid scheduler ID')
 sub=dict(at=time.time(),job_id=jid,job=plan['job'],root=str(r),output=plan['output'],plan=str(path),plan_sha256=sha(path),source=deployment['commit'],group='BranPO',size='3b',outer=150,repeat_index=index,repeat_count=3,checkpoint_identity=plan['checkpoint_identity'],gpus=8,no_auto_retry=True)
 once(r/'submission.json',sub)
 row=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True);require('JobName='+plan['job']+' ' in row and 'JobState=PENDING' in row and 'Reason=JobHeldUser' in row,'Unexpected held allocation')
 subprocess.check_call(['scontrol','release',jid]);print('SUBMISSION='+json.dumps(sub),flush=True);return sub
